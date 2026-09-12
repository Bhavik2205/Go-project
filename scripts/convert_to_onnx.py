#!/usr/bin/env python3
"""Export a FinBERT sentiment model to ONNX for the Go inference path.

Exports the base ProsusAI/finbert by default, or a fine-tuned directory
produced by scripts/retrain_sentiment.py via --model-dir.

Writes models/manifest.json so the /api/v1/models endpoint can report where the
deployed model came from. Without that, an .onnx file on disk is anonymous and
there is no way to tell a fine-tuned model from the stock one.

Usage:
    python scripts/convert_to_onnx.py
    python scripts/convert_to_onnx.py --model-dir models/sentiment_finetuned
    python scripts/convert_to_onnx.py --out models/sentiment_optimized.onnx
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import BertForSequenceClassification

MANIFEST = Path("models/manifest.json")
DEFAULT_BASE = "ProsusAI/finbert"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-dir", default=DEFAULT_BASE,
                   help="HuggingFace id or a local fine-tuned directory.")
    p.add_argument("--out", default="models/sentiment.onnx")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--opset", type=int, default=12,
                   help="ONNX opset. 11+ keeps ONNX Runtime happy.")
    return p.parse_args()


class FinBERTWrapper(torch.nn.Module):
    """Returns bare logits.

    torch.onnx.export cannot serialise the ModelOutput dataclass HuggingFace
    returns, so the forward pass is unwrapped here.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_manifest(entry: dict):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if MANIFEST.exists():
        try:
            data = json.loads(MANIFEST.read_text())
        except json.JSONDecodeError:
            print("  existing manifest was unreadable; replacing it")
    data.setdefault("history", []).append(entry)
    data["latest_export"] = entry
    MANIFEST.write_text(json.dumps(data, indent=2))
    print(f"manifest updated: {MANIFEST}")


def main():
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("PyTorch ONNX default opset:", torch.onnx._constants.ONNX_DEFAULT_OPSET)
    print(f"Loading {args.model_dir}...")

    model = BertForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    dummy_ids = torch.randint(0, 100, (1, args.max_length))
    dummy_mask = torch.ones((1, args.max_length), dtype=torch.long)

    torch.onnx.export(
        FinBERTWrapper(model),
        (dummy_ids, dummy_mask),
        str(out),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        # Batch stays dynamic so the Go side can score several articles at once.
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        do_constant_folding=True,
        opset_version=args.opset,
    )

    size = out.stat().st_size
    print(f"exported to {out} ({size / (1024 * 1024):.1f} MB)")

    write_manifest({
        "stage": "export",
        "source_model": args.model_dir,
        "is_finetuned": args.model_dir != DEFAULT_BASE,
        "onnx_path": str(out),
        "size_bytes": size,
        "sha256": sha256(out),
        "opset": args.opset,
        "max_length": args.max_length,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    main()
