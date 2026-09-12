#!/usr/bin/env python3
"""Fine-tune FinBERT on labelled news stored in the database.

Reads rows from `news_articles` that already carry a sentiment_label, fine-tunes
ProsusAI/finbert on them, and writes the result to models/sentiment_finetuned/.
Run scripts/convert_to_onnx.py afterwards to export it for the Go inference
path, or use `make ml-pipeline` to do both.

The script refuses to train on too little data rather than producing a model
that looks trained but has learned nothing: fine-tuning FinBERT on a few dozen
rows converges to predicting the majority class and reports a healthy-looking
accuracy while being useless.

Usage:
    python scripts/retrain_sentiment.py --dry-run
    python scripts/retrain_sentiment.py --epochs 3 --min-rows 500
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_MODEL = "ProsusAI/finbert"
OUTPUT_DIR = Path("models/sentiment_finetuned")
MANIFEST = Path("models/manifest.json")

# FinBERT's label order. Keep this aligned with what the Go inference path
# expects, or scores will be confidently wrong rather than obviously broken.
LABELS = ["positive", "negative", "neutral"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dsn", default="", help="Postgres DSN. Defaults to the DB_* env vars.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--min-rows", type=int, default=500,
                   help="Refuse to train on fewer labelled rows than this.")
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--limit", type=int, default=100_000)
    p.add_argument("--dry-run", action="store_true",
                   help="Report how much labelled data exists and exit.")
    return p.parse_args()


def dsn_from_env(explicit: str) -> str:
    if explicit:
        return explicit
    user = os.getenv("DB_USER")
    name = os.getenv("DB_NAME")
    if not user or not name:
        sys.exit("Set DB_USER and DB_NAME (or pass --dsn).")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    pwd = os.getenv("DB_PASSWORD", "")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{name}"


def load_rows(dsn: str, limit: int):
    """Fetch labelled articles. Returns a list of (text, label_id)."""
    try:
        import psycopg2
    except ImportError:
        sys.exit("psycopg2 is required: pip install psycopg2-binary")

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            # Title plus description is what the live pipeline scores, so train
            # on the same shape rather than on full article bodies.
            cur.execute(
                """
                SELECT COALESCE(title, '') || '. ' || COALESCE(description, ''),
                       LOWER(sentiment_label)
                FROM news_articles
                WHERE sentiment_label IS NOT NULL
                  AND sentiment_label <> ''
                  AND deleted_at IS NULL
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    out, skipped = [], 0
    for text, label in rows:
        if label not in LABEL_TO_ID or not text.strip():
            skipped += 1
            continue
        out.append((text.strip(), LABEL_TO_ID[label]))
    if skipped:
        print(f"  skipped {skipped} rows with an unusable label or empty text")
    return out


def class_counts(rows):
    counts = {name: 0 for name in LABELS}
    for _, label_id in rows:
        counts[LABELS[label_id]] += 1
    return counts


def write_manifest(entry: dict):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if MANIFEST.exists():
        try:
            data = json.loads(MANIFEST.read_text())
        except json.JSONDecodeError:
            print("  existing manifest was unreadable; replacing it")
    data.setdefault("history", []).append(entry)
    data["latest"] = entry
    MANIFEST.write_text(json.dumps(data, indent=2))
    print(f"manifest updated: {MANIFEST}")


def main():
    args = parse_args()
    dsn = dsn_from_env(args.dsn)

    print(f"Loading labelled articles (limit {args.limit})...")
    rows = load_rows(dsn, args.limit)
    counts = class_counts(rows)

    print(f"  usable rows : {len(rows)}")
    for name, n in counts.items():
        print(f"  {name:<9}: {n}")

    if args.dry_run:
        print("\n--dry-run: nothing was trained.")
        if len(rows) < args.min_rows:
            print(f"Not enough data: {len(rows)} rows, need at least {args.min_rows}.")
        return

    if len(rows) < args.min_rows:
        sys.exit(
            f"Refusing to train on {len(rows)} rows (minimum {args.min_rows}).\n"
            "Fine-tuning on this little data produces a model that predicts the\n"
            "majority class and still reports a plausible accuracy. Score more\n"
            "articles first, or lower --min-rows deliberately."
        )

    present = [n for n, c in counts.items() if c > 0]
    if len(present) < 2:
        sys.exit(
            f"Only one class present ({present}). A classifier trained on a single\n"
            "label cannot learn anything useful."
        )

    # Imported late so --dry-run works without the ML stack installed.
    try:
        import torch
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoTokenizer,
            BertForSequenceClassification,
            get_linear_schedule_with_warmup,
        )
    except ImportError as e:
        sys.exit(f"Missing ML dependency: {e}\npip install -r requirements.txt")

    texts = [t for t, _ in rows]
    labels = [l for _, l in rows]

    # Stratify so a rare class does not vanish from the validation split.
    train_x, val_x, train_y, val_y = train_test_split(
        texts, labels, test_size=args.val_split, random_state=42, stratify=labels
    )
    print(f"\nTrain {len(train_x)} / validate {len(val_x)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = BertForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=len(LABELS))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")

    class NewsDataset(Dataset):
        def __init__(self, x, y):
            self.x, self.y = x, y

        def __len__(self):
            return len(self.x)

        def __getitem__(self, i):
            enc = tokenizer(
                self.x[i],
                truncation=True,
                padding="max_length",
                max_length=args.max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.y[i], dtype=torch.long),
            }

    train_loader = DataLoader(NewsDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(NewsDataset(val_x, val_y), batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += out.loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += preds.size(0)

        acc = correct / total if total else 0.0
        best_acc = max(best_acc, acc)
        print(f"epoch {epoch}/{args.epochs}  loss {running / len(train_loader):.4f}  val_acc {acc:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nmodel saved to {OUTPUT_DIR}")

    write_manifest({
        "stage": "finetune",
        "base_model": BASE_MODEL,
        "output_dir": str(OUTPUT_DIR),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rows_total": len(rows),
        "rows_train": len(train_x),
        "rows_val": len(val_x),
        "class_counts": counts,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "val_accuracy": round(best_acc, 4),
        "labels": LABELS,
    })

    print(f"\nNext: python scripts/convert_to_onnx.py --model-dir {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
