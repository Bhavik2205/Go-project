from transformers import BertForSequenceClassification
import torch

print("✅ PyTorch ONNX Opset version:", torch.onnx._constants.ONNX_DEFAULT_OPSET)

model_name = "ProsusAI/finbert"
save_path = "models/sentiment.onnx"

# Load FinBERT model
model = BertForSequenceClassification.from_pretrained(model_name)
model.eval()

# Dummy input (batch size: 1, sequence length: 128)
dummy_input = {
    "input_ids": torch.randint(0, 100, (1, 128)),
    "attention_mask": torch.ones((1, 128), dtype=torch.long)
}

# Define wrapper to extract logits only (not ModelOutput object)
class FinBERTWrapper(torch.nn.Module):
    def __init__(self, model):
        super(FinBERTWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits  # Important: return only logits

# Wrap the model
wrapped_model = FinBERTWrapper(model)

# Export to ONNX
torch.onnx.export(
    wrapped_model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    save_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch_size"}, "attention_mask": {0: "batch_size"}, "logits": {0: "batch_size"}},
    do_constant_folding=True,
    opset_version=12  # or 11+ to be safe for ONNX Runtime compatibility
)

print(f"✅ FinBERT model exported correctly to {save_path}")
