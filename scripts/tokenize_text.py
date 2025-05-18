# scripts/tokenize_text.py

import sys
import json
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")

def tokenize(text):
    tokens = tokenizer(text, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    return {
        "input_ids": tokens["input_ids"][0].tolist(),
        "attention_mask": tokens["attention_mask"][0].tolist()
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    input_text = " ".join(sys.argv[1:])

    try:
        result = tokenize(input_text)
        print(json.dumps(result))  # ONLY this goes to stdout
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)
