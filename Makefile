.PHONY: run shell build test seed seed-dry ml-train ml-train-dry ml-export ml-pipeline ml-status

run:
	docker build -t ml-bot .
	docker run --rm -it --name ml-bot ml-bot

shell:
	docker run --rm -it -v $(PWD):/app --name ml-bot-dev ml-bot bash

# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------

# internal/model needs cgo for the ONNX runtime and does not build on hosts
# without it, so it is excluded here rather than failing the whole build.
GO_PKGS = $(shell go list ./... | grep -v "internal/model$$")

build:
	go build $(GO_PKGS)

test:
	go test $(GO_PKGS)

# ---------------------------------------------------------------------------
# Database seeding
# ---------------------------------------------------------------------------

seed-dry:
	go run ./cmd/seed --dry-run

seed:
	go run ./cmd/seed

# ---------------------------------------------------------------------------
# ML pipeline
#
# The full path is: score news -> fine-tune on those labels -> export to ONNX.
# Each step writes models/manifest.json, which /api/v1/models reports, so the
# deployed .onnx can always be traced back to what produced it.
# ---------------------------------------------------------------------------

# Reports how much labelled data exists without training. Safe to run anywhere.
ml-train-dry:
	python scripts/retrain_sentiment.py --dry-run

# Fine-tunes FinBERT on labelled rows in news_articles.
ml-train:
	python scripts/retrain_sentiment.py

# Exports the stock FinBERT to ONNX.
ml-export:
	python scripts/convert_to_onnx.py

# Fine-tune, then export the fine-tuned model over the deployed one.
ml-pipeline:
	python scripts/retrain_sentiment.py
	python scripts/convert_to_onnx.py --model-dir models/sentiment_finetuned

# Shows what is currently deployed.
ml-status:
	@cat models/manifest.json 2>/dev/null || echo "No models/manifest.json yet. Run 'make ml-export' first."
