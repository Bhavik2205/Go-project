Download from: https://...

#steps to generate sentiment_optimized.onnx

1) pip install onnxruntime onnxruntime-tools
2) python -m onnxruntime.tools.optimizer_cli --help
3) python -m onnxruntime.tools.optimizer_cli \
  --input models/sentiment.onnx \
  --output models/sentiment_optimized.onnx \
  --optimize_level 99
