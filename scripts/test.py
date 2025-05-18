import onnx

model = onnx.load("models/sentiment.onnx")
print("✅ Model outputs:")
for output in model.graph.output:
    print(f"  - {output.name} ({output.type})")