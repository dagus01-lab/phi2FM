import onnx

# Load your ONNX model
model_path = "model_nano.onnx"  # Replace with your ONNX model file path
model = onnx.load(model_path)

# Run shape inference
print(f"Running shape inference for: {model_path}")
inferred_model = onnx.shape_inference.infer_shapes(model)

# Check shapes in the graph
print("Shapes of intermediate values:")
for value_info in inferred_model.graph.value_info:
    shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
    print(f"{value_info.name}: {shape}")

# Optionally, save the inferred model to inspect later
onnx.save(inferred_model, "model_nano_inferred.onnx")
print("Inferred model saved as: model_nano_inferred.onnx")
