import argparse
import torch
import onnxruntime as ort
import numpy as np

# === Parse command-line arguments ===
def parse_args():
    parser = argparse.ArgumentParser(description="Script to test memory usage and ONNX conversion.")
    parser.add_argument("-s", "--save_to_onnx", type=str, default="True",
                        help="Whether to save to ONNX (True/False).")
    parser.add_argument("-m", "--model_size", type=str, default="nano",
                        help="Model size, e.g. nano, mini, tiny, small, light, base.")
    parser.add_argument("-t", "--task", type=str, default="segmentation",
                        help="Task name, e.g. segmentation, classification, etc.")
    return parser.parse_args()


def main():
    # === Parse args ===
    args = parse_args()
    
    # Convert the string argument for `save_to_onnx` into a bool
    save_to_onnx = args.save_to_onnx.lower() == "true"
    model_size = args.model_size
    task = args.task

    # ==== The rest of your script (with minor modifications to use the variables) ====
    
    from models.model_foundation.model_foundation_local_rev2 import get_phisat2_model

    torch.cuda.reset_peak_memory_stats()  # Reset peak memory metrics
    memory_initial = torch.cuda.memory_allocated()

    fp16 = True
    input_size = 256
    batch_size = 1

    # Define a dummy data generator
    def generate_dummy_data(batch_size, channels, height, width, half=False, device='cuda'):
        if half:
            return torch.randn(batch_size, channels, height, width).half().to(device)
        else:
            return torch.randn(batch_size, channels, height, width).to(device)

    # Load the model with the specified size
    model = get_phisat2_model(
        model_size=model_size,
        return_model=f'downstream_{task}',
        input_dim=8,
        img_size=input_size
    )
    model.eval()

    for name, module in model.named_modules():
        if module.training:
            raise ValueError(f"Model is in training mode! Module: {name}")

    if save_to_onnx:
        print("Saving model to ONNX format...")
        model_path = f"onnx/model_{model_size}.onnx"
        dummy_input = generate_dummy_data(1, 8, input_size, input_size, half=False, device='cpu')
        print(f"Dummy input shape: {dummy_input.shape}")
        model.eval()
        # torch.onnx.export(model, dummy_input, model_path, opset_version=9, verbose=False, dynamic_axes={})
        torch.onnx.export(
            model,
            dummy_input,
            model_path,
            opset_version=9,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=None,  # or just omit
        )

        print(f"Starting Comparison PyTorch vs ONNX models...")

        # Load ONNX model using ONNX Runtime
        ort_session = ort.InferenceSession(model_path)

        # Function to run ONNX inference
        def run_onnx_inference(session, input_data):
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            outputs = session.run([output_name], {input_name: input_data.numpy()})
            return outputs[0]

        # Run several times
        for i in range(3):
            dummy_input = generate_dummy_data(batch_size, 8, input_size, input_size, half=False, device='cpu')

            # Run ONNX model inference
            onnx_output = run_onnx_inference(ort_session, dummy_input)

            # Run PyTorch model inference
            with torch.no_grad():
                output = model(dummy_input)

            # Convert PyTorch output to match ONNX output for comparison
            torch_output_np = output.cpu().numpy()

            # Comparing outputs
            if np.allclose(torch_output_np, onnx_output, atol=1e-4, rtol=1e-2):
                print(f"Test {i}: Test passed!")
            else:
                diff = np.abs(torch_output_np - onnx_output)
                print(f"Test {i}: Mismatch!  --- Absolute diff: {np.max(diff):.2e} , "
                      f"Relative diff: {np.max(diff / (np.abs(torch_output_np) + 1e-8)):.2e}")

    # Ensure model is in evaluation mode and moved to the CUDA device
    if fp16:
        model.half().to('cuda')
    else:
        model.to('cuda')

    # Memory profiling before running the model
    torch.cuda.reset_peak_memory_stats()  # Reset to track new peak memory stats
    memory_before = torch.cuda.memory_allocated()

    # Generate and run the model on multiple batches
    for _ in range(10):  # Adjust the number of runs as needed
        dummy_input = generate_dummy_data(batch_size, 8, input_size, input_size, half=fp16)
        with torch.no_grad():
            output = model(dummy_input)

    # Memory profiling after running the model
    memory_after = torch.cuda.memory_allocated()
    peak_memory_during = torch.cuda.max_memory_allocated()  # Get peak memory

    # Calculate the memory usage and return the difference
    memory_usage = memory_after - memory_before
    peak_memory_usage = peak_memory_during

    from tabulate import tabulate

    # Define the headers and the data rows
    headers = ["Description", "Memory Usage (MB)"]
    data = [
        ["Model Size", model_size],
        ["Peak Memory Usage", peak_memory_usage / 1024**2],
        ["Memory Usage During Inference", memory_usage / 1024**2],
        ["Memory for Model Parameters", memory_before / 1024**2]
    ]

    # Create and print the table
    table = tabulate(data, headers=headers, tablefmt="grid")
    print(table)

    if memory_initial > 0:
        raise MemoryError(f"Memory Leak Detected! Initial memory: {memory_initial / 1024**2:.2f} MB")

    print("Memory usage test finished!")


if __name__ == "__main__":
    main()
