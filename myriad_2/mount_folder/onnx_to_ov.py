import subprocess
import argparse


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run OpenVINO Model Optimizer with a specified ONNX model.")
parser.add_argument(
    "-m", "--model_file",
    type=str,
    default="model_nano.onnx",
    help="Name of the ONNX model file to optimize (default: model_nano.onnx)"
)
args = parser.parse_args()

# Define the paths
base_folder = '/home/mount'
model_file = args.model_file
model_path = f"{base_folder}/onnx/{model_file}"
output_dir = f"{base_folder}/ov/"
mo_script_path = "/opt/intel/openvino/deployment_tools/model_optimizer/mo.py"

# Specify the input shape
input_shape = "[1,8,224,224]"  # Include brackets and ensure no spaces within the brackets - include this because the model has dynamic input shape (batch size), and OpenVINO cannot handle it

# Run the Model Optimizer command with the active Python environment
subprocess.run([
    "python3", mo_script_path,
    "--input_model", model_path,
    "--data_type", "FP16",
    "--output_dir", output_dir,
    "--input_shape", input_shape
], check=True)
