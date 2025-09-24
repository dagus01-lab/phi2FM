import subprocess
import os

# Convert encoder ONNX to OpenVINO
def convert_onnx_to_openvino(onnx_path, output_dir, model_name):
    """Convert ONNX model to OpenVINO format"""
    cmd = [
        "mo",
        "--input_model", onnx_path,
        "--output_dir", output_dir,
        "--model_name", model_name,
        # "--data_type", "FP16",  # Use FP16 for better performance
        "--compress_to_fp16"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Successfully converted {onnx_path}")
        print(f"Output files in {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {onnx_path}: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

# Set paths
onnx_encoder_path = "/home/gdaga/phi2FM/downstream/onnx/phisat2net_geoaware_best_encoder.onnx"
onnx_decoder_path = "/home/gdaga/phi2FM/downstream/onnx/phisat2net_geoaware_best_decoder.onnx"
openvino_output_dir = "/home/gdaga/phi2FM/downstream/openvino_models"

# Create output directory
os.makedirs(openvino_output_dir, exist_ok=True)

# Convert both models
convert_onnx_to_openvino(onnx_encoder_path, openvino_output_dir, "phisat_encoder")
convert_onnx_to_openvino(onnx_decoder_path, openvino_output_dir, "phisat_decoder")