# RumbleGuide OpenVINO Setup and Usage Guide

Follow these steps to initialize the Docker container with the OpenVINO environment, convert your ONNX model to OpenVINO IR format, and run inference.

## Troubleshooting

- **Docker Issues:** Ensure Docker is installed, running, you have internet connection, and sufficient permissions.
- **Model Conversion Errors:** Verify that your ONNX model is valid (used opset versions 12 or lower). Even if it is, some layers may still be incompatible with OpenVINO 2020.3 (e.g. MaxPool if using opset version 10).
- **Script Errors:** Check that you are using the correct command-line arguments and file paths (Windows vs Unix).
- **Must have already ONNX model:** you should have already converted the model to ONNX (use ...)

And most importantly, good luck!

## Steps

### 1. Set Up RumbleGuideOpenVINO

- Clone the [RumbleGuideOpenVINO](https://github.com/sirbastiano/RumbleGuideOpenVINO/tree/main) repository to your local machine.
- Use the Makefile provided in this folder instead (move it to the RumbleGuideOpenVINO folder) (At least for running on NVIDIA Jetson, I needed it)
- Modify the `MOUNT` and `MYRIAD_UDEV_RULES` directories in the Makefile


### 2. Run the Docker Container

Open your terminal and change to the `RumbleGuideOpenVINO` folder:

```bash
cd RumbleGuideOpenVINO
```

Pull the latest changes using:

```bash
make pull
```

Start the Docker container, which initializes the OpenVINO environment and mounts the `mount_folder`:

```bash
make run
```

### 3. Convert an ONNX Model to OpenVINO IR Format

Inside the Docker container, convert your ONNX model (e.g., `your_model_name.onnx`) to OpenVINO IR (version 2020.3) by running:

```bash
python onnx_to_ov.py --model_file your_model_name.onnx
```

### 4. Run the Inference Engine

Once the model conversion is complete, run the inference engine.

```bash
python inference_engine.py --device MYRIAD --model_file your_model_name
```

##### Notes
-  Can also run on CPU by using argument `--device CPU`
-  Provide the base name of your model (without file extensions, as `.xml` and `.bin` will be appended later)
-  The OV files must be saved in the `mount_folder/ov` directory (automatically saved here if used `onnx_to_ov.py`)



## Additional Files
You can also run the following two auxiliary files.
##### `check_onnx_shapes.py` 
Check the shapes of the layers of the ONNX model

##### `calculate_inference_memory.py`
Calculate (using the PyTorch model and CUDA) the expected memory required to run the model in 16-bit precision. You can use the following arguments:

- `--save_to_onnx`: convert the model to ONNX
- `--model_size`: with options debugNano, nano, mini, tiny, xsmall, small, light, base, large, xlarge, xxlarge
- `--task`: segmentation or classification



---