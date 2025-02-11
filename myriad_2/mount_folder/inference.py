import numpy as np
import argparse
from openvino.inference_engine import IECore

def main(device, model_file):
    # Initialize the Inference Engine
    ie = IECore()

    # Construct paths to the model files
    model_xml = f"/home/mount/ov/{model_file}.xml"
    model_bin = f"/home/mount/ov/{model_file}.bin"
    print(f"Using model_file: {model_file}")

    # Read the network
    net = ie.read_network(model=model_xml, weights=model_bin)

    # Get the input and output blobs
    input_blob = next(iter(net.inputs))
    output_blob = next(iter(net.outputs))

    # Display input and output information
    print(f"Input blob name: {input_blob}")
    print(f"Input blob shape: {net.inputs[input_blob].shape}")
    print(f"Output blob name: {output_blob}")
    print(f"Output blob shape: {net.outputs[output_blob].shape}")

    # Create arbitrary input data with the required shape
    input_shape = net.inputs[input_blob].shape
    dummy_input = np.random.rand(*input_shape).astype(np.float16)
    print(f"INPUT SHAPE: {dummy_input.shape}")

    # Load the model to the specified device
    exec_net = ie.load_network(network=net, device_name=device)
    print("Network loaded")

    # Run inference
    res = exec_net.infer(inputs={input_blob: dummy_input})

    # Extract and print the output
    output = res[output_blob]
    print(f"Output shape: {output.shape}")
    print(f"Output data:\n{output}")

    print(f"Model was run correctly on {device}!!! Best day of my life!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a model using OpenVINO.')
    parser.add_argument('-m', '--model_file', default='model',
                        help='Base name of the model (no .xml/.bin). Default is "model".')
    parser.add_argument('-d', '--device', default='CPU', choices=['CPU', 'MYRIAD'],
                        help='Specify the device to run the model on, e.g., CPU or MYRIAD.')
    args = parser.parse_args()
    main(args.device, args.model_file)
