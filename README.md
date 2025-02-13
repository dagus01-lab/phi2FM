# ΦsatNet

**You can find a better description of each part of the code in the README inside each folder (`pretrain`, `downstream`, `data_simulation`)**

## Prerequisites

- To run the code, you should install this repo as a module in Python.

- To install as module, run from the top directory of the repository (/phi2FM) `pip install -e .`

- Now you are able to use `import pretrain`, `import downstream`, `import data_simulation`

- You can also use deeper imports (e.f. `from downstream.models.phisatnet_downstream import PhiSatNetDownstream`)

## 1. Simulate Data
The folder `data_simulation` contains the scripts to simulate Φsat-2 data from L1C data. It contains:

- `phileo-bench`: to simulate Φsat-2 data from PhilEO-Bench dataset
- `pretraining`: to download MajorTOM pretraining data and convert it to Φsat-2 data 
- `simulator`: contains the main logic for running the simulator (except the simulator workflow itself, which is described in either `pretraining` or `phileo-bench`).
- `tiff_to_np_patches`: the other scripts create tiff files of Φsat-2 data of a big shape (e.g. 2048x2048). This folder divides these files into smaller `.npy` patches to feed into the model (e.g. 256x256).


## 2. Pretrain

Runs the pretraining. This folder is pretty easy to run, as it just requires to change the `.yml` file with your directories, and number of GPUs to use.

## 3. Downstream

Run downstream experiments (right now only for PhilEO-Bench).

- If want to add new datasets, modify `load_data.py`, `data_protocol.py` and `visualize.py`
- If want to add new models, modify `models` folder, and add it to the `training_script.py` (the main file running everything).


## 4. Myriad

Explains how to convert the model to OpenVINO 2020.3

- The conversion to ONNX should happen in the `downstream` folder tho (see `convert_to_onnx` function under `downstream/utils/utils.py`)



