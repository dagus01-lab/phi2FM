# 3. Model Benchmarking - Downstream Tasks

## Changes I did to PhilEO-Bench
- Land cover classification is now multi-label land cover classification
- When doing few-shot, ensure the distribution of values is the same for every possibl n-shot, while still balancing the data as much as possible (e.g. if almost no patches contain buildings, no much learning will occur. What I changed is maximizing building in patches to encourage learning while keeping the same distribution in the amount of buildings for every n-shot).

## Models

Contains the architecture of different models tested. Right now, it has the architectures (both in segmentation and classification) for:
- Prithvi 1.0
- SatMAE
- A pretrained U-Net (GeoAware)
- Another pretrained U-Net (uniphi)
- A pretrained ViT
- **phisatnet**

## Utils

#### `data_protocol.py`

- Quite specific for PhilEO-Bench, but if desired could also be implemented for other tasks if saved appropriately (`.npy` files of shape (N, C, H, W), with N being any number, even different for each file).
- Loads as a memmemaped array using Buteo all the dataset.
- I modified it to be "more fair" when choosing the for land cover classification, and building (both segmentation and classification) tasks
- Imagine we have the data saved as `.npy` files of shape (15, 8, 128, 128). This script combines all files to a memmaped `x_train` of shape (34000, 8, 128, 128), so it is easier to loop in the DataLoader.

#### `load_data.py`

- Quite specific for PhilEO-Bench, since it converts the memmaped buteo arrays to a Dataloader.
- Data preprocessing occurs here. Look at `callback_preprocess_phisatnet`. If want a more general view of preprocessing, look instead at the `load_data.py` function in the `pretrain` folder (specifically at `TransformX`).

#### `training_loops.py`

- Manages the training, validation, plot creation, etc.
- I think it should not need much modification for different downstream tasks.

#### `visualize.py`

- Manages output visualization within the `training_loops.py`
- Should be modified or removed since it is extremely specific to the tasks of PhilEO-Bench.


#### Other files

Just auxiliary files required by those explained above. 

## `training_script.py`

Main file that essentially runs everything. Here I do an overview in order of execution so it is a bit easier to understand. Note that 

1. `if __name__ == "__main__":` reads the yaml file, and then runs the main function. You can see that there is an if statement set to `True` which loops over many n_shot, linear probing and full finetuning, and classification of segmentation. I basically used it to be able to run all the experiments without creating a YAML file for each. I guess for other datasets or tasks you should change it. **If want to use the default behavior (run `main` function with args of YAML file, simply set `if False:`).**

2. The arguments of the main function all come from the YAML file.

3. **Multi GPU Setup**: the first part creates the setup if using multi-GPU. Please note that while I implemented the pretraining with DDP, I did not do it for the downstream tasks. You can use DP tho. And implmenting DDP should just require to change `training_loops.py` as the `training_script.py` is already fine for DDP. You can check the `training_loops.py` in the pretraining as it is just a few changes that must be done (setting the data sampler arguments at each epoch, and not calling the model again to generate the images to plot). Also need to ensure that the data is in a correct format for DDP, and change the dataloaders to use data sampler.

4. Next, some assertions are done to ensure the number of channels is correct (you should change this assertion if adding new downstream tasks), and another assertion to ensure not using both n_shot and split_ratio


