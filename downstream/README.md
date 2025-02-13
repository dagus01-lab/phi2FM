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





