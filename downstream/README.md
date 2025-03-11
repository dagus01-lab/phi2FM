# 3. Model Benchmarking - Downstream Tasks

Example of how to run:

`python training_script.py -r args/phimultigpu/dino.yml`

## Changes I did to PhilEO-Bench
- Land cover classification is now multi-label land cover classification
- When doing few-shot, ensure the distribution of values is the same for every possibl n-shot, while still balancing the data as much as possible (e.g. if almost no patches contain buildings, no much learning will occur. What I changed is maximizing building in patches to encourage learning while keeping the same distribution in the amount of buildings for every n-shot).

## Models

Contains the architecture of different models tested. Right now, it has the architectures (both in segmentation and classification) for:
- Prithvi 1.0
- SatMAE
- SeCo
- Moco (ssl4eo-s12)
- DINO (ssl4eo-s12)
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

Below is a concise, step-by-step overview of the `training_script.py` and what each section does. The `training_script.py` in the pretraining is very similar (a bit simpler).

### **High-level flow**

1. **`if __name__ == "__main__":`**  
   - Reads the YAML file and invokes the `main` function.  
   - Currently, there's an `if True:` block that loops over different `n_shot` values, runs linear probing or full finetuning, and switches between classification and segmentation tasks. If you prefer to stick to the default (run `main` with YAML arguments), change this to `if False:`.

2. **YAML arguments**  
   - All arguments for `main` come directly from the YAML file.  






### **Inside `main`**

1. **Multi-GPU Setup**  
   - Sets up distributed or data parallel training (DDP or DP).  
   - *Note*: DDP was fully implemented for pretraining but not for downstream tasks (i.e. this file).
     - You can check the `training_loops.py` in the pretraining as it is just a few changes that must be done (setting the data sampler arguments at each epoch, and not calling the model again to generate the images to plot).
     - Also need to ensure that the data is in a correct format for DDP, and change the dataloaders to use data sampler.

2. **Define the model**  
   - Loads the chosen model (pretrained or randomly initialized).
   - There is an option (`downstream_model_path`) to load entire downstream model weights instead of just feature extractor. Useful for debugging.
   - Works for Prithvi 1, SatMAE, phisatnet, and the three pretrained PhilEO-Bench models (two U-Nets and one ViT).  
   - Afterwards, it wraps the model in DP or DDP if needed, and prints a summary.
   - If want to modify these, look at the function `get_models_pretrained`. `get_models` could also be used but I don't think is interesting since the weights are random (it is more for pretraining).



3. **Construct Output Folder**  
   - Uses the base output path, current date, experiment name, and data partition protocol (e.g., `n_shot` or `split_ratio`).


4. **Load Datasets**  
   - Asserts the number of channels needed for the task and that `n_shot` and `split_ratio` aren’t both used at once.  
   - Chooses the data folder (e.g., 128 vs. 224 resolution) to read data from.  
   - Gets datasets as memmaped buteo arrays, prints their shapes, and sets up the corresponding data loaders.
   - If `n_shot` is 0, it switches from training to inference mode (since there’s effectively no training set). It needs to have at least n_shot = 1 for script to work but it won't be used. Also, `.inference` function is used and not `.test` because the latter requires to use `.train` first.
   - If want to debug or use another code, there's an option to return already here the main function with the dataloaders (useful for debugging).
   - `by_region` is by default True. If wanting to use `False` (probably the case for other downstream tasks), should create a .csv with the file names.

5. **Initialize the Trainer**  
   - Configures learning rate (only exponential warmup currently implemented, linear warmup with `min_lr` was used during pretraining but I used the lr of PhilEO-Bench in downstream).  
   - Initializes the trainer from `training_loops.py`.

6. **Training / Testing / Inference Workflow**  
   - Depending on `train_mode`, calls the appropriate trainer functions (`train`, `test`, `inference`) in sequence.

6. **6. Training / testing / inference workflow**: essentially calls the training, testing, and/or inference functions of the trainer as desired.

7. **Finish Script**  
   - Saves all parameters to a YAML in the output folder.  
   - If using DDP, runs cleanup routines.


