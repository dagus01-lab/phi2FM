from data_loader2 import get_zarr_dataloader, NormalizeChannels
from tqdm import tqdm
import torch
import zarr

# Path to the input Zarr dataset
#zarr_path = "/Data/fire_dataset/fire_dataset.zarr"
zarr_path = "/Data/worldfloods/worldfloods.zarr"
# Select dataset split: "trainval" or "test"
dataset_set = "trainval"
#zarr.open(zarr_path)
# Initialize a PyTorch DataLoader from a Zarr-based dataset
_, _, dataloader = get_zarr_dataloader(
    zarr_path=zarr_path,                     # Path to the Zarr archive
    dataset_set=dataset_set,                 # Dataset subset to use
    batch_size=16,                           # Number of samples per batch
    shuffle=True,                            # Enable shuffling (useful for training)
    num_workers=4,                           # Number of parallel workers for loading
    #transform=NormalizeChannels(min_max=True),  # Normalize input channels to [0, 1]
    task_filter="segmentation",              # Only load data for the "segmentation" task
    metadata_keys=["sensor", "timestamp", "geolocation", "crs"],   # Include auxiliary metadata fields
)


all_unique_labels = set()

try:
    for idx, batch in enumerate(tqdm(dataloader, desc="Processing Batches")):
        for task in batch['tasks']:
            labels = batch[f'{task}_label']  # Might be shape (B, H, W) or list of scalars
    
            # Case 1: If labels is a tensor (e.g. B x H x W)
            if isinstance(labels, torch.Tensor):
                unique_vals = torch.unique(labels)
                all_unique_labels.update(unique_vals.cpu().numpy().tolist())
    
            # Case 2: If labels is a list/array of scalars
            elif isinstance(labels, (list, tuple)):
                for label in labels:
                    if isinstance(label, torch.Tensor):
                        unique_vals = torch.unique(label)
                    else:
                        # If label is scalar (e.g. float32), wrap in tensor first
                        label_tensor = torch.tensor(label)
                        unique_vals = torch.unique(label_tensor)
    
                    all_unique_labels.update(unique_vals.cpu().numpy().tolist())
    
            else:
                raise TypeError(f"Unexpected label type: {type(labels)}")
except Exception as e:
    print(e)

# Final result
print(f"\nAll unique label values seen across all batches and tasks: {sorted(all_unique_labels)}")
print(f"Total number of classes: {len(all_unique_labels)}")
