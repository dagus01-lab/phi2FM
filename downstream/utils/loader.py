from data_loader import get_zarr_dataloader, NormalizeChannels
from tqdm import tqdm
import torch
import numpy as np 

# Path to the input Zarr dataset
zarr_path = "/Data/worldfloods/worldfloods.zarr"
# Select dataset split: "trainval" or "test"
dataset_set = "trainval"

# Step 1: Compute dataset-wide per-band mean and std
print("Computing mean and std across dataset...")
_, _, dataloader = get_zarr_dataloader(
    zarr_path=zarr_path,
    dataset_set=dataset_set,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    task_filter="segmentation",
    metadata_keys=["sensor", "timestamp", "geolocation", "crs"],
    num_classes=4
)

sum_ = 0
sum_sq = 0
total_pixels = 0

for batch in tqdm(dataloader, desc="Computing stats"):
    for task in batch['tasks']:
        images = np.array(batch[f'{task}_img'])  # shape: (B, H, W, C)
        if images.ndim != 4:
            raise ValueError("Expected image tensor of shape (B, H, W, C)")
        batch_size, height, width, _ = images.shape
        pixels_in_batch = batch_size * height * width

        sum_ += images.sum(axis=(0, 1, 2))
        sum_sq += (images ** 2).sum(axis=(0, 1, 2))
        total_pixels += pixels_in_batch

mean = sum_ / total_pixels
std = np.sqrt((sum_sq / total_pixels) - (mean ** 2))

print("Mean per band:", mean.tolist())
print("Stddev per band:", std.tolist())
