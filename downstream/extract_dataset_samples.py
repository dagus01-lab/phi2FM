#!/usr/bin/env python3
"""
Script to extract samples from different downstream task datasets and save them
for visualization. Extracts images and labels from 5000 n-shot configurations
for specified tasks.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pickle
import cv2
from collections import OrderedDict

# Add parent directory to path for imports
parent_dir = str(Path(__file__).resolve().parent.parent) if '__file__' in globals() else str(Path().resolve().parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.load_data import load_data
from utils.training_utils import read_yaml
from training_script import get_models_pretrained, get_models
from training_script import MODELS_224, MODELS_224_r30

# Configuration
OUTPUT_BASE_DIR = "/home/gdaga/phi2FM/downstream/extracted_samples"
NUM_SAMPLES_PER_TASK = 100  # Number of samples to extract per task

# Task to dataset directory mapping
DATASET_DIRS = { 
    "fire": "/Data/fire_dataset/fire_dataset.zarr", 
    "burned_area": "/Data/lpl_burned_area/burned.zarr", 
    "clouds": "/Data/phisatnet_clouds/phisatnet_clouds.zarr", 
    "worldfloods": "/Data/worldfloods/worldfloods.zarr", 
}

# Task to configuration directory mapping
CONFIG_DIRS = {
    "fire": "args/finetune_FMs/fire",
    "burned_area": "args/finetune_FMs/lpl_burned_area",
    "clouds": "args/finetune_FMs/phisatnet_clouds",
    "worldfloods": "args/finetune_FMs/worldfloods",
}

# Class labels for each task
TASK_LABELS = {
    "fire": ['safe', 'fire', 'burnt', 'water'],
    "burned_area": ['Background', 'Burned Area', 'Clouds', 'Waterbodies'],
    "clouds": ['No cloud', 'Cloud', 'value2', 'value3', 'value4'],
    "worldfloods": ['Clouds', 'Land', 'Water'],
}

# Task output channels (for validation)
TASK_OUTPUT_CHANNELS = {
    'fire': 4, 
    'burned_area': 4, 
    'clouds': 5, 
    'worldfloods': 3
}

def find_5000_nshot_files():
    """
    Find the files containing sample IDs for 5000 n-shot configurations
    by looking through the experiment structure.
    """
    base_dir = Path('/Data/phi2FM_n_shot')
    nshot_files = {}
    
    for mode in ['lp']:  # Focus on linear probing mode
        mode_dir = base_dir / mode
        if not mode_dir.exists():
            continue
            
        for model_dir in mode_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            for task_dir in model_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                    
                task_name = task_dir.name
                if task_name not in DATASET_DIRS:
                    continue
                    
                # Look for nested task directory
                nested = task_dir / task_dir.name
                if not nested.exists():
                    continue
                
                # Find runs with 5000 n_shot
                for run_dir in nested.iterdir():
                    if not run_dir.is_dir():
                        continue
                        
                    artifacts_path = run_dir / 'artifacts.json'
                    if not artifacts_path.exists():
                        continue
                    
                    try:
                        with open(artifacts_path, 'r') as f:
                            data = json.load(f)
                    except (json.JSONDecodeError, IOError):
                        continue
                    
                    # Check if this is a 5000 n_shot experiment
                    n_shots = data.get("training_parameters", {}).get('n_shot')
                    if n_shots == 5000:
                        # Look for data partition files
                        data_files = list(run_dir.glob("*train*.pkl")) + list(run_dir.glob("*test*.pkl"))
                        if data_files and task_name not in nshot_files:
                            nshot_files[task_name] = str(run_dir)
                            print(f"Found 5000 n-shot config for {task_name}: {run_dir}")
                            break
    
    return nshot_files

def load_sample_ids_from_config(config_dir):
    """
    Load sample IDs from the 5000 n-shot configuration files.
    """
    config_path = Path(config_dir)
    
    # Look for pickle files containing sample IDs
    train_files = list(config_path.glob("*train*.pkl"))
    test_files = list(config_path.glob("*test*.pkl"))
    
    sample_ids = []
    
    for file_path in train_files + test_files:
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    sample_ids.extend(data)
                elif isinstance(data, dict) and 'indices' in data:
                    sample_ids.extend(data['indices'])
                elif isinstance(data, dict):
                    # Try to find sample indices in the dictionary
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0:
                            sample_ids.extend(value)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    return list(set(sample_ids))  # Remove duplicates

def preprocess_image_for_saving(tensor, channel_first=True, rgb_indices=None, quantile_range=(0.02, 0.98)):
    """
    Preprocess a multi-spectral tensor for RGB display and saving.
    Based on the logic from plot_inference_2.ipynb
    """
    import torch
    
    # Convert to tensor if needed and ensure it's on CPU
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    elif isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
    else:
        raise ValueError(f"Input must be numpy array or torch tensor, got {type(tensor)}")
    
    # Handle batch dimension
    if tensor.ndim == 4:  # [B,C,H,W] → [C,H,W] (take first sample)
        tensor = tensor[0]
    elif tensor.ndim == 2:  # [H,W] → return grayscale handling
        return _handle_grayscale(tensor, quantile_range)
    elif tensor.ndim != 3:
        raise ValueError(f"Expected 2D, 3D, or 4D tensor, got {tensor.ndim}D with shape {tensor.shape}")
    
    # Handle channel order
    if not channel_first and tensor.ndim == 3:
        tensor = tensor.permute(2, 0, 1)  # [H,W,C] → [C,H,W]
    
    # Default RGB indices (adjust based on your data)
    if rgb_indices is None:
        rgb_indices = [2, 1, 0]  # Assuming bands are ordered for Sentinel-2-like data
    
    # Check if we have enough channels for RGB
    if tensor.shape[0] < max(rgb_indices) + 1:
        if tensor.shape[0] == 1:
            return _handle_grayscale(tensor.squeeze(0), quantile_range)
        else:
            print(f"Warning: Only {tensor.shape[0]} channels available, need at least {max(rgb_indices)+1} for RGB")
            return None
    
    # Extract RGB channels and convert to [H,W,3]
    try:
        rgb = tensor[rgb_indices].permute(1, 2, 0).float()  # [H,W,3]
    except IndexError as e:
        print(f"Error extracting RGB channels {rgb_indices} from tensor with {tensor.shape[0]} channels: {e}")
        return None
    
    # Normalize using quantiles
    if rgb.numel() > 0:
        q_low = max(0.0, min(1.0, quantile_range[0]))
        q_high = max(0.0, min(1.0, quantile_range[1]))
        
        lo = torch.quantile(rgb, q_low)
        hi = torch.quantile(rgb, q_high)
        
        # Avoid division by zero
        if hi > lo:
            rgb = torch.clamp((rgb - lo) / (hi - lo), 0, 1)
        else:
            rgb = torch.zeros_like(rgb)
    else:
        rgb = torch.zeros_like(rgb)
    
    return rgb.numpy()

def _handle_grayscale(tensor_2d, quantile_range):
    """Helper function to handle grayscale images"""
    import torch
    
    img_gray = tensor_2d.float()
    q_low = max(0.0, min(1.0, quantile_range[0]))
    q_high = max(0.0, min(1.0, quantile_range[1]))
    
    if img_gray.numel() > 0:
        lo = torch.quantile(img_gray, q_low)
        hi = torch.quantile(img_gray, q_high)
        img_gray = torch.clamp((img_gray - lo) / (hi - lo + 1e-6), 0, 1)
    else:
        img_gray = torch.zeros_like(img_gray)
    
    # Convert grayscale to RGB by repeating channels
    rgb = torch.stack([img_gray, img_gray, img_gray], dim=-1)  # [H,W,3]
    return rgb.numpy()

def process_label_for_saving(label, task_name):
    """
    Process label for saving in one-hot encoded format.
    """
    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label)
    elif not isinstance(label, torch.Tensor):
        label = torch.tensor(label)
    
    label = label.cpu()
    
    # Convert to one-hot encoding if not already
    if task_name in ['fire', 'burned_area', 'clouds', 'worldfloods']:
        if label.ndim == 2:  # Segmentation map [H,W]
            num_classes = TASK_OUTPUT_CHANNELS[task_name]
            # Create one-hot encoding
            one_hot = torch.zeros(num_classes, label.shape[0], label.shape[1])
            one_hot.scatter_(0, label.unsqueeze(0).long(), 1)
            return one_hot.numpy()
        elif label.ndim == 3 and label.shape[0] == TASK_OUTPUT_CHANNELS[task_name]:
            # Already one-hot encoded
            return label.numpy()
        elif label.ndim == 1:  # Classification case
            num_classes = TASK_OUTPUT_CHANNELS[task_name]
            one_hot = torch.zeros(num_classes)
            if len(label) == 1:
                one_hot[label.long()] = 1
            else:
                one_hot = label.float()  # Assume it's already a probability vector
            return one_hot.numpy()
    
    return label.numpy()

def get_dataset_for_task(task_name, sample_model_name='seco'):
    """
    Get the dataset for a specific task using the same logic as in the notebook.
    """
    # Use a sample config file to get dataset parameters
    config_file = os.path.join(CONFIG_DIRS[task_name], f"{sample_model_name}.yml")
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    # Read config
    args = read_yaml(config_file)
    
    # Set parameters for dataset loading
    n_shot = 5000
    batch_size = 1  # We'll iterate one by one
    input_channels = args.input_channels
    output_channels = TASK_OUTPUT_CHANNELS[task_name]
    
    # Choose dataset path based on model requirements
    model_name = args.model_name
    if model_name in MODELS_224_r30:
        dataset_folder = args.data_path_224_30m
    elif model_name in MODELS_224:
        dataset_folder = args.data_path_224_10m
    else:
        dataset_folder = args.data_path_128_10m
    
    # Determine other parameters
    crop_images = True if model_name == 'phileo_precursor' or model_name == 'phileo_precursor_classifier' else False
    patch_size = (256, 256) if task_name != "fire" else None
    
    # Load dataset
    print(f"Loading dataset for task {task_name}...")
    weights, pos_weight, _, dl_test, _, _ = load_data(
        dataset_folder,
        with_augmentations=False,  # No augmentations for extraction
        num_workers=4,
        batch_size=batch_size,
        downstream_task=task_name,
        model_name=model_name.split('_')[0],
        device='cpu',  # Use CPU for data loading
        pad_bands=args.pad_bands,
        crop_images=crop_images, 
        num_classes=output_channels, 
        n=n_shot, 
        weights_dir=task_name, 
        patch_size=patch_size
    )
    
    return dl_test

def extract_samples_for_task(task_name, num_samples=NUM_SAMPLES_PER_TASK):
    """
    Extract samples for a specific task and save them.
    """
    print(f"\n=== Extracting samples for task: {task_name} ===")
    
    # Create output directories
    task_output_dir = Path(OUTPUT_BASE_DIR) / task_name
    img_dir = task_output_dir / "img"
    label_dir = task_output_dir / "label"
    
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get dataset
        dl_test = get_dataset_for_task(task_name)
        
        # Determine sample indices to extract
        dataset_size = len(dl_test.dataset)
        print(f"Dataset size: {dataset_size}")
        
        # Use evenly spaced indices
        if num_samples >= dataset_size:
            sample_indices = list(range(dataset_size))
        else:
            step = dataset_size // num_samples
            sample_indices = list(range(0, dataset_size, step))[:num_samples]
        
        print(f"Extracting {len(sample_indices)} samples...")
        
        # Extract and save samples
        for i, sample_idx in enumerate(tqdm(sample_indices, desc=f"Extracting {task_name}")):
            try:
                # Get sample
                sample = dl_test.dataset[sample_idx]
                img = sample['img']
                label = sample['label']
                
                # Process image for RGB display
                processed_img = preprocess_image_for_saving(img, channel_first=True, rgb_indices=[2, 1, 0])
                
                if processed_img is not None:
                    # Convert to 8-bit for saving
                    img_8bit = (processed_img * 255).astype(np.uint8)
                    
                    # Save image
                    img_filename = f"{task_name}_sample_{sample_idx:05d}.png"
                    img_path = img_dir / img_filename
                    cv2.imwrite(str(img_path), cv2.cvtColor(img_8bit, cv2.COLOR_RGB2BGR))
                
                # Process label for one-hot encoding
                processed_label = process_label_for_saving(label, task_name)
                
                # Save label
                label_filename = f"{task_name}_sample_{sample_idx:05d}.npy"
                label_path = label_dir / label_filename
                np.save(str(label_path), processed_label)
                
            except Exception as e:
                print(f"Error processing sample {sample_idx}: {e}")
                continue
        
        print(f"Successfully extracted samples for {task_name}")
        print(f"Images saved to: {img_dir}")
        print(f"Labels saved to: {label_dir}")
        
    except Exception as e:
        print(f"Error extracting samples for {task_name}: {e}")

def main():
    """
    Main function to extract samples from all specified tasks.
    """
    print("Dataset Sample Extraction Script")
    print("================================")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print(f"Samples per task: {NUM_SAMPLES_PER_TASK}")
    print(f"Tasks to process: {list(DATASET_DIRS.keys())}")
    
    # Create base output directory
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # Process each task
    for task_name in DATASET_DIRS.keys():
        try:
            extract_samples_for_task(task_name, NUM_SAMPLES_PER_TASK)
        except Exception as e:
            print(f"Failed to process task {task_name}: {e}")
            continue
    
    print("\n=== Extraction Complete ===")
    print(f"All samples saved to: {OUTPUT_BASE_DIR}")
    
    # Print summary
    print("\nSummary:")
    for task_name in DATASET_DIRS.keys():
        task_dir = Path(OUTPUT_BASE_DIR) / task_name
        if task_dir.exists():
            img_count = len(list((task_dir / "img").glob("*.png")))
            label_count = len(list((task_dir / "label").glob("*.npy")))
            print(f"  {task_name}: {img_count} images, {label_count} labels")

if __name__ == "__main__":
    main()