#!/usr/bin/env python3
"""
Simple script to extract samples from downstream task datasets.
Extracts images and labels for visualization, using the same processing
logic as the plot_inference_2.ipynb notebook.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Add parent directory to path for imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.load_data import load_data
from utils.training_utils import read_yaml

# ==================== CONFIGURATION ====================

# Output directory (change this to your desired location)
OUTPUT_BASE_DIR = "/home/gdaga/phi2FM/downstream/extracted_samples"

# Number of samples to extract per task
NUM_SAMPLES_PER_TASK = 50

# Task configurations
TASKS = {
    "fire": {
        "dataset_dir": "/Data/fire_dataset/fire_dataset.zarr",
        "config_file": "args/finetune_FMs/fire/seco.yml",
        "output_channels": 4,
        "labels": ['safe', 'fire', 'burnt', 'water'], 
        "classes": ['safe', 'fire', 'burnt', 'water'],
        "num_samples": 10  # Reduced for testing
    },
    "burned_area": {
        "dataset_dir": "/Data/lpl_burned_area/burned.zarr",
        "config_file": "args/finetune_FMs/lpl_burned_area/seco.yml", 
        "output_channels": 4,
        "labels": ['Background', 'Burned Area', 'Clouds', 'Waterbodies'], 
        "classes": ['Background', 'Burned Area', 'Clouds', 'Waterbodies'],
        "num_samples": 150
    },
    "clouds": {
        "dataset_dir": "/Data/phisatnet_clouds/phisatnet_clouds.zarr",
        "config_file": "args/finetune_FMs/phisatnet_clouds/seco.yml",
        "output_channels": 5,  # After aggregation: 2 classes (no clouds, clouds)
# 0,1 -> 0 (no clouds)
# 2,3,4 -> 1 (clouds)
        "labels": ['c1', 'c2', 'c3', 'c4', 'c5'], 
        "classes": ['c1', 'c2', 'c3', 'c4', 'c5'], #'No clouds', 'Clouds'],  # This will trigger aggregation
        "num_samples": 100
    },
    "worldfloods": {
        "dataset_dir": "/Data/worldfloods/worldfloods.zarr",
        "config_file": "args/finetune_FMs/worldfloods/seco.yml",
        "output_channels": 3,
        "labels": ['Clouds', 'Land', 'Water'], 
        "classes": ['Clouds', 'Land', 'Water'],
        "num_samples": 150
    },
    "anomaly_detection": {
        "dataset_dir": "/Data/anomaly_detection/marine_area_dataset.zarr",
        "config_file": "args/finetune_FMs/anomaly_detection/seco.yml",
        "output_channels": 9,
        "labels": ['NO DATA', 'CLEAR WATER', 'TURBID WATER', 'LAND', 'PLASTIC', 'OIL', 'ALGAE', 'SEDIMENTS', 'CLOUD'],
        "classes": ['NO DATA', 'CLEAR WATER', 'TURBID WATER', 'LAND', 'PLASTIC', 'OIL', 'ALGAE', 'SEDIMENTS', 'CLOUD'],
        "num_samples": 150
    },
    "lancover_classification": {
        "dataset_dir": "/Data/phisatnet/phileo-bench_lc.zarr",
        "config_file": "args/finetune_FMs/phileo_bench-lc/seco.yml",
        "output_channels": 11,
        "labels": [    
            "Tree Cover",
            "Shrubland",
            "Grassland",
            "Cropland",
            "Built-up",
            "Bare/Sparse Vegetation",
            "Snow and Ice",
            "Permanent Water",
            "Herbaceous Wetland",
            "Mangroves",
            "Moss and Lichen",
        ],
        "classes": [    
            "Tree Cover",
            "Shrubland",
            "Grassland",
            "Cropland",
            "Built-up",
            "Bare/Sparse Vegetation",
            "Snow and Ice",
            "Permanent Water",
            "Herbaceous Wetland",
            "Mangroves",
            "Moss and Lichen",
        ],
        "num_samples": 100
    },
    "building_regression": {
        "dataset_dir": "/Data/phisatnet/phileo-bench_building.zarr",
        "config_file": "args/finetune_FMs/anomaly_detection/seco.yml",
        "output_channels": 1,
        "labels": None,
        "classes": None,  # No preprocessing, return original labels
        "task_type": "regression",  # Identify as regression task
        "num_samples": 100  # Reduced for testing
    },
    "roads_regression": {
        "dataset_dir": "/Data/phisatnet/phileo-bench_roads.zarr",
        "config_file": "args/finetune_FMs/anomaly_detection/seco.yml",
        "output_channels": 1,
        "labels": None,
        "classes": None,  # No preprocessing, return original labels
        "task_type": "regression",  # Identify as regression task
        "num_samples": 100  # Reduced for testing
    }
}

# ==================== HELPER FUNCTIONS ====================

# Color mapping from create_colored_labels.py
COLOR_MAPS = {
    0 : (0  , 100,   0),    # Tree cover
    1 : (255, 187,  34),    # Shrubland
    2 : (255, 255,  76),    # Grassland
    3 : (240, 150, 255),    # Cropland
    4 : (250,   0,   0),    # Built-up
    5 : (180, 180, 180),    # Bare / sparse vegetation
    6 : (240, 240, 240),    # Snow and Ice
    7 : (0  , 100, 200),    # Permanent water bodies
    8 : (0  , 150, 160),    # Herbaceous wetland
    9 : (0  , 207, 117),    # Mangroves
    10: (250, 230, 160),    # Moss and lichen
}

def preprocess_image_for_rgb(tensor, rgb_indices=[2, 1, 0], quantile_range=(0.02, 0.98)):
    """
    Convert multi-spectral tensor to RGB for visualization.
    Based on the logic from plot_inference_2.ipynb
    """
    # Convert to tensor and move to CPU
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    tensor = tensor.detach().cpu()
    
    # Handle batch dimension
    if tensor.ndim == 4:
        tensor = tensor[0]  # Take first sample
    
    # Ensure channel-first format [C,H,W]
    if tensor.ndim == 3 and tensor.shape[2] < tensor.shape[1]:
        tensor = tensor.permute(2, 0, 1)
    
    # Check if we have enough channels for RGB
    if tensor.shape[0] < max(rgb_indices) + 1:
        print(f"Warning: Only {tensor.shape[0]} channels available, creating grayscale")
        # Convert to grayscale
        if tensor.shape[0] >= 1:
            gray = tensor[0].float()
            rgb = torch.stack([gray, gray, gray], dim=0)
        else:
            return None
    else:
        # Extract RGB channels
        rgb = tensor[rgb_indices].float()
    
    # Convert to [H,W,C] format
    rgb = rgb.permute(1, 2, 0)
    
    # Normalize using quantiles
    if rgb.numel() > 0:
        lo = torch.quantile(rgb, quantile_range[0])
        hi = torch.quantile(rgb, quantile_range[1])
        
        if hi > lo:
            rgb = torch.clamp((rgb - lo) / (hi - lo), 0, 1)
        else:
            rgb = torch.zeros_like(rgb)
    
    return rgb.numpy()

def aggregate_labels_for_task(label, task_name):
    """
    Apply task-specific label aggregation.
    For clouds task: aggregate labels 0,1 -> 0 (no clouds), and 2,3,4 -> 1 (clouds)
    """
    if task_name == "clouds":
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label)
        label = label.cpu()
        
        # Create aggregated label
        aggregated_label = torch.zeros_like(label)
        
        # No clouds: labels 0,1 -> 0
        aggregated_label[(label == 0) | (label == 1)] = 0
        
        # Clouds: labels 2,3,4 -> 1
        aggregated_label[(label == 2) | (label == 3) | (label == 4)] = 1
        
        return aggregated_label
    
    return label

def process_label_to_onehot(label, num_classes, task_name=None, classes=None):
    """
    Convert label to one-hot encoding.
    If classes is None, return the original label without preprocessing.
    """
    # Handle None classes case - return original label without preprocessing
    if classes is None:
        if isinstance(label, torch.Tensor):
            return label.cpu().numpy()
        return label
    
    # Apply task-specific label aggregation
    if task_name:
        label = aggregate_labels_for_task(label, task_name)
    
    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label)
    label = label.cpu()
    
    if label.ndim == 2:  # Segmentation map [H,W]
        # Convert to one-hot [C,H,W]
        one_hot = torch.zeros(num_classes, label.shape[0], label.shape[1])
        one_hot.scatter_(0, label.unsqueeze(0).long(), 1)
        return one_hot.numpy()
    elif label.ndim == 3 and label.shape[0] == num_classes:
        # Already one-hot encoded
        return label.numpy()
    elif label.ndim == 1 or (label.ndim == 0):  # Classification
        # Convert to one-hot vector
        one_hot = torch.zeros(num_classes)
        if label.ndim == 0:
            one_hot[label.long()] = 1
        else:
            one_hot = label.float()
        return one_hot.numpy()
    
    return label.numpy()

def convert_onehot_to_class_indices(onehot_label):
    """
    Convert one-hot encoded label to class indices.
    """
    if onehot_label.ndim == 1:
        # Classification: [C] -> scalar
        return np.argmax(onehot_label)
    elif onehot_label.ndim == 2:
        # Already class indices [H, W]
        return onehot_label.astype(np.uint8)
    elif onehot_label.ndim == 3:
        # Segmentation one-hot: [C, H, W] -> [H, W]
        return np.argmax(onehot_label, axis=0).astype(np.uint8)
    else:
        raise ValueError(f"Unexpected label dimensions: {onehot_label.shape}")

def create_colored_label_image(onehot_label, color_map, is_regression=False):
    """
    Create a colored image from one-hot encoded label using the color map.
    For regression tasks, visualize floating-point values as intensity.
    """
    # Handle regression case (floating-point values)
    if is_regression:
        if onehot_label.ndim == 3 and onehot_label.shape[0] == 1:
            # Single channel floating-point data [1, H, W]
            data = onehot_label[0]
        elif onehot_label.ndim == 2:
            # Single channel floating-point data [H, W]
            data = onehot_label
        else:
            print(f"Warning: Unexpected regression data shape: {onehot_label.shape}")
            data = onehot_label
        
        # Normalize to 0-255 range for visualization
        if data.max() > data.min():
            normalized = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(data, dtype=np.uint8)
        
        # Create colored image using a colormap (e.g., viridis-like: blue to yellow)
        h, w = normalized.shape
        colored_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Apply colormap: low values = blue, high values = yellow/red
        # Blue component: high for low values
        colored_img[:, :, 0] = 255 - normalized  # Blue channel (BGR format)
        # Green component: medium for middle values
        colored_img[:, :, 1] = normalized  # Green channel
        # Red component: high for high values
        colored_img[:, :, 2] = normalized  # Red channel
        
        return colored_img
    
    # Handle classification/segmentation case (discrete classes)
    class_indices = convert_onehot_to_class_indices(onehot_label)
    
    if np.isscalar(class_indices):
        # Classification: create a small colored square
        color = color_map.get(int(class_indices), (128, 128, 128))  # Default gray
        colored_img = np.full((100, 100, 3), color, dtype=np.uint8)
        return colored_img
    else:
        # Segmentation: create colored image
        h, w = class_indices.shape
        colored_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Map each class to its color (BGR format)
        for class_idx, bgr_color in color_map.items():
            mask = (class_indices == class_idx)
            colored_img[mask] = bgr_color
            
        return colored_img

def create_legend_patches(task_info, color_map):
    """
    Create legend patches for matplotlib plots.
    """
    labels = task_info.get('labels', [])
    patches_list = []
    labels_list = []
    
    for i, label_name in enumerate(labels):
        # Get color (BGR -> RGB for matplotlib)
        bgr_color = color_map.get(i, (128, 128, 128))  # Default gray
        rgb_color = (bgr_color[2]/255, bgr_color[1]/255, bgr_color[0]/255)  # Normalize to [0,1]
        
        # Create patch
        patch = patches.Rectangle((0, 0), 1, 1, facecolor=rgb_color, edgecolor='black', linewidth=0.5)
        patches_list.append(patch)
        labels_list.append(f"Class {i}: {label_name}")
    
    return patches_list, labels_list

def create_comparison_plots(task_name, config, samples_per_plot=50):
    """
    Create matplotlib plots comparing original images with colored labels.
    """
    print(f"\n=== Creating comparison plots for {task_name} ===")
    
    task_output_dir = Path(OUTPUT_BASE_DIR) / task_name
    img_dir = task_output_dir / "img"
    label_dir = task_output_dir / "label"
    plots_dir = task_output_dir / "comparison_plots"
    
    # Create plots directory
    plots_dir.mkdir(exist_ok=True)
    
    # Get all image and label files
    img_files = sorted(list(img_dir.glob("*.png")))
    label_files = sorted(list(label_dir.glob("*.npy")))
    
    if not img_files or not label_files:
        print(f"No image or label files found for {task_name}")
        return
    
    # Match image and label files
    matched_pairs = []
    for img_file in img_files:
        label_file = label_dir / (img_file.stem + ".npy")
        if label_file.exists():
            matched_pairs.append((img_file, label_file))
    
    if not matched_pairs:
        print(f"No matching image-label pairs found for {task_name}")
        return
    
    print(f"Found {len(matched_pairs)} matching image-label pairs")
    
    # Create legend patches (only for non-regression tasks)
    is_regression = config.get("task_type") == "regression"
    if is_regression:
        patches_list, labels_list = [], []
    else:
        patches_list, labels_list = create_legend_patches(config, COLOR_MAPS)
    
    # Create plots with samples_per_plot pairs each
    num_plots = (len(matched_pairs) + samples_per_plot - 1) // samples_per_plot
    
    for plot_idx in range(num_plots):
        start_idx = plot_idx * samples_per_plot
        end_idx = min(start_idx + samples_per_plot, len(matched_pairs))
        current_pairs = matched_pairs[start_idx:end_idx]
        
        print(f"Creating comparison plot {plot_idx + 1}/{num_plots} with {len(current_pairs)} pairs...")
        
        # Calculate grid dimensions
        n_pairs = len(current_pairs)
        cols = 10  # 5 pairs per row (image + label = 2 columns per pair)
        rows = ((n_pairs * 2) + cols - 1) // cols  # Each pair needs 2 columns
        
        # Create figure with proper size
        fig = plt.figure(figsize=(20, max(12, rows * 2)))
        
        # Create main grid for the samples
        main_gs = GridSpec(rows + 1, cols, figure=fig, height_ratios=[0.1] + [1] * rows)
        
        # Add legend at the top
        legend_ax = fig.add_subplot(main_gs[0, :])
        if patches_list:
            legend_ax.legend(patches_list, labels_list, loc='center', 
                           ncol=min(6, len(labels_list)), frameon=True,
                           fontsize=8, title=f"{task_name.replace('_', ' ').title()} - Classes")
        else:
            # For regression tasks, show colorbar legend
            legend_ax.text(0.5, 0.5, f"{task_name.replace('_', ' ').title()} - Regression Values\n(Blue=Low, Red=High)", 
                          ha='center', va='center', fontsize=10, fontweight='bold')
        legend_ax.axis('off')
        
        # Add image-label pairs
        for pair_idx, (img_file, label_file) in enumerate(current_pairs):
            try:
                # Load image
                img = cv2.imread(str(img_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Load and process label
                onehot_label = np.load(label_file)
                is_regression = config.get("task_type") == "regression"
                colored_label = create_colored_label_image(onehot_label, COLOR_MAPS, is_regression=is_regression)
                colored_label = cv2.cvtColor(colored_label, cv2.COLOR_BGR2RGB)
                
                # Calculate position in grid
                grid_pos = pair_idx * 2  # Each pair takes 2 columns
                row = (grid_pos // cols) + 1  # +1 to account for legend row
                col = grid_pos % cols
                
                # Add original image
                if col < cols:
                    ax_img = fig.add_subplot(main_gs[row, col])
                    ax_img.imshow(img)
                    ax_img.set_title(f"Image {img_file.stem.split('_')[-1]}", fontsize=8)
                    ax_img.axis('off')
                
                # Add colored label
                if col + 1 < cols:
                    ax_label = fig.add_subplot(main_gs[row, col + 1])
                    ax_label.imshow(colored_label)
                    ax_label.set_title(f"Label {img_file.stem.split('_')[-1]}", fontsize=8)
                    ax_label.axis('off')
                
            except Exception as e:
                print(f"Error processing pair {pair_idx}: {e}")
                continue
        
        # Save plot
        plot_filename = f"{task_name}_comparison_plot_{plot_idx + 1:02d}.png"
        plot_path = plots_dir / plot_filename
        
        plt.tight_layout()
        plt.savefig(str(plot_path), dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved comparison plot: {plot_path}")
    
    print(f"Created {num_plots} comparison plots for {task_name}")

def create_binary_masks_from_onehot(onehot_label, max_classes=5):
    """
    Convert one-hot encoded label to individual binary masks.
    Returns list of binary masks (white=present, black=absent).
    """
    binary_masks = []
    num_classes = min(onehot_label.shape[0], max_classes)
    
    for class_idx in range(num_classes):
        # Create binary mask: white (255) where class is present, black (0) where absent
        mask = (onehot_label[class_idx] * 255).astype(np.uint8)
        # Convert to 3-channel for display
        mask_rgb = np.stack([mask, mask, mask], axis=-1)
        binary_masks.append(mask_rgb)
    
    return binary_masks

def create_binary_comparison_plots(task_name, config, samples_per_plot=10):
    """
    Create comparison plots with binary masks instead of colored labels.
    Each plot contains 10 products, each product has 1 original image + 5 binary masks.
    Skips regression tasks as they don't have discrete classes.
    """
    print(f"Creating binary mask comparison plots for {task_name}...")
    
    # Skip regression tasks
    if config.get("task_type") == "regression":
        print(f"Skipping binary mask plots for regression task {task_name}")
        return
    
    # Setup directories
    base_dir = Path(f"extracted_samples/{task_name}")
    img_dir = base_dir / "img"
    label_dir = base_dir / "label"
    plots_dir = base_dir / "binary_comparison_plots"
    plots_dir.mkdir(exist_ok=True)
    
    if not img_dir.exists() or not label_dir.exists():
        print(f"Missing directories for {task_name}")
        return
    
    # Get matching files
    img_files = sorted(list(img_dir.glob("*.png")))
    
    # Match image and label files
    matched_pairs = []
    for img_file in img_files:
        label_file = label_dir / (img_file.stem + ".npy")
        if label_file.exists():
            matched_pairs.append((img_file, label_file))
    
    if not matched_pairs:
        print(f"No matching image-label pairs found for {task_name}")
        return
    
    print(f"Found {len(matched_pairs)} matching image-label pairs")
    
    # Get class names for legend
    class_names = config.get("class_names", [f"Class {i}" for i in range(5)])
    
    # Create plots with samples_per_plot products each
    num_plots = (len(matched_pairs) + samples_per_plot - 1) // samples_per_plot
    
    for plot_idx in range(num_plots):
        start_idx = plot_idx * samples_per_plot
        end_idx = min(start_idx + samples_per_plot, len(matched_pairs))
        current_pairs = matched_pairs[start_idx:end_idx]
        
        print(f"Creating binary comparison plot {plot_idx + 1}/{num_plots} with {len(current_pairs)} products...")
        
        # Calculate grid dimensions: 6 columns per product (1 image + 5 masks)
        n_products = len(current_pairs)
        cols_per_product = 6  # 1 image + 5 binary masks
        total_cols = cols_per_product * min(2, n_products)  # Max 2 products per row
        products_per_row = 2 if n_products > 1 else 1
        rows = ((n_products + products_per_row - 1) // products_per_row) + 1  # +1 for legend
        
        # Create figure with proper size
        fig = plt.figure(figsize=(24, max(8, rows * 4)))
        
        # Create main grid
        main_gs = GridSpec(rows, total_cols, figure=fig, 
                          height_ratios=[0.1] + [1] * (rows-1))
        
        # Add legend at the top
        legend_ax = fig.add_subplot(main_gs[0, :])
        legend_patches = [plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', label='Class Present'),
                         plt.Rectangle((0, 0), 1, 1, facecolor='black', edgecolor='black', label='Class Absent')]
        legend_ax.legend(legend_patches, ['Class Present (White)', 'Class Absent (Black)'], 
                        loc='center', ncol=2, frameon=True, fontsize=10,
                        title=f"{task_name.replace('_', ' ').title()} - Binary Masks")
        legend_ax.axis('off')
        
        # Add products
        for product_idx, (img_file, label_file) in enumerate(current_pairs):
            try:
                # Load image
                img = cv2.imread(str(img_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Load and process label
                onehot_label = np.load(label_file)
                binary_masks = create_binary_masks_from_onehot(onehot_label, max_classes=5)
                
                # Calculate position in grid
                row_idx = (product_idx // products_per_row) + 1  # +1 for legend
                col_start = (product_idx % products_per_row) * cols_per_product
                
                # Add original image
                ax_img = fig.add_subplot(main_gs[row_idx, col_start])
                ax_img.imshow(img)
                ax_img.set_title(f"Image {img_file.stem.split('_')[-1]}", fontsize=8)
                ax_img.axis('off')
                
                # Add binary masks
                for mask_idx, binary_mask in enumerate(binary_masks):
                    if col_start + mask_idx + 1 < total_cols:
                        ax_mask = fig.add_subplot(main_gs[row_idx, col_start + mask_idx + 1])
                        ax_mask.imshow(binary_mask)
                        class_name = class_names[mask_idx] if mask_idx < len(class_names) else f"Class {mask_idx}"
                        ax_mask.set_title(f"{class_name}", fontsize=8)
                        ax_mask.axis('off')
                
            except Exception as e:
                print(f"Error processing product {product_idx}: {e}")
                continue
        
        # Save plot
        plot_filename = f"{task_name}_binary_comparison_plot_{plot_idx + 1:02d}.png"
        plot_path = plots_dir / plot_filename
        
        plt.tight_layout()
        plt.savefig(str(plot_path), dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Saved binary comparison plot: {plot_path}")
    
    print(f"Created {num_plots} binary comparison plots for {task_name}")

def debug_dataset_structure(dl_test, task_name):
    """
    Debug function to understand the dataset structure and sample ID mapping.
    """
    print(f"\n=== Debugging dataset structure for {task_name} ===")
    
    dataset = dl_test.dataset
    print(f"Dataset type: {type(dataset)}")
    print(f"Dataset size: {len(dataset)}")
    
    # Check for sample ID attributes
    sample_id_attrs = ['sample_ids', 'ids', 'keys', 'samples', 'file_list']
    for attr in sample_id_attrs:
        if hasattr(dataset, attr):
            attr_value = getattr(dataset, attr)
            print(f"Found attribute '{attr}': {type(attr_value)}, length: {len(attr_value) if hasattr(attr_value, '__len__') else 'N/A'}")
            if hasattr(attr_value, '__len__') and len(attr_value) > 0:
                print(f"  First few values: {attr_value[:5] if len(attr_value) >= 5 else attr_value}")
    
    # Check first sample structure
    if len(dataset) > 0:
        first_sample = dataset[0]
        print(f"First sample keys: {list(first_sample.keys()) if isinstance(first_sample, dict) else 'Not a dict'}")
        
        # Look for sample ID in the sample itself
        if isinstance(first_sample, dict):
            id_keys = ['sample_id', 'id', 'idx', 'key', 'name', 'filename']
            for key in id_keys:
                if key in first_sample:
                    print(f"Found sample ID key '{key}': {first_sample[key]}")
    
    print("=" * 50)

def load_dataset(task_name, config, num_samples=5000):
    """
    Load dataset for a specific task using configuration file.
    """
    print(f"Loading dataset for {task_name}...")
    
    # Read configuration
    args = read_yaml(config["config_file"])
    
    # Override specific parameters
    # For tasks with classes=None, use location-based clustering instead of class-based n_shot
    if config.get("classes") is None:
        args.n_shot = None  # Use location-based clustering
        print(f"Task {task_name} has classes=None, using location-based clustering")
    else:
        args.n_shot = 5000  # Use 5000 n-shot configuration
        
    batch_size = 1      # Process one sample at a time
    
    # Determine dataset path based on model
    model_name = args.model_name
    if hasattr(args, 'data_path_224_30m') and model_name in getattr(args, 'models_224_r30', []):
        dataset_folder = args.data_path_224_30m
    elif hasattr(args, 'data_path_224_10m'):
        dataset_folder = args.data_path_224_10m
    else:
        dataset_folder = args.data_path_128_10m
    
    # Set other parameters
    crop_images = model_name in ['phileo_precursor', 'phileo_precursor_classifier']
    patch_size = (256, 256) if task_name != "fire" else None
    
    try:
        # Load data
        weights, pos_weight, _, dl_test, _, _ = load_data(
            dataset_folder,
            with_augmentations=False,
            num_workers=4,
            batch_size=batch_size,
            downstream_task=task_name,
            model_name=model_name.split('_')[0],
            device='cpu',
            pad_bands=getattr(args, 'pad_bands', 10),
            crop_images=crop_images,
            num_classes=config["output_channels"],
            n=num_samples,
            weights_dir=task_name,
            patch_size=patch_size
        )
        
        return dl_test
        
    except Exception as e:
        import traceback
        print(f"Error loading dataset for {task_name}: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return None

def extract_samples(task_name, config):
    """
    Extract samples for a specific task.
    """
    num_samples = config.get("num_samples", NUM_SAMPLES_PER_TASK)
    print(f"\n=== Extracting {num_samples} samples for {task_name} ===")
    
    # Create output directories
    task_output_dir = Path(OUTPUT_BASE_DIR) / task_name
    img_dir = task_output_dir / "img"
    label_dir = task_output_dir / "label"
    
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dl_test = load_dataset(task_name, config)
    if dl_test is None:
        print(f"Failed to load dataset for {task_name}")
        return
    
    # Debug dataset structure to understand sample ID mapping
    debug_dataset_structure(dl_test, task_name)
    
    # Determine sample indices
    dataset_size = len(dl_test.dataset)
    print(f"Dataset size: {dataset_size}")
    
    if num_samples >= dataset_size:
        sample_indices = list(range(dataset_size))
    else:
        # Use evenly spaced samples
        step = max(1, dataset_size // num_samples)
        sample_indices = list(range(0, dataset_size, step))[:num_samples]
    
    print(f"Extracting {len(sample_indices)} samples...")
    
    # Extract samples
    successful_extractions = 0
    for i, sample_idx in enumerate(tqdm(sample_indices, desc=f"Processing {task_name}")):
        try:
            # Get sample from dataset
            sample = dl_test.dataset[sample_idx]
            img = sample['img']
            label = sample['label']
            
            # Try to get the actual sample ID from the dataset
            # This depends on how the dataset is structured
            actual_sample_id = None
            if hasattr(dl_test.dataset, 'sample_ids') and sample_idx < len(dl_test.dataset.sample_ids):
                actual_sample_id = dl_test.dataset.sample_ids[sample_idx]
            elif 'sample_id' in sample:
                actual_sample_id = sample['sample_id']
            elif hasattr(dl_test.dataset, 'ids') and sample_idx < len(dl_test.dataset.ids):
                actual_sample_id = dl_test.dataset.ids[sample_idx]
            else:
                # Fallback to dataset index if no actual ID is available
                actual_sample_id = sample_idx
                print(f"Warning: Could not find actual sample ID for index {sample_idx}, using index as ID")
            
            # Process image for RGB visualization
            rgb_img = preprocess_image_for_rgb(img)
            
            if rgb_img is not None:
                # Convert to 8-bit and save as PNG
                img_8bit = (rgb_img * 255).astype(np.uint8)
                img_filename = f"{task_name}_sample_{actual_sample_id:05d}.png"
                img_path = img_dir / img_filename
                cv2.imwrite(str(img_path), cv2.cvtColor(img_8bit, cv2.COLOR_RGB2BGR))
            
            # Process label to one-hot encoding
            onehot_label = process_label_to_onehot(
                label, 
                config["output_channels"], 
                task_name=task_name, 
                classes=config.get("classes")
            )
            
            # Save label as numpy array
            label_filename = f"{task_name}_sample_{actual_sample_id:05d}.npy"
            label_path = label_dir / label_filename
            np.save(str(label_path), onehot_label)
            
            # Debug: Print first few actual vs index IDs
            if i < 5:
                print(f"Sample index {sample_idx} -> Actual ID {actual_sample_id}")
            
            successful_extractions += 1
            
        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")
            continue
    
    print(f"Successfully extracted {successful_extractions} samples for {task_name}")
    print(f"Images saved to: {img_dir}")
    print(f"Labels saved to: {label_dir}")
    
    # Create comparison plots
    create_comparison_plots(task_name, config)

def main():
    """
    Main function to extract samples from all configured tasks.
    """
    print("Dataset Sample Extraction Script")
    print("=" * 50)
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print(f"Samples per task: {NUM_SAMPLES_PER_TASK}")
    print(f"Tasks: {list(TASKS.keys())}")
    
    # Create base output directory
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # Process each task
    for task_name, config in TASKS.items():
        # Check if config file exists
        if not os.path.exists(config["config_file"]):
            print(f"Warning: Config file not found for {task_name}: {config['config_file']}")
            continue
            
        try:
            extract_samples(task_name, config)
        except Exception as e:
            print(f"Failed to process {task_name}: {e}")
            continue
    
    # Print final summary
    print("\n" + "=" * 50)
    print("EXTRACTION COMPLETE")
    print("=" * 50)
    
    for task_name in TASKS.keys():
        task_dir = Path(OUTPUT_BASE_DIR) / task_name
        if task_dir.exists():
            img_count = len(list((task_dir / "img").glob("*.png")))
            label_count = len(list((task_dir / "label").glob("*.npy")))
            plot_count = len(list((task_dir / "comparison_plots").glob("*.png"))) if (task_dir / "comparison_plots").exists() else 0
            print(f"{task_name:20}: {img_count:3d} images, {label_count:3d} labels, {plot_count:2d} comparison plots")
    
    print(f"\nAll files saved to: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()