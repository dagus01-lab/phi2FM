#!/usr/bin/env python3
"""
Script to generate colored label images from numpy label files.
Maps class indices to different colors for visualization.
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==================== CONFIGURATION ====================

# Base directory containing the extracted samples
BASE_DIR = "/home/gdaga/phi2FM/downstream/extracted_samples"

# Color mapping for each task (BGR format for OpenCV)
TASKS = {
    "fire": {
        "dataset_dir": "/Data/fire_dataset/fire_dataset.zarr",
        "config_file": "args/finetune_FMs/fire/seco.yml",
        "output_channels": 4,
        "labels": ['safe', 'fire', 'burnt', 'water'], 
        "num_samples": 150
    },
    "burned_area": {
        "dataset_dir": "/Data/lpl_burned_area/burned.zarr",
        "config_file": "args/finetune_FMs/lpl_burned_area/seco.yml", 
        "output_channels": 4,
        "labels": ['Background', 'Burned Area', 'Clouds', 'Waterbodies'], 
        "num_samples": 150
    },
    "clouds": {
        "dataset_dir": "/Data/phisatnet_clouds/phisatnet_clouds.zarr",
        "config_file": "args/finetune_FMs/phisatnet_clouds/seco.yml",
        "output_channels": 5,
        "labels": ['No cloud', 'Cloud', 'value2', 'value3', 'value4'], 
        "num_samples": 100
    },
    "worldfloods": {
        "dataset_dir": "/Data/worldfloods/worldfloods.zarr",
        "config_file": "args/finetune_FMs/worldfloods/seco.yml",
        "output_channels": 3,
        "labels": ['Clouds', 'Land', 'Water'], 
        "num_samples": 150
    },
    "anomaly_detection": {
        "dataset_dir": "/Data/anomaly_detection/marine_area_dataset.zarr",
        "config_file": "args/finetune_FMs/anomaly_detection/seco.yml",
        "output_channels": 9,
        "labels": ['NO DATA', 'CLEAR WATER', 'TURBID WATER', 'LAND', 'PLASTIC', 'OIL', 'ALGAE', 'SEDIMENTS', 'CLOUD'],
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
        "num_samples": 100
    },
    "building_regression": {
        "dataset_dir": "/Data/anomaly_detection/marine_area_dataset.zarr",
        "config_file": "args/finetune_FMs/anomaly_detection/seco.yml",
        "output_channels": 1,
        "labels": None,
        "num_samples": 100
    },
    "roads_regression": {
        "dataset_dir": "/Data/anomaly_detection/marine_area_dataset.zarr",
        "config_file": "args/finetune_FMs/anomaly_detection/seco.yml",
        "output_channels": 1,
        "labels": None,
        "num_samples": 100
    }
}
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

def convert_onehot_to_class_indices(onehot_label):
    """
    Convert one-hot encoded label to class indices.
    
    Args:
        onehot_label: numpy array, can be:
            - [C, H, W] for segmentation (one-hot)
            - [C] for classification (one-hot)
            - [H, W] for segmentation (already class indices)
            - scalar for classification (already class index)
    
    Returns:
        numpy array of class indices
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

def create_colored_image(class_indices, color_map):
    """
    Create a colored image from class indices using the color map.
    
    Args:
        class_indices: numpy array of class indices
        color_map: dictionary mapping class index to BGR color tuple
    
    Returns:
        BGR image as numpy array
    """
    if np.isscalar(class_indices):
        # Classification: create a small colored square
        color = color_map.get(int(class_indices), (128, 128, 128))  # Default gray
        colored_img = np.full((100, 100, 3), color, dtype=np.uint8)
        return colored_img
    else:
        # Segmentation: create colored image
        h, w = class_indices.shape
        colored_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Map each class to its color
        for class_idx, color in color_map.items():
            mask = (class_indices == class_idx)
            colored_img[mask] = color
            
        return colored_img

def create_legend_plot(task_name, task_info, color_map, output_dir):
    """
    Create a legend plot showing class names and their corresponding colors.
    
    Args:
        task_name: Name of the task
        task_info: Dictionary containing task information including labels
        color_map: Dictionary mapping class index to BGR color tuple
        output_dir: Directory to save the legend plot
    """
    labels = task_info.get('labels')
    if not labels:
        print(f"No labels defined for task {task_name}, skipping legend creation")
        return
    
    num_classes = len(labels)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, max(6, num_classes * 0.5)))
    
    # Create color patches and labels
    patches_list = []
    labels_list = []
    
    for i, label_name in enumerate(labels):
        # Get color (BGR -> RGB for matplotlib)
        bgr_color = color_map.get(i, (128, 128, 128))  # Default gray
        rgb_color = (bgr_color[2]/255, bgr_color[1]/255, bgr_color[0]/255)  # Normalize to [0,1]
        
        # Create patch
        patch = patches.Rectangle((0, 0), 1, 1, facecolor=rgb_color, edgecolor='black', linewidth=1)
        patches_list.append(patch)
        labels_list.append(f"Class {i}: {label_name}")
    
    # Create legend
    legend = ax.legend(patches_list, labels_list, loc='center', 
                      bbox_to_anchor=(0.5, 0.5), frameon=True, 
                      fontsize=12, title=f"{task_name.replace('_', ' ').title()} - Class Legend",
                      title_fontsize=14)
    
    # Hide axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Save the plot
    legend_filename = f"{task_name}_legend.png"
    legend_path = output_dir / legend_filename
    
    plt.savefig(str(legend_path), dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Legend saved to: {legend_path}")

def process_task(task_name):
    """
    Process all label files for a specific task.
    """
    print(f"\n=== Processing {task_name} ===")
    
    task_dir = Path(BASE_DIR) / task_name
    label_dir = task_dir / "label"
    output_dir = task_dir / "label_images"
    
    if not label_dir.exists():
        print(f"Label directory not found: {label_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Get color map for this task
    color_map = COLOR_MAPS #.get(task_name, COLOR_MAPS["fire"])  # Default to fire colors
    
    # Create legend plot if task info is available
    task_info = TASKS.get(task_name, {})
    if task_info:
        print(f"Creating legend for {task_name}...")
        create_legend_plot(task_name, task_info, color_map, task_dir)  # Save in parent directory
    
    # Process all .npy files
    label_files = list(label_dir.glob("*.npy"))
    
    if not label_files:
        print(f"No .npy files found in {label_dir}")
        return
    
    print(f"Found {len(label_files)} label files")
    
    for label_file in tqdm(label_files, desc=f"Converting {task_name}"):
        try:
            # Load label
            onehot_label = np.load(label_file)
            
            # Convert to class indices
            class_indices = convert_onehot_to_class_indices(onehot_label)
            
            # Create colored image
            colored_img = create_colored_image(class_indices, color_map)
            
            # Save colored image
            output_filename = label_file.stem + ".png"  # Remove .npy, add .png
            output_path = output_dir / output_filename
            
            cv2.imwrite(str(output_path), colored_img)
            
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
            continue
    
    print(f"Colored label images saved to: {output_dir}")

def print_color_legend():
    """
    Print the color mapping legend for reference.
    """
    print("\n" + "="*60)
    print("COLOR MAPPING LEGEND")
    print("="*60)

    for class_idx, bgr_color in COLOR_MAPS.items():
        # Convert BGR to RGB for display
        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
        print(f"  Class {class_idx}: RGB{rgb_color} (BGR{bgr_color})")

def main():
    """
    Main function to process all tasks.
    """
    print("Label to Colored Image Converter")
    print("="*50)
    print(f"Base directory: {BASE_DIR}")
    
    # Print color legend
    print_color_legend()
    
    # Find all task directories
    base_path = Path(BASE_DIR)
    if not base_path.exists():
        print(f"Base directory not found: {BASE_DIR}")
        return
    
    task_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    if not task_dirs:
        print("No task directories found")
        return
    
    print(f"\nFound task directories: {[d.name for d in task_dirs]}")
    
    # Process each task
    for task_dir in task_dirs:
        task_name = task_dir.name
        process_task(task_name)
    
    # Final summary
    print("\n" + "="*50)
    print("CONVERSION COMPLETE")
    print("="*50)
    
    for task_dir in task_dirs:
        task_name = task_dir.name
        label_images_dir = task_dir / "label_images"
        if label_images_dir.exists():
            count = len(list(label_images_dir.glob("*.png")))
            print(f"{task_name:15}: {count:3d} colored label images")
    
    print(f"\nAll colored label images saved in respective 'label_images' folders")

if __name__ == "__main__":
    main()