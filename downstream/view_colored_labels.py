#!/usr/bin/env python3
"""
Simple script to view the generated colored label images alongside the original RGB images.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
BASE_DIR = "/home/gdaga/phi2FM/downstream/extracted_samples"
SAMPLE_ID = "00000"  # Change this to view different samples

def show_sample_comparison(task_name, sample_id):
    """
    Show side-by-side comparison of RGB image, original label, and colored label.
    """
    task_dir = Path(BASE_DIR) / task_name
    
    # File paths
    rgb_img_path = task_dir / "img" / f"{task_name}_sample_{sample_id}.png"
    colored_label_path = task_dir / "label_images" / f"{task_name}_sample_{sample_id}.png"
    original_label_path = task_dir / "label" / f"{task_name}_sample_{sample_id}.npy"
    
    # Check if files exist
    if not all([rgb_img_path.exists(), colored_label_path.exists(), original_label_path.exists()]):
        print(f"Missing files for {task_name} sample {sample_id}")
        return
    
    # Load images
    rgb_img = cv2.imread(str(rgb_img_path))
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    
    colored_label = cv2.imread(str(colored_label_path))
    colored_label = cv2.cvtColor(colored_label, cv2.COLOR_BGR2RGB)
    
    original_label = np.load(original_label_path)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RGB Image
    axes[0].imshow(rgb_img)
    axes[0].set_title(f'RGB Image\n{task_name} sample {sample_id}')
    axes[0].axis('off')
    
    # Colored Label
    axes[1].imshow(colored_label)
    axes[1].set_title(f'Colored Label\nShape: {colored_label.shape}')
    axes[1].axis('off')
    
    # Original Label Info
    if original_label.ndim == 1:
        # Classification - show as text
        class_idx = np.argmax(original_label)
        axes[2].text(0.5, 0.5, f'Classification\nClass: {class_idx}\nProbabilities:\n{original_label}', 
                    ha='center', va='center', fontsize=12, transform=axes[2].transAxes)
        axes[2].set_title(f'Original Label\nShape: {original_label.shape}')
    else:
        # Segmentation - show class map
        if original_label.ndim == 3:
            class_map = np.argmax(original_label, axis=0)
        else:
            class_map = original_label
        
        im = axes[2].imshow(class_map, cmap='tab10', vmin=0, vmax=4)
        axes[2].set_title(f'Class Indices\nShape: {original_label.shape}')
        plt.colorbar(im, ax=axes[2], shrink=0.6)
    
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Show samples for all available tasks.
    """
    print(f"Viewing sample {SAMPLE_ID} for all tasks")
    print("=" * 50)
    
    base_path = Path(BASE_DIR)
    tasks = [d.name for d in base_path.iterdir() if d.is_dir()]
    
    for task in tasks:
        try:
            show_sample_comparison(task, SAMPLE_ID)
        except Exception as e:
            print(f"Error showing {task}: {e}")

if __name__ == "__main__":
    main()