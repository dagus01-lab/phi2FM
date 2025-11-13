#!/usr/bin/env python3
"""
Simple script to display the generated legend plots.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

# Configuration
BASE_DIR = "/home/gdaga/phi2FM/downstream/extracted_samples"

def show_legends():
    """
    Display all generated legend plots.
    """
    base_path = Path(BASE_DIR)
    legend_files = list(base_path.glob("*/*_legend.png"))
    
    if not legend_files:
        print("No legend files found")
        return
    
    print(f"Found {len(legend_files)} legend files")
    
    # Create subplots
    n_legends = len(legend_files)
    cols = 2
    rows = (n_legends + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, legend_file in enumerate(legend_files):
        # Load and display legend
        img = mpimg.imread(str(legend_file))
        
        if i < len(axes):
            axes[i].imshow(img)
            axes[i].set_title(legend_file.stem.replace('_legend', '').replace('_', ' ').title())
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(legend_files), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def show_single_legend(task_name):
    """
    Display a single legend plot.
    """
    legend_path = Path(BASE_DIR) / task_name / f"{task_name}_legend.png"
    
    if not legend_path.exists():
        print(f"Legend not found: {legend_path}")
        return
    
    img = mpimg.imread(str(legend_path))
    
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title(f"{task_name.replace('_', ' ').title()} Legend")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Show all legends
    show_legends()
    
    # Uncomment to show individual legends
    # show_single_legend("clouds")
    # show_single_legend("fire")