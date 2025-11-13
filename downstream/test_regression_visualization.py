#!/usr/bin/env python3
"""
Test script to verify that regression task visualization works correctly.
"""

import numpy as np
import sys
import os
sys.path.append('/home/gdaga/phi2FM/downstream')

from extract_samples_simple import create_colored_label_image, TASKS
import matplotlib.pyplot as plt

def test_regression_visualization():
    """Test regression task visualization with synthetic floating-point data."""
    print("Testing regression visualization...")
    
    # Create synthetic floating-point data (similar to what roads/buildings would have)
    h, w = 64, 64
    
    # Test case 1: Single channel [H, W] format
    test_data_hw = np.random.rand(h, w) * 10.0  # Values between 0 and 10
    
    # Test case 2: Single channel [1, H, W] format  
    test_data_1hw = np.random.rand(1, h, w) * 5.0  # Values between 0 and 5
    
    # Test case 3: Gradient pattern
    x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    gradient_data = x + y  # Values between 0 and 2
    
    test_cases = [
        ("Random [H,W]", test_data_hw),
        ("Random [1,H,W]", test_data_1hw),
        ("Gradient [H,W]", gradient_data)
    ]
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Regression Task Visualization Test", fontsize=16)
    
    for i, (name, data) in enumerate(test_cases):
        # Original data
        axes[0, i].imshow(data[0] if data.ndim == 3 else data, cmap='viridis')
        axes[0, i].set_title(f"{name} - Original")
        axes[0, i].axis('off')
        
        # Processed visualization
        colored_image = create_colored_label_image(data, {}, is_regression=True)
        # Convert BGR to RGB for matplotlib
        colored_image_rgb = colored_image[:, :, [2, 1, 0]]
        axes[1, i].imshow(colored_image_rgb)
        axes[1, i].set_title(f"{name} - Processed")
        axes[1, i].axis('off')
        
        print(f"✓ {name}: Shape {data.shape}, Range [{np.min(data):.2f}, {np.max(data):.2f}]")
    
    # Save test plot
    plt.tight_layout()
    plt.savefig('/home/gdaga/phi2FM/downstream/test_regression_visualization.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Test completed successfully!")
    print("✓ Test plot saved to: /home/gdaga/phi2FM/downstream/test_regression_visualization.png")

def check_task_configurations():
    """Check that task configurations are correct."""
    print("\nChecking task configurations...")
    
    regression_tasks = []
    classification_tasks = []
    
    for task_name, config in TASKS.items():
        task_type = config.get("task_type", "classification")
        if task_type == "regression":
            regression_tasks.append(task_name)
        else:
            classification_tasks.append(task_name)
    
    print(f"✓ Regression tasks: {regression_tasks}")
    print(f"✓ Classification tasks: {classification_tasks}")
    
    # Verify regression task configs
    for task_name in regression_tasks:
        config = TASKS[task_name]
        classes = config.get("classes")
        labels = config.get("labels")
        print(f"  - {task_name}: classes={classes}, labels={labels}")

if __name__ == "__main__":
    test_regression_visualization()
    check_task_configurations()