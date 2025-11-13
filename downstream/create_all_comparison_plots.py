#!/usr/bin/env python3
"""
Script to create comparison plots for all existing extracted tasks.
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from extract_samples_simple import create_comparison_plots, TASKS, OUTPUT_BASE_DIR

def create_all_comparison_plots():
    """Create comparison plots for all tasks that have extracted data."""
    base_path = Path(OUTPUT_BASE_DIR)
    
    if not base_path.exists():
        print(f"Base directory not found: {OUTPUT_BASE_DIR}")
        return
    
    # Find all task directories with extracted data
    task_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    for task_dir in task_dirs:
        task_name = task_dir.name
        
        # Check if task has images and labels
        img_dir = task_dir / "img"
        label_dir = task_dir / "label"
        
        if not (img_dir.exists() and label_dir.exists()):
            print(f"Skipping {task_name}: missing img or label directory")
            continue
        
        # Check if we have config for this task
        if task_name not in TASKS:
            print(f"Skipping {task_name}: no config found")
            continue
        
        config = TASKS[task_name]
        
        try:
            create_comparison_plots(task_name, config, samples_per_plot=50)
        except Exception as e:
            print(f"Error creating plots for {task_name}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON PLOTS SUMMARY")
    print("="*60)
    
    for task_dir in task_dirs:
        task_name = task_dir.name
        plots_dir = task_dir / "comparison_plots"
        if plots_dir.exists():
            plot_count = len(list(plots_dir.glob("*.png")))
            print(f"{task_name:20}: {plot_count:2d} comparison plots")

if __name__ == "__main__":
    create_all_comparison_plots()