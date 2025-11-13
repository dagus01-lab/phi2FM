#!/usr/bin/env python3
"""
Test script to create binary mask comparison plots for a single task.
"""

import sys
from pathlib import Path

# Add the current directory to the Python path
sys.path.append(str(Path(__file__).parent))

from extract_samples_simple import create_binary_comparison_plots

def main():
    """Test binary mask comparison plots on clouds task."""
    
    task_name = "clouds"
    config = {
        "config_file": "args/finetuning_clouds.yaml", 
        "class_names": ["Clear", "Cloud shadow", "Semi-transparent", "Cloud", "Missing"]
    }
    
    print(f"Testing binary mask comparison plots for {task_name}...")
    
    try:
        create_binary_comparison_plots(task_name, config, samples_per_plot=10)
        
        # Check results
        plots_dir = Path(f"extracted_samples/{task_name}/binary_comparison_plots")
        if plots_dir.exists():
            plot_files = list(plots_dir.glob("*.png"))
            print(f"Successfully created {len(plot_files)} binary comparison plots")
            for plot_file in plot_files:
                print(f"  - {plot_file.name}")
        else:
            print("No plots directory found")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()