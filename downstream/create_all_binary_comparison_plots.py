#!/usr/bin/env python3
"""
Script to create binary mask comparison plots for all downstream tasks.
Creates plots in separate directories to preserve original colored plots.
"""

import sys
from pathlib import Path

# Add the current directory to the Python path
sys.path.append(str(Path(__file__).parent))

from extract_samples_simple import create_binary_comparison_plots

def main():
    """Create binary mask comparison plots for all tasks."""
    
    # Task configurations with class names
    tasks = {
        "burned_area": {
            "config_file": "args/finetuning_burned_area.yaml",
            "class_names": ["No burn", "Low severity", "Moderate-low severity", "Moderate-high severity"]
        },
        "clouds": {
            "config_file": "args/finetuning_clouds.yaml", 
            "class_names": ["Clear", "Cloud shadow", "Semi-transparent", "Cloud", "Missing"]
        },
        "worldfloods": {
            "config_file": "args/finetuning_worldfloods.yaml",
            "class_names": ["Land", "Water", "Cloud"]
        },
        "fire": {
            "config_file": "args/finetuning_fire.yaml",
            "class_names": ["No fire", "Low confidence fire", "Nominal confidence fire", "High confidence fire"]
        },
        "anomaly_detection": {
            "config_file": "args/finetuning_anomaly_detection.yaml",
            "class_names": ["Background", "Oil spill", "Look-alike", "Ship", "Land", "Coastline", "Marine debris", "Seaweed", "Clouds"]
        }
    }
    
    total_plots = 0
    
    print("Creating binary mask comparison plots for all tasks...")
    print("=" * 60)
    
    for task_name, config in tasks.items():
        print(f"\nProcessing {task_name}...")
        try:
            # Check if extracted samples exist
            base_dir = Path(f"extracted_samples/{task_name}")
            if not base_dir.exists():
                print(f"  Skipping {task_name}: no extracted samples found")
                continue
                
            # Create binary comparison plots
            create_binary_comparison_plots(task_name, config, samples_per_plot=10)
            
            # Count created plots
            plots_dir = base_dir / "binary_comparison_plots"
            if plots_dir.exists():
                plot_files = list(plots_dir.glob("*.png"))
                total_plots += len(plot_files)
                print(f"  Created {len(plot_files)} binary comparison plots for {task_name}")
            
        except Exception as e:
            print(f"  Error processing {task_name}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print(f"Binary mask comparison plot generation complete!")
    print(f"Total binary plots created: {total_plots}")
    print("\nBinary plots are saved in:")
    print("  extracted_samples/{task_name}/binary_comparison_plots/")
    print("\nOriginal colored plots are preserved in:")
    print("  extracted_samples/{task_name}/comparison_plots/")

if __name__ == "__main__":
    main()