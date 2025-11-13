#!/usr/bin/env python3
"""
Test script to create comparison plots for a single task.
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the functions from the main script
from extract_samples_simple import create_comparison_plots, TASKS

def test_comparison_plots():
    """Test comparison plot creation for clouds task."""
    task_name = "clouds"
    config = TASKS[task_name]
    
    print(f"Testing comparison plots for {task_name}")
    create_comparison_plots(task_name, config, samples_per_plot=50)

if __name__ == "__main__":
    test_comparison_plots()