#!/usr/bin/env python3
"""
Debug script to investigate regression dataset labels.
"""

import numpy as np
import zarr
import sys
from pathlib import Path

def investigate_dataset(dataset_path, task_name):
    """Investigate a zarr dataset to understand label format and values."""
    print(f"\n=== Investigating {task_name} ===")
    print(f"Dataset path: {dataset_path}")
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return
    
    try:
        # Open zarr dataset
        store = zarr.open(dataset_path, mode='r')
        print(f"✓ Successfully opened zarr store")
        
        # Print dataset structure
        print(f"Dataset keys: {list(store.keys())}")
        
        # Check if we have typical zarr structure
        if 'labels' in store:
            labels = store['labels']
            print(f"Labels shape: {labels.shape}")
            print(f"Labels dtype: {labels.dtype}")
            
            # Sample some labels to understand the values
            sample_size = min(10, labels.shape[0])
            print(f"Sampling first {sample_size} labels...")
            
            for i in range(sample_size):
                label = labels[i]
                print(f"  Label {i}: shape={label.shape}, dtype={label.dtype}")
                print(f"    min={np.min(label):.4f}, max={np.max(label):.4f}")
                print(f"    mean={np.mean(label):.4f}, std={np.std(label):.4f}")
                print(f"    unique_values={len(np.unique(label))}")
                if len(np.unique(label)) < 20:  # Only show if not too many unique values
                    print(f"    unique_values_sample={np.unique(label)[:10]}")
                print()
                
                if i >= 3:  # Limit output
                    break
                    
        elif 'label' in store:
            label = store['label']
            print(f"Label shape: {label.shape}")
            print(f"Label dtype: {label.dtype}")
            
            # Sample some labels
            sample_size = min(10, label.shape[0])
            print(f"Sampling first {sample_size} labels...")
            
            for i in range(sample_size):
                lbl = label[i]
                print(f"  Label {i}: shape={lbl.shape}, dtype={lbl.dtype}")
                print(f"    min={np.min(lbl):.4f}, max={np.max(lbl):.4f}")
                print(f"    mean={np.mean(lbl):.4f}, std={np.std(lbl):.4f}")
                print(f"    unique_values={len(np.unique(lbl))}")
                if len(np.unique(lbl)) < 20:
                    print(f"    unique_values_sample={np.unique(lbl)[:10]}")
                print()
                
                if i >= 3:
                    break
        else:
            print(f"❓ No 'labels' or 'label' key found. Available keys: {list(store.keys())}")
            
            # Try to find any key that might contain labels
            for key in store.keys():
                if 'label' in key.lower() or 'target' in key.lower() or 'mask' in key.lower():
                    print(f"Found potential label key: {key}")
                    data = store[key]
                    print(f"  Shape: {data.shape}, dtype: {data.dtype}")
                    if data.size > 0:
                        sample = data[0] if len(data.shape) > 0 else data
                        print(f"  Sample min/max: {np.min(sample):.4f}/{np.max(sample):.4f}")
                        
    except Exception as e:
        print(f"❌ Error investigating dataset: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to investigate regression datasets."""
    
    datasets_to_check = [
        ("/Data/phileo-bench_building.zarr", "building_regression"),
        ("/Data/phileo-bench_roads.zarr", "roads_regression"),
        ("/Data/fire_dataset/fire_dataset.zarr", "fire_dataset")  # For comparison
    ]
    
    for dataset_path, task_name in datasets_to_check:
        investigate_dataset(dataset_path, task_name)

if __name__ == "__main__":
    main()