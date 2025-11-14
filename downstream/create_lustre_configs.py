#!/usr/bin/env python3
"""
Script to create Lustre configuration files from finetune_FMs configs.
Updates paths to use /lustre/projects/1001/gdaga/home as base path.
"""

import os
import shutil
import yaml
from pathlib import Path

# Base paths
SOURCE_DIR = Path("args/finetune_FMs")
TARGET_DIR = Path("args/lustre")
LUSTRE_BASE = "/lustre/projects/1001/gdaga/home"

# Path mappings
PATH_MAPPINGS = {
    "/Data/worldfloods/worldfloods.zarr": f"{LUSTRE_BASE}/worldfloods.zarr",
    "/Data/phisatnet_dataset/phileo-bench_roads.zarr": f"{LUSTRE_BASE}/phileo-bench_roads.zarr",
    "/Data/phisatnet_clouds.zarr": f"{LUSTRE_BASE}/phisatnet_clouds.zarr",
    "/Data/burned_area.zarr": f"{LUSTRE_BASE}/burned_area.zarr",
    "/Data/anomaly_detection.zarr": f"{LUSTRE_BASE}/anomaly_detection.zarr",
    "/Data/fire.zarr": f"{LUSTRE_BASE}/fire.zarr",
    "/home/gdaga/pretrained_weights/": f"{LUSTRE_BASE}/pretrained_weights/",
    "/Data/phi2FM_n_shot": f"{LUSTRE_BASE}/phi2FM_models",
    "/Data/phi2FM_models": f"{LUSTRE_BASE}/phi2FM_models",
}

def update_paths(config):
    """Recursively update paths in configuration dictionary."""
    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, str):
                # Check each path mapping
                for old_path, new_path in PATH_MAPPINGS.items():
                    if old_path in value:
                        config[key] = value.replace(old_path, new_path)
            elif isinstance(value, (dict, list)):
                update_paths(value)
    elif isinstance(config, list):
        for i, item in enumerate(config):
            if isinstance(item, str):
                for old_path, new_path in PATH_MAPPINGS.items():
                    if old_path in item:
                        config[i] = item.replace(old_path, new_path)
            elif isinstance(item, (dict, list)):
                update_paths(item)
    return config

def copy_and_update_configs():
    """Copy configuration files and update paths."""
    
    # Task directories to copy
    tasks = ["anomaly_detection", "fire", "lpl_burned_area", 
             "phisatnet_clouds", "roads", "worldfloods"]
    
    total_copied = 0
    
    for task in tasks:
        source_task_dir = SOURCE_DIR / task
        target_task_dir = TARGET_DIR / task
        
        if not source_task_dir.exists():
            print(f"⚠️  Source directory not found: {source_task_dir}")
            continue
        
        # Create target directory
        target_task_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each YAML file
        for yaml_file in source_task_dir.glob("*.yml"):
            try:
                # Read original config
                with open(yaml_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Update paths
                config = update_paths(config)
                
                # Write updated config
                target_file = target_task_dir / yaml_file.name
                with open(target_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                
                print(f"✓ Created: {target_file}")
                total_copied += 1
                
            except Exception as e:
                print(f"✗ Error processing {yaml_file}: {e}")
    
    print(f"\n✓ Successfully copied and updated {total_copied} configuration files")
    print(f"✓ Configuration files are in: {TARGET_DIR.absolute()}")

if __name__ == "__main__":
    copy_and_update_configs()
