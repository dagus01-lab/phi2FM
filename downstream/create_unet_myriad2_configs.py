#!/usr/bin/env python3
"""
Script to create UNet Myriad2 configuration files for all downstream tasks.
"""

import os
import yaml
from pathlib import Path

# Base directory
FINETUNE_DIR = Path("args/finetune_FMs")

# Task configurations (task_name, output_channels, data_path, model_variant)
TASK_CONFIGS = {
    "anomaly_detection": {
        "output_channels": 9,
        "data_path": "/Data/anomaly_detection/marine_area_dataset.zarr",
        "model_name": "unet_myriad2_baseline",  # segmentation
        "freeze_pretrained": False,
    },
    "fire": {
        "output_channels": 4,
        "data_path": "/Data/fire_dataset/fire_dataset.zarr",
        "model_name": "unet_myriad2_baseline_classifier",  # classification
        "freeze_pretrained": False,
    },
    "lpl_burned_area": {
        "output_channels": 4,
        "data_path": "/Data/lpl_burned_area/burned.zarr",
        "model_name": "unet_myriad2_baseline",  # segmentation
        "freeze_pretrained": True,
    },
    "phisatnet_clouds": {
        "output_channels": 5,
        "data_path": "/Data/phisatnet_clouds.zarr",
        "model_name": "unet_myriad2_baseline",  # segmentation
        "downstream_task": "clouds",
        "freeze_pretrained": True,
    },
    "roads": {
        "output_channels": 1,
        "data_path": "/Data/phisatnet_dataset/phileo-bench_roads.zarr",
        "model_name": "unet_myriad2_baseline",  # segmentation
        "freeze_pretrained": True,
    },
    "worldfloods": {
        "output_channels": 3,
        "data_path": "/Data/worldfloods/worldfloods.zarr",
        "model_name": "unet_myriad2_baseline",  # segmentation
        "freeze_pretrained": True,
    },
}

# Pretrained model path
PRETRAINED_PATH = "/home/gdaga/hydranet/experiments/distillation_production_20251113_001051/baseline/config_baseline/best_downstream_model.pt"

def create_config(task_dir_name, task_config):
    """Create UNet Myriad2 config for a specific task."""
    
    # Get downstream task name (handle special case for clouds)
    downstream_task = task_config.get("downstream_task", task_dir_name)
    
    config = {
        "experiment_name": f"unet_myriad2_baseline/{downstream_task}",
        "downstream_task": downstream_task,
        "model_name": task_config["model_name"],
        "augmentations": True,
        "batch_size": 16,
        "model_device": "cuda",
        "generator_device": "cuda",
        "num_workers": 16,
        "early_stop": 15,
        "epochs": 200,
        "input_channels": 8,
        "output_channels": task_config["output_channels"],
        "input_size": 224,
        "lr": 0.0001,
        "lr_scheduler": "reduce_on_plateau",
        "n_shot": [50, 100, 500, 1000, 5000],
        "split_ratio": None,
        "regions": None,
        "vis_val": True,
        "warmup": True,
        "warmup_steps": 5,
        "warmup_gamma": 10,
        "min_lr": None,
        "pretrained_model_path": PRETRAINED_PATH,
        "freeze_pretrained": task_config["freeze_pretrained"],
        "data_path_128_10m": task_config["data_path"],
        "data_path_224_10m": task_config["data_path"],
        "data_path_224_30m": task_config["data_path"],
        "train_mode": "train_test",
        "downstream_model_path": None,
        "data_path_inference_128": task_config["data_path"],
        "data_path_inference_224": task_config["data_path"],
        "output_path": "/Data/phi2FM_n_shot",
        "data_parallel": False,
        "device_ids": [0],
        "wandb": True,
        "patch_size": 224,
        "shrink_val_set": 0.1,
        "pad_bands": 8,
        "only_get_datasets": False,
    }
    
    return config

def create_all_configs():
    """Create UNet Myriad2 configs for all tasks."""
    
    created_count = 0
    
    for task_dir_name, task_config in TASK_CONFIGS.items():
        # Create task directory if it doesn't exist
        task_dir = FINETUNE_DIR / task_dir_name
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Create config
        config = create_config(task_dir_name, task_config)
        
        # Save config file
        output_file = task_dir / "unet_myriad2_baseline.yml"
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Created: {output_file}")
        created_count += 1
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully created {created_count} UNet Myriad2 configurations")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    create_all_configs()
