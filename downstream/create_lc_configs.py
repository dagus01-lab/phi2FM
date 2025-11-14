#!/usr/bin/env python3
"""
Script to create land cover (lc) configuration files for all models.
"""

import os
import yaml
from pathlib import Path

# Base directory
FINETUNE_DIR = Path("args/finetune_FMs")
LC_DIR = FINETUNE_DIR / "phileo_bench-lc"

# Ensure LC directory exists
LC_DIR.mkdir(parents=True, exist_ok=True)

# Land cover task configuration
LC_CONFIG = {
    "downstream_task": "lc",
    "output_channels": 11,
    "input_channels_default": 8,  # Most models use 8 channels
    "data_path": "/Data/phisatnet/phileo-bench_lc.zarr",
}

# Model-specific configurations
MODEL_CONFIGS = {
    # PhiSatNet and UNet Myriad2
    "phisatnet": {
        "model_name": "phisatnet",
        "experiment_name": "phisatnet_downstream/lc",
        "pretrained_model_path": "/home/gdaga/pretrained_weights/phisat2net_geoaware_best.pt",
        "freeze_pretrained": True,
        "input_channels": 8,
        "pad_bands": None,
        "input_size": 224,
    },
    "unet_myriad2_baseline": {
        "model_name": "unet_myriad2_baseline",
        "experiment_name": "unet_myriad2_baseline/lc",
        "pretrained_model_path": "/home/gdaga/hydranet/experiments/distillation_production_20251113_001051/baseline/config_baseline/best_downstream_model.pt",
        "freeze_pretrained": True,
        "input_channels": 8,
        "pad_bands": 8,
        "input_size": 224,
    },
    
    # Foundation models
    "satmae": {
        "model_name": "SatMAE",
        "experiment_name": "satmae_downstream/lc",
        "pretrained_model_path": "/home/gdaga/pretrained_weights/satmae-vitlarge-800.pth",
        "freeze_pretrained": True,
        "input_channels": 10,
        "pad_bands": 10,
        "input_size": 224,
    },
    "prithvi": {
        "model_name": "prithvi",
        "experiment_name": "prithvi_downstream/lc",
        "pretrained_model_path": "/home/gdaga/pretrained_weights/Prithvi_100M.pt",
        "freeze_pretrained": True,
        "input_channels": 6,
        "pad_bands": 6,
        "input_size": 224,
    },
    "seco": {
        "model_name": "seasonal_contrast",
        "experiment_name": "phi2_seasonal_contrast/lc",
        "pretrained_model_path": "/home/gdaga/pretrained_weights/seco_resnet50_1m.ckpt",
        "freeze_pretrained": True,
        "input_channels": 10,
        "pad_bands": 10,
        "input_size": 224,
    },
    
    # SSL4EO models
    "moco": {
        "model_name": "moco",
        "experiment_name": "moco_downstream/lc",
        "pretrained_model_path": "/home/gdaga/pretrained_weights/moco_resnet50_200ep.pth.tar",
        "freeze_pretrained": True,
        "input_channels": 13,
        "pad_bands": 13,
        "input_size": 224,
    },
    "dino": {
        "model_name": "dino",
        "experiment_name": "dino_downstream/lc",
        "pretrained_model_path": "/home/gdaga/pretrained_weights/dino_resnet50_ep200.pth",
        "freeze_pretrained": True,
        "input_channels": 13,
        "pad_bands": 13,
        "input_size": 224,
    },
    "gassl": {
        "model_name": "gassl",
        "experiment_name": "gassl_downstream/lc",
        "pretrained_model_path": "/home/gdaga/pretrained_weights/gassl_resnet50.pth",
        "freeze_pretrained": True,
        "input_channels": 13,
        "pad_bands": 13,
        "input_size": 224,
    },
    "caco": {
        "model_name": "caco",
        "experiment_name": "caco_downstream/lc",
        "pretrained_model_path": "/home/gdaga/pretrained_weights/caco_resnet50.pth",
        "freeze_pretrained": True,
        "input_channels": 13,
        "pad_bands": 13,
        "input_size": 224,
    },
    
    # Other models
    "geoaware": {
        "model_name": "GeoAware_core_nano",
        "experiment_name": "geoaware_downstream/lc",
        "pretrained_model_path": "/home/gdaga/pretrained_weights/geoaware_encoder_nano_mse.pt",
        "freeze_pretrained": True,
        "input_channels": 10,
        "pad_bands": 10,
        "input_size": 128,
    },
    "vit": {
        "model_name": "vit_cnn_gc",
        "experiment_name": "vit_downstream/lc",
        "pretrained_model_path": "/home/gdaga/pretrained_weights/vit_large_gc_pretrained.pt",
        "freeze_pretrained": True,
        "input_channels": 10,
        "pad_bands": 10,
        "input_size": 224,
    },
    "uniphi": {
        "model_name": "phileo_precursor",
        "experiment_name": "uniphi_downstream/lc",
        "pretrained_model_path": "/home/gdaga/pretrained_weights/phileo_precursor.ckpt",
        "freeze_pretrained": True,
        "input_channels": 10,
        "pad_bands": 10,
        "input_size": 128,
    },
}

def create_lc_config(model_key, model_config):
    """Create land cover config for a specific model."""
    
    config = {
        "experiment_name": model_config["experiment_name"],
        "downstream_task": LC_CONFIG["downstream_task"],
        "model_name": model_config["model_name"],
        "augmentations": True,
        "batch_size": 32 if model_config["input_size"] == 224 else 16,
        "model_device": "cuda",
        "generator_device": "cuda",
        "num_workers": 16,
        "early_stop": 15,
        "epochs": 200,
        "input_channels": model_config["input_channels"],
        "output_channels": LC_CONFIG["output_channels"],
        "input_size": model_config["input_size"],
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
        "pretrained_model_path": model_config["pretrained_model_path"],
        "freeze_pretrained": model_config["freeze_pretrained"],
        "data_path_128_10m": LC_CONFIG["data_path"],
        "data_path_224_10m": LC_CONFIG["data_path"],
        "data_path_224_30m": LC_CONFIG["data_path"],
        "train_mode": "train_test",
        "downstream_model_path": None,
        "data_path_inference_128": LC_CONFIG["data_path"],
        "data_path_inference_224": LC_CONFIG["data_path"],
        "output_path": "/Data/phi2FM_n_shot",
        "data_parallel": False,
        "device_ids": [0],
        "pad_bands": model_config["pad_bands"],
        "only_get_datasets": False,
        "wandb": True,
        "patch_size": model_config["input_size"],
        "shrink_val_set": 0.1,
    }
    
    return config

def create_all_lc_configs():
    """Create land cover configs for all models."""
    
    created_count = 0
    
    for model_key, model_config in MODEL_CONFIGS.items():
        # Create config
        config = create_lc_config(model_key, model_config)
        
        # Save config file
        output_file = LC_DIR / f"{model_key}.yml"
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Created: {output_file}")
        created_count += 1
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully created {created_count} land cover configurations")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    create_all_lc_configs()
