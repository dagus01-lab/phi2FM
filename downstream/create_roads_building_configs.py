#!/usr/bin/env python3
"""
Script to create configuration files for all models for roads and building tasks.
Uses worldfloods configs as reference for model parameters.
"""

import yaml
from pathlib import Path

# Model configurations from worldfloods
MODEL_CONFIGS = {
    'caco': {
        'model_name': 'caco',
        'experiment_prefix': 'phi2_caco',
        'pretrained_model_path': '/lustre/projects/1001/gdaga/home/pretrained_weights/resnet50_seco_geo_1m_200.pth',
        'input_channels': 4,
        'pad_bands': 4
    },
    'seco': {
        'model_name': 'seasonal_contrast',
        'experiment_prefix': 'phi2_seasonal_contrast',
        'pretrained_model_path': '/lustre/projects/1001/gdaga/home/pretrained_weights/seco_resnet50_1m.ckpt',
        'input_channels': 10,
        'pad_bands': 10
    },
    'phisatnet': {
        'model_name': 'phisatnet',
        'experiment_prefix': 'phisatnet_downstream',
        'pretrained_model_path': '/lustre/projects/1001/gdaga/home/pretrained_weights/phisat2net_geoaware_best.pt',
        'input_channels': 8,
        'pad_bands': None
    },
    'dino': {
        'model_name': 'dino',
        'experiment_prefix': 'phi2_dino',
        'pretrained_model_path': '/lustre/projects/1001/gdaga/home/pretrained_weights/B13_rn50_dino_0099.pth',
        'input_channels': 13,
        'pad_bands': 13
    },
    'moco': {
        'model_name': 'moco',
        'experiment_prefix': 'phi2_moco',
        'pretrained_model_path': '/lustre/projects/1001/gdaga/home/pretrained_weights/B13_rn50_moco_0099.pth',
        'input_channels': 13,
        'pad_bands': 13
    },
    'gassl': {
        'model_name': 'gassl',
        'experiment_prefix': 'phi2_gassl',
        'pretrained_model_path': '/lustre/projects/1001/gdaga/home/pretrained_weights/B13_rn50_gassl_0099.pth',
        'input_channels': 13,
        'pad_bands': 3
    },
    'geoaware': {
        'model_name': 'GeoAware_mh_pred_core_nano',
        'experiment_prefix': 'phi2_geoaware',
        'pretrained_model_path': '/lustre/projects/1001/gdaga/home/pretrained_weights/GeoAware_mh_pred_core_nano_2000ep_s2_rn50.pth',
        'input_channels': 13,
        'pad_bands': 13
    },
    'prithvi': {
        'model_name': 'prithvi',
        'experiment_prefix': 'phi2_prithvi',
        'pretrained_model_path': '/lustre/projects/1001/gdaga/home/pretrained_weights/Prithvi_100M.pt',
        'input_channels': 6,
        'pad_bands': 6
    },
    'satmae': {
        'model_name': 'SatMAE',
        'experiment_prefix': 'phi2_satmae',
        'pretrained_model_path': '/lustre/projects/1001/gdaga/home/pretrained_weights/pretrain-vit-large-e199.pth',
        'input_channels': 10,
        'pad_bands': 10
    },
    'uniphi': {
        'model_name': 'phileo_precursor',
        'experiment_prefix': 'phi2_uniphi',
        'pretrained_model_path': '/lustre/projects/1001/gdaga/home/pretrained_weights/phileo_precursor_swinb_si_ckpt.pth',
        'input_channels': 12,
        'pad_bands': 12
    },
    'vit': {
        'model_name': 'vit_cnn_gc',
        'experiment_prefix': 'phi2_vit',
        'pretrained_model_path': '/lustre/projects/1001/gdaga/home/pretrained_weights/ckpt_ViT.pth',
        'input_channels': 13,
        'pad_bands': 13
    },
}

# Task configurations
TASK_CONFIGS = {
    'roads': {
        'downstream_task': 'roads',
        'output_channels': 1,
        'batch_size': 32,
        'dataset_path': '/lustre/projects/1001/gdaga/home/phileo-bench_roads.zarr'
    },
    'building': {
        'downstream_task': 'building',
        'output_channels': 1,
        'batch_size': 32,
        'dataset_path': '/lustre/projects/1001/gdaga/home/phileo-bench_building.zarr'
    }
}

def create_config(task, model_key, model_config):
    """Create a configuration dictionary for a task and model."""
    task_config = TASK_CONFIGS[task]
    
    config = {
        'experiment_name': f"{model_config['experiment_prefix']}/{task}",
        'downstream_task': task_config['downstream_task'],
        'model_name': model_config['model_name'],
        'augmentations': True,
        'batch_size': task_config['batch_size'],
        'model_device': 'cuda',
        'generator_device': 'cuda',
        'num_workers': 16,
        'early_stop': 15,
        'epochs': 200,
        'input_channels': model_config['input_channels'],
        'output_channels': task_config['output_channels'],
        'input_size': 224,
        'lr': 0.0001,
        'lr_scheduler': 'reduce_on_plateau',
        'n_shot': [50, 100, 500, 1000, 5000],
        'split_ratio': None,
        'regions': None,
        'vis_val': True,
        'warmup': True,
        'warmup_steps': 5,
        'warmup_gamma': 10,
        'min_lr': None,
        'pretrained_model_path': model_config['pretrained_model_path'],
        'freeze_pretrained': True,
        'data_path_128_10m': task_config['dataset_path'],
        'data_path_224_10m': task_config['dataset_path'],
        'data_path_224_30m': task_config['dataset_path'],
        'train_mode': 'train_test',
        'downstream_model_path': None,
        'data_path_inference_128': task_config['dataset_path'],
        'data_path_inference_224': task_config['dataset_path'],
        'output_path': '/lustre/projects/1001/gdaga/home/phi2FM_models',
        'data_parallel': False,
        'device_ids': [0],
        'pad_bands': model_config['pad_bands'],
        'only_get_datasets': False
    }
    
    return config

def generate_configs():
    """Generate configuration files for roads and building tasks."""
    
    base_dir = Path('args/lustre')
    
    total_created = 0
    
    for task in ['roads', 'building']:
        task_dir = base_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 Creating configs for {task}:")
        
        for model_key, model_config in MODEL_CONFIGS.items():
            config = create_config(task, model_key, model_config)
            
            # Save config
            output_file = task_dir / f"{model_key}.yml"
            with open(output_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            print(f"  ✓ Created: {model_key}.yml")
            total_created += 1
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully created {total_created} configuration files")
    print(f"  - roads: {len(MODEL_CONFIGS)} configs")
    print(f"  - building: {len(MODEL_CONFIGS)} configs")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    generate_configs()
