#!/usr/bin/env python3
"""
Script to expand configuration files with n_shot lists into separate files.
Creates individual configs for each n_shot value and freeze_pretrained combination.
"""

import os
import yaml
from pathlib import Path
from itertools import product

# Base directories
LUSTRE_DIR = Path("args/lustre")
EXPANDED_DIR = Path("args/lustre_expanded")

def load_yaml_preserving_comments(file_path):
    """Load YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data, file_path):
    """Save YAML file."""
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

def expand_config(config_path):
    """Expand a config file if it has n_shot as a list."""
    
    # Load config
    config = load_yaml_preserving_comments(config_path)
    
    # Get n_shot and freeze_pretrained values
    n_shot = config.get('n_shot', 0)
    freeze_pretrained = config.get('freeze_pretrained', True)
    
    # Check if n_shot is a list
    if isinstance(n_shot, list):
        n_shot_values = n_shot
    else:
        n_shot_values = [n_shot]
    
    # Check if freeze_pretrained is a list (though uncommon)
    if isinstance(freeze_pretrained, list):
        freeze_values = freeze_pretrained
    else:
        freeze_values = [freeze_pretrained]
    
    # If only single values, no expansion needed
    if len(n_shot_values) == 1 and len(freeze_values) == 1:
        return [(config, None)]
    
    # Create expanded configs
    expanded_configs = []
    
    for n_shot_val, freeze_val in product(n_shot_values, freeze_values):
        # Create a copy of the config
        new_config = config.copy()
        new_config['n_shot'] = n_shot_val
        new_config['freeze_pretrained'] = freeze_val
        if 'warmp_steps' in new_config:
            new_config['warmup_steps'] = new_config['warmp_steps']
            del new_config['warmup_steps']  # Remove warmup_steps if present
        # Update experiment name to include n_shot and freeze info
        if 'experiment_name' in new_config:
            base_name = new_config['experiment_name']
            freeze_suffix = "frozen" if freeze_val else "unfrozen"
            new_config['experiment_name'] = f"{base_name}_nshot{n_shot_val}_{freeze_suffix}"
        
        # Create suffix for filename
        freeze_suffix = "frozen" if freeze_val else "unfrozen"
        suffix = f"_nshot{n_shot_val}_{freeze_suffix}"
        
        expanded_configs.append((new_config, suffix))
    
    return expanded_configs

def process_all_configs():
    """Process all configuration files in lustre directory."""
    
    # Create expanded directory
    EXPANDED_DIR.mkdir(exist_ok=True)
    
    total_original = 0
    total_expanded = 0
    
    # Process each task directory (sorted for consistent output)
    for task_dir in sorted(LUSTRE_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        
        task_name = task_dir.name
        expanded_task_dir = EXPANDED_DIR / task_name
        expanded_task_dir.mkdir(exist_ok=True)
        
        print(f"\n📁 Processing task: {task_name}")
        
        # Process each YAML file in the task directory
        for yaml_file in task_dir.glob("*.yml"):
            total_original += 1
            model_name = yaml_file.stem
            
            try:
                expanded_configs = expand_config(yaml_file)
                
                if len(expanded_configs) == 1 and expanded_configs[0][1] is None:
                    # No expansion needed, just copy
                    output_file = expanded_task_dir / yaml_file.name
                    save_yaml(expanded_configs[0][0], output_file)
                    print(f"  ✓ {model_name}.yml (no expansion needed)")
                    total_expanded += 1
                else:
                    # Save expanded configs
                    for config, suffix in expanded_configs:
                        output_file = expanded_task_dir / f"{model_name}{suffix}.yml"
                        save_yaml(config, output_file)
                        total_expanded += 1
                    
                    print(f"  ✓ {model_name}.yml → {len(expanded_configs)} configs")
                
            except Exception as e:
                print(f"  ✗ Error processing {yaml_file}: {e}")
    
    print(f"\n{'='*60}")
    print(f"✓ Processing complete!")
    print(f"  Original configs: {total_original}")
    print(f"  Expanded configs: {total_expanded}")
    print(f"  Output directory: {EXPANDED_DIR.absolute()}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    process_all_configs()
