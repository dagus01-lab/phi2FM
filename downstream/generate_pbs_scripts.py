#!/usr/bin/env python3
"""
Script to generate PBS job scripts for expanded configuration files.
Creates one PBS script per expanded configuration.
"""

import os
from pathlib import Path

# PBS template
PBS_TEMPLATE = """#!/bin/bash
#PBS -N {job_name}
#PBS -q {queue}
#PBS -l walltime={walltime}
#PBS -l select=1:ngpus=4:ncpus=96:mem=739g

source /lustre/projects/1001/miniconda3/bin/activate
conda activate esa-phisatnet

cd /lustre/projects/1001/gdaga/home/phi2FM/downstream

python training_script.py -r "args/lustre_expanded/{task}/{config_file}"
"""

# Configuration
TASKS = {
    "worldfloods": "WF",
    "phisatnet_clouds": "CLD",
    "roads": "RD",
    "building": "BD",
    "lpl_burned_area": "BA",
    "fire": "FR",
    "anomaly_detection": "AD"
}

def generate_pbs_scripts(queue="gpu4_dbg", walltime="00:05:00"):
    """Generate PBS scripts for all expanded configuration files."""
    
    pbs_dir = Path("pbs_scripts_expanded")
    pbs_dir.mkdir(exist_ok=True)
    
    config_dir = Path("args/lustre_expanded")
    
    if not config_dir.exists():
        print(f"Error: {config_dir} does not exist. Run expand_configs.py first.")
        return
    
    generated = 0
    
    for task, task_abbr in TASKS.items():
        task_config_dir = config_dir / task
        
        if not task_config_dir.exists():
            continue
        
        # Get all config files for this task
        config_files = sorted(task_config_dir.glob("*.yml"))
        
        print(f"\n📁 {task}: {len(config_files)} configs")
        
        for config_file in config_files:
            # Extract model and parameters from filename
            # e.g., caco_nshot50_frozen.yml
            config_name = config_file.stem  # without .yml
            
            # Create job name (truncated if needed for PBS limits)
            job_name = f"{task_abbr}_{config_name}"
            if len(job_name) > 15:  # PBS job name limit
                # Shorten: e.g., WF_caco_n50_f
                parts = config_name.split('_')
                model = parts[0][:4]  # First 4 chars of model
                nshot = parts[1].replace('nshot', 'n') if len(parts) > 1 else ''
                freeze = 'f' if 'frozen' in config_name else 'u'
                job_name = f"{task_abbr}_{model}_{nshot}_{freeze}"
            
            # Generate PBS script content
            pbs_content = PBS_TEMPLATE.format(
                job_name=job_name,
                queue=queue,
                walltime=walltime,
                task=task,
                config_file=config_file.name
            )
            
            # Write PBS script
            pbs_file = pbs_dir / f"{task}_{config_name}.sh"
            with open(pbs_file, 'w') as f:
                f.write(pbs_content)
            
            # Make executable
            os.chmod(pbs_file, 0o755)
            
            generated += 1
        
        print(f"  ✓ Generated {len(config_files)} PBS scripts")
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully generated {generated} PBS scripts")
    print(f"✓ Scripts are in: {pbs_dir.absolute()}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate PBS job scripts for expanded configs")
    parser.add_argument("--queue", default="gpu4_dbg", 
                       help="PBS queue (default: gpu4_dbg)")
    parser.add_argument("--walltime", default="00:05:00",
                       help="Wall time (default: 00:05:00)")
    
    args = parser.parse_args()
    
    generate_pbs_scripts(queue=args.queue, walltime=args.walltime)
