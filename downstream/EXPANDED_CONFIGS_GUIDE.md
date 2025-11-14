# Expanded Configuration and PBS Scripts Setup

## Overview

This setup contains **296 individual configuration files** and corresponding PBS scripts, with each configuration having:
- A single `n_shot` value (50, 100, 500, 1000, or 5000)
- A single `freeze_pretrained` value (true)

## Directory Structure

```
downstream/
├── args/
│   ├── lustre/                      # Original configs (60 files)
│   └── lustre_expanded/             # Expanded configs (296 files)
│       ├── anomaly_detection/       # 55 configs
│       ├── fire/                    # 55 configs
│       ├── lpl_burned_area/         # 61 configs (includes phisatnet with n_shot=0)
│       ├── phisatnet_clouds/        # 55 configs
│       ├── roads/                   # 10 configs
│       └── worldfloods/             # 60 configs
│
├── pbs_scripts_expanded/            # PBS scripts (296 files)
│   ├── worldfloods_*.sh
│   ├── phisatnet_clouds_*.sh
│   ├── roads_*.sh
│   ├── lpl_burned_area_*.sh
│   ├── fire_*.sh
│   └── anomaly_detection_*.sh
│
├── expand_configs.py                # Script that created expanded configs
├── generate_pbs_scripts_expanded.py # Script that created PBS scripts
└── submit_jobs.sh                   # Master submission script
```

## Configuration Naming Convention

Expanded configs follow this pattern:
```
{model}_nshot{value}_frozen.yml
```

Examples:
- `caco_nshot50_frozen.yml`
- `seco_nshot1000_frozen.yml`
- `phisatnet_nshot5000_frozen.yml`

## PBS Script Details

Each PBS script:
- Uses absolute path with `qsub /path/to/script.sh`
- Points to corresponding config in `args/lustre_expanded/`
- Has a unique job name (e.g., `WF_caco_n50_f`)
- Configured for debug queue (5 minutes)

Example PBS script:
```bash
#!/bin/bash
#PBS -P 1001
#PBS -N WF_caco_n50_f
#PBS -q gpu4_dbg
#PBS -l walltime=00:05:00
#PBS -l select=1:ngpus=4:ncpus=96:mem=739g

source /lustre/projects/1001/miniconda3/bin/activate
conda activate esa-phisatnet
cd /lustre/projects/1001/gdaga/home/phi2FM/downstream
python training_script.py -r "args/lustre_expanded/worldfloods/caco_nshot50_frozen.yml"
```

## Using submit_jobs.sh

The submission script now works with expanded configurations.

### Basic Commands

**Count all scripts:**
```bash
./submit_jobs.sh --count
```

**List sample scripts:**
```bash
./submit_jobs.sh --list
```

**Check job status:**
```bash
./submit_jobs.sh --status
```

### Submission Options

**1. Submit by Task (all n_shot values for one task):**
```bash
./submit_jobs.sh --task worldfloods        # Submits 60 jobs
./submit_jobs.sh --task roads              # Submits 10 jobs
./submit_jobs.sh --task phisatnet_clouds   # Submits 55 jobs
```

**2. Submit by Model (all n_shot values for one model across all tasks):**
```bash
./submit_jobs.sh --model caco              # Submits 30 jobs
./submit_jobs.sh --model seco              # Submits 30 jobs
./submit_jobs.sh --model phisatnet         # Submits 31 jobs
```

**3. Submit by N-Shot (specific n_shot across all tasks and models):**
```bash
./submit_jobs.sh --nshot 50                # Submits 59 jobs
./submit_jobs.sh --nshot 1000              # Submits 59 jobs
./submit_jobs.sh --nshot 5000              # Submits 59 jobs
```

**4. Submit a Single Job:**
```bash
./submit_jobs.sh --script worldfloods_caco_nshot50_frozen.sh
```

**5. Submit ALL Jobs (use with caution!):**
```bash
./submit_jobs.sh --all                     # Submits ALL 296 jobs!
```

## Task-Specific Breakdown

| Task | Total Scripts | N-Shot Values | Models |
|------|--------------|---------------|--------|
| worldfloods | 60 | 50,100,500,1000,5000 | 12 models |
| phisatnet_clouds | 55 | 50,100,500,1000,5000 | 11 models |
| lpl_burned_area | 61 | 50,100,500,1000,5000 + 1×n_shot=0 | 12 models + 1 special |
| fire | 55 | 50,100,500,1000,5000 | 11 models |
| anomaly_detection | 55 | 50,100,500,1000,5000 | 11 models |
| roads | 10 | 50,100,500,1000,5000 | 2 models |

## N-Shot Distribution

| N-Shot Value | Number of Configs |
|--------------|-------------------|
| 50 | 59 |
| 100 | 59 |
| 500 | 59 |
| 1000 | 59 |
| 5000 | 59 |
| 0 | 1 (lpl_burned_area/phisatnet only) |

## Practical Examples

### Example 1: Test one configuration
```bash
# Test single job first
./submit_jobs.sh --script worldfloods_caco_nshot50_frozen.sh

# Check it's running
./submit_jobs.sh --status
```

### Example 2: Run all n_shot=50 experiments
```bash
# Submit all 50-shot experiments across all tasks/models
./submit_jobs.sh --nshot 50

# Monitor progress
watch -n 30 'qstat -u $USER'
```

### Example 3: Run complete worldfloods experiments
```bash
# Submit all worldfloods experiments (all models, all n_shot values)
./submit_jobs.sh --task worldfloods

# This submits 60 jobs:
#   - 12 models × 5 n_shot values = 60
```

### Example 4: Run all experiments for caco model
```bash
# Submit caco across all tasks
./submit_jobs.sh --model caco

# This submits 30 jobs:
#   - 6 tasks × 5 n_shot values = 30
```

### Example 5: Batch submission with delay
```bash
# Submit 5 jobs at a time with delays
for nshot in 50 100 500 1000 5000; do
    echo "Submitting n_shot=$nshot experiments..."
    ./submit_jobs.sh --nshot $nshot
    echo "Waiting 5 minutes before next batch..."
    sleep 300
done
```

## Regenerating Scripts

If you need to regenerate the configurations or PBS scripts:

**1. Regenerate expanded configs:**
```bash
python expand_configs.py
```

**2. Regenerate PBS scripts:**
```bash
# Default (debug queue, 5 minutes)
python generate_pbs_scripts_expanded.py

# Production settings
python generate_pbs_scripts_expanded.py --queue gpu4 --walltime 02:00:00
```

## Production Configuration

For production runs (not debug):

**1. Regenerate PBS scripts with production settings:**
```bash
python generate_pbs_scripts_expanded.py --queue gpu4 --walltime 02:00:00
```

**2. Or manually edit individual scripts:**
```bash
# Change in the PBS script:
#PBS -q gpu4           # Instead of gpu4_dbg
#PBS -l walltime=02:00:00  # Instead of 00:05:00
```

## Monitoring Jobs

**Check all your jobs:**
```bash
qstat -u $USER
```

**Count running jobs:**
```bash
qstat -u $USER | grep " R " | wc -l
```

**Count queued jobs:**
```bash
qstat -u $USER | grep " Q " | wc -l
```

**Watch jobs in real-time:**
```bash
watch -n 10 'qstat -u $USER'
```

**Check specific job details:**
```bash
qstat -f <job_id>
```

## Output Files

After job completion, check:
- Standard output: `{JOB_NAME}.o{job_id}`
- Standard error: `{JOB_NAME}.e{job_id}`
- Model outputs: `/lustre/projects/1001/gdaga/home/phi2FM_models/`

## Important Notes

1. **Debug Queue Limits**: Current scripts use `gpu4_dbg` with 5-minute walltime
2. **Job Limits**: Check your HPC facility's limits on concurrent jobs
3. **Storage**: 296 training runs will generate significant output
4. **Monitoring**: Use `--count` and `--status` frequently
5. **Testing**: Always test with `--script` for a single job first

## Summary Statistics

- **Total Configurations**: 296
- **Total PBS Scripts**: 296
- **Tasks**: 6
- **Models**: 12 unique models
- **N-Shot Values**: 5 main values (50, 100, 500, 1000, 5000) + 1 special (0)
- **Estimated Runtime**: Depends on task complexity and hardware

---

**Generated**: November 13, 2025  
**Configuration Path**: `args/lustre_expanded/`  
**PBS Scripts Path**: `pbs_scripts_expanded/`  
**Submission Script**: `submit_jobs.sh`
