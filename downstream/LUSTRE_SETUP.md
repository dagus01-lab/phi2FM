# Lustre Cluster Setup - Summary

## Overview

Created a complete setup for running training jobs on the Lustre cluster with base path `/lustre/projects/1001/gdaga/home`.

## What Was Created

### 1. Configuration Files (`args/lustre/`)
- **60 YAML configuration files** organized by task
- All paths updated to use `/lustre/projects/1001/gdaga/home` as base

**Tasks:**
- `anomaly_detection/` - 11 configs
- `fire/` - 11 configs  
- `lpl_burned_area/` - 13 configs
- `phisatnet_clouds/` - 11 configs
- `roads/` - 2 configs
- `worldfloods/` - 12 configs

**Path Updates:**
- Datasets: `/lustre/projects/1001/gdaga/home/{task}.zarr`
- Pretrained weights: `/lustre/projects/1001/gdaga/home/pretrained_weights/`
- Output models: `/lustre/projects/1001/gdaga/home/phi2FM_models`

### 2. PBS Job Scripts (`pbs_scripts/`)
- **64 PBS scripts** for submitting training jobs
- Configured for debug queue (`gpu4_dbg`) with 5-minute walltime
- All scripts use 4 GPUs, 96 CPUs, 739GB memory

**PBS Configuration:**
```bash
#PBS -P 1001
#PBS -N {JOB_NAME}
#PBS -q gpu4_dbg
#PBS -l walltime=00:05:00
#PBS -l select=1:ngpus=4:ncpus=96:mem=739g
```

### 3. Helper Scripts

**`create_lustre_configs.py`**
- Copies configs from `args/finetune_FMs/` to `args/lustre/`
- Updates all paths to use Lustre base path
- Run: `python create_lustre_configs.py`

**`generate_pbs_scripts.py`**
- Generates PBS scripts for all task/model combinations
- Supports custom queue and walltime
- Run: `python generate_pbs_scripts.py --queue gpu4 --walltime 02:00:00`

**`submit_jobs.sh`**
- Master script for job submission
- Supports various submission patterns

## Quick Start Guide

### 1. Verify Lustre Setup

Ensure your data is in the correct location:
```bash
/lustre/projects/1001/gdaga/home/
├── worldfloods.zarr
├── phisatnet_clouds.zarr
├── phileo-bench_roads.zarr
├── burned_area.zarr
├── fire.zarr
├── anomaly_detection.zarr
├── pretrained_weights/
│   ├── resnet50_seco_geo_1m_200.pth
│   ├── seco_resnet50_1m.ckpt
│   └── ... (other model weights)
└── phi2FM/
    └── downstream/
```

### 2. Submit Jobs

**Submit a single job:**
```bash
cd /lustre/projects/1001/gdaga/home/phi2FM/downstream
qsub pbs_scripts/worldfloods_caco.sh
```

**Using the helper script:**
```bash
# List all available scripts
./submit_jobs.sh --list

# Submit all jobs for worldfloods
./submit_jobs.sh --task worldfloods

# Submit all jobs for caco model
./submit_jobs.sh --model caco

# Submit a specific job
./submit_jobs.sh --script worldfloods_caco.sh

# Submit all jobs (60 total)
./submit_jobs.sh --all

# Check job status
./submit_jobs.sh --status
```

### 3. Monitor Jobs

```bash
# Check your jobs
qstat -u $USER

# Check specific job details
qstat -f <job_id>

# View job output (after completion)
cat WF_CACO.o<job_id>
cat WF_CACO.e<job_id>
```

## Dataset Requirements

Download datasets from HuggingFace using the provided script:

```bash
cd /lustre/projects/1001/gdaga/home/phi2FM/downloads
python hf_download.py
```

**Required datasets:**
- `ESA-PhiLab-Edge/OEOBench-Burnt_Area_Dataset` → `burned_area.zarr`
- `ESA-PhiLab-Edge/OEOBench-WorldFloods` → `worldfloods.zarr`
- `ESA-PhiLab-Edge/OEOBench-AcquaAnom` → `anomaly_detection.zarr`
- `ESA-PhiLab-Edge/OEOBench-BurnScape` → `fire.zarr`
- Cloud and roads datasets (check specific sources)

## Configuration for Production

For production runs (not testing):

1. **Update PBS scripts with longer walltime:**
   ```bash
   python generate_pbs_scripts.py --queue gpu4 --walltime 02:00:00
   ```

2. **Modify individual scripts if needed:**
   Edit `pbs_scripts/{task}_{model}.sh` to change:
   - Queue: Change `gpu4_dbg` to `gpu4`
   - Walltime: Increase as needed
   - Resources: Adjust GPUs/CPUs/memory

3. **Batch size adjustments:**
   Edit YAML files in `args/lustre/{task}/` to adjust batch sizes based on GPU memory

## Task Specifications

| Task | Type | Output Channels | Loss Function |
|------|------|-----------------|---------------|
| worldfloods | Segmentation | 3 | Cross-Entropy |
| phisatnet_clouds | Classification | 5 → 2 (aggregated) | Cross-Entropy |
| roads | Regression | 1 | MSE |
| lpl_burned_area | Segmentation | 2 | Cross-Entropy |
| fire | Classification | 2 | Cross-Entropy |
| anomaly_detection | Classification | 2 | Cross-Entropy |

## Important Notes

1. **Label Mapping**: 
   - Clouds task uses label aggregation: `{0:0, 1:0, 2:0, 3:0, 4:1}`
   - Landcover task uses class remapping: `{10:0, 20:1, ..., 100:10}`

2. **Environment**:
   - All scripts assume `esa-phisatnet` conda environment
   - Located at `/lustre/projects/1001/miniconda3/`

3. **Debug Queue**:
   - Current scripts use `gpu4_dbg` for quick testing
   - Limited to 5-minute walltime
   - Change to `gpu4` for production

4. **Output Location**:
   - Models saved to: `/lustre/projects/1001/gdaga/home/phi2FM_models`
   - Check this directory for trained models

## Files Created

```
downstream/
├── args/lustre/                    # 60 config files
├── pbs_scripts/                    # 64 PBS scripts + README
├── create_lustre_configs.py        # Config generator
├── generate_pbs_scripts.py         # PBS script generator
├── submit_jobs.sh                  # Job submission helper
└── LUSTRE_SETUP.md                 # This file
```

## Troubleshooting

**Jobs fail immediately:**
- Check dataset paths exist
- Verify conda environment is activated
- Review error file: `cat <JOB_NAME>.e<job_id>`

**Out of memory errors:**
- Reduce `batch_size` in YAML config
- Reduce `num_workers` in YAML config

**Dataset not found:**
- Verify zarr files exist in `/lustre/projects/1001/gdaga/home/`
- Check permissions: `ls -la /lustre/projects/1001/gdaga/home/`

**Jobs queued but not running:**
- Check queue status: `qstat -Q`
- May need to wait for resources
- Consider using different queue

## Next Steps

1. ✅ Configuration files created
2. ✅ PBS scripts generated
3. ⏳ Download/verify datasets on Lustre
4. ⏳ Copy pretrained weights to Lustre
5. ⏳ Test with debug jobs (5 minutes)
6. ⏳ Update to production settings
7. ⏳ Run full training experiments

---

**Generated**: November 13, 2025
**Base Path**: `/lustre/projects/1001/gdaga/home`
**Total Configs**: 60
**Total PBS Scripts**: 64
