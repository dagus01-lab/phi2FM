# UNet Myriad2 Model Configuration Summary

## Overview
Successfully created UNet Myriad2 baseline model configurations for all downstream tasks in the phi2FM project.

## Configuration Details

### Model Architecture
- **Model Name**: `unet_myriad2_baseline` (segmentation) or `unet_myriad2_baseline_classifier` (classification)
- **Base Filters**: 16
- **Depth**: 3 encoder/decoder levels
- **Input Channels**: 8 (Sentinel-2 bands)
- **Input Size**: 224x224
- **Pretrained Weights**: `/home/gdaga/hydranet/experiments/distillation_production_20251113_001051/baseline/config_baseline/best_downstream_model.pt`

### Downstream Tasks Configuration

| Task | Output Channels | Model Variant | Freeze Pretrained | Data Path |
|------|----------------|---------------|-------------------|-----------|
| anomaly_detection | 9 | segmentation | False | /Data/anomaly_detection/marine_area_dataset.zarr |
| fire | 4 | classification | False | /Data/fire_dataset/fire_dataset.zarr |
| lpl_burned_area | 4 | segmentation | True | /Data/lpl_burned_area/burned.zarr |
| phileo_bench-lc | 11 | segmentation | True | /Data/phisatnet/phileo-bench_lc.zarr |
| phisatnet_clouds | 5 | segmentation | True | /Data/phisatnet_clouds.zarr |
| roads | 1 | segmentation | True | /Data/phisatnet_dataset/phileo-bench_roads.zarr |
| worldfloods | 3 | segmentation | True | /Data/worldfloods/worldfloods.zarr |

### Training Configuration
- **N-shot values**: [50, 100, 500, 1000, 5000]
- **Batch size**: 16
- **Learning rate**: 0.0001
- **LR scheduler**: reduce_on_plateau
- **Epochs**: 200
- **Early stopping patience**: 15
- **Warmup**: Enabled (5 steps, gamma=10)
- **Augmentations**: Enabled
- **Visualization**: Enabled (vis_val=True)
- **WandB logging**: Enabled

## Generated Configurations

### Original Configs (finetune_FMs)
- Location: `downstream/args/finetune_FMs/`
- **7 tasks** × **1 base config** = **7 configs**

### Lustre Configs (HPC Environment)
- Location: `downstream/args/lustre/`
- Paths updated to: `/lustre/projects/1001/gdaga/home/`
- **7 tasks** × **1 config per task** = **7 configs**

### Expanded Configs (Individual n_shot)
- Location: `downstream/args/lustre_expanded/`
- **7 tasks** × **5 n_shot values** = **35 configs**

Breakdown by task:
- anomaly_detection: 5 configs (unfrozen)
- fire: 5 configs (unfrozen, classifier variant)
- lpl_burned_area: 5 configs (frozen)
- phileo_bench-lc: 5 configs (frozen) **[NEW]**
- phisatnet_clouds: 5 configs (frozen)
- roads: 5 configs (frozen)
- worldfloods: 5 configs (frozen)

## HPC Path Mappings

### Data Paths
```
Local → Lustre
/Data/anomaly_detection/marine_area_dataset.zarr → /lustre/projects/1001/gdaga/home/anomaly_detection.zarr
/Data/fire_dataset/fire_dataset.zarr → /lustre/projects/1001/gdaga/home/fire.zarr
/Data/lpl_burned_area/burned.zarr → /lustre/projects/1001/gdaga/home/lpl_burned_area.zarr
/Data/phisatnet/phileo-bench_lc.zarr → /lustre/projects/1001/gdaga/home/phileo-bench_lc.zarr
/Data/phisatnet_clouds.zarr → /lustre/projects/1001/gdaga/home/phisatnet_clouds.zarr
/Data/phisatnet_dataset/phileo-bench_roads.zarr → /lustre/projects/1001/gdaga/home/phileo-bench_roads.zarr
/Data/worldfloods/worldfloods.zarr → /lustre/projects/1001/gdaga/home/worldfloods.zarr
```

### Model Paths
```
Local pretrained weights:
/home/gdaga/hydranet/experiments/distillation_production_20251113_001051/baseline/config_baseline/best_downstream_model.pt

→ Lustre pretrained weights:
/lustre/projects/1001/gdaga/home/pretrained_weights/unet_myriad2_baseline.pt

Output models:
/Data/phi2FM_n_shot → /lustre/projects/1001/gdaga/home/phi2FM_models
```

## Usage Examples

### Local Training
```bash
cd /home/gdaga/phi2FM/downstream
python training_script.py -r args/finetune_FMs/lpl_burned_area/unet_myriad2_baseline.yml
```

### HPC Training (Single n_shot)
```bash
cd /home/gdaga/phi2FM/downstream
python training_script.py -r args/lustre_expanded/lpl_burned_area/unet_myriad2_baseline_nshot5000_frozen.yml
```

### Batch Submission (All n_shot values for a task)
Use your HPC scheduler to submit all configs for a specific task:
```bash
# Example for lpl_burned_area
for config in args/lustre_expanded/lpl_burned_area/unet_myriad2_baseline_nshot*_frozen.yml; do
    sbatch your_job_script.sh $config
done
```

## Model Comparison

The UNet Myriad2 model follows the same configuration pattern as PhiSatNet for consistency:
- Uses MODELS_224 (224x224 input resolution)
- Uses CNN_PRETRAINED_LIST
- Uses same training loops (e.g., TrainSegmentationBurned for burned_area)
- Uses same dataloader configurations
- Supports both segmentation and classification tasks

## Files Created

### Files Created

### Scripts
1. `create_unet_myriad2_configs.py` - Creates base configs for all tasks
2. `create_lc_configs.py` - Creates land cover configs for all models **[NEW]**
3. `create_lustre_configs.py` (updated) - Converts paths for HPC environment
4. `expand_configs.py` - Expands n_shot lists into individual configs

### Configuration Directories
1. `args/finetune_FMs/*/unet_myriad2_baseline.yml` - 7 base configs
2. `args/lustre/*/unet_myriad2_baseline.yml` - 7 HPC configs
3. `args/lustre_expanded/*/unet_myriad2_baseline_nshot*_*.yml` - 35 expanded configs

## Total Statistics

- **Total Tasks**: 7
- **Base Configs**: 7
- **Lustre Configs**: 7
- **Expanded Configs**: 35
- **N-shot Values**: 5 per task
- **Total Experiments**: 35 (7 tasks × 5 n_shot values)

## All Model Configurations Summary

### Complete Ecosystem
- **Total Tasks**: 7 (anomaly_detection, fire, lpl_burned_area, phileo_bench-lc, phisatnet_clouds, roads, worldfloods)
- **Total Models**: 12 (phisatnet, unet_myriad2_baseline, satmae, prithvi, seco, moco, dino, gassl, caco, geoaware, vit, uniphi)
- **Total Original Configs**: 78
- **Total Expanded Configs**: 491
- **N-shot Values**: [50, 100, 500, 1000, 5000]

## Next Steps

1. **Transfer pretrained weights to HPC**:
   ```bash
   scp /home/gdaga/hydranet/experiments/distillation_production_20251113_001051/baseline/config_baseline/best_downstream_model.pt \
       your_hpc:/lustre/projects/1001/gdaga/home/pretrained_weights/unet_myriad2_baseline.pt
   ```

2. **Transfer datasets to HPC** (if not already done):
   - Ensure all zarr datasets are available at `/lustre/projects/1001/gdaga/home/`

3. **Submit jobs**:
   - Create SLURM/PBS job scripts for your HPC scheduler
   - Submit individual experiments or batch submit all configs

4. **Monitor training**:
   - Validation images will be saved to `{output_path}/val_images/`
   - WandB logging enabled for remote monitoring
   - Check logs for model performance metrics
