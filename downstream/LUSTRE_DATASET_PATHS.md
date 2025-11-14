# Lustre HPC Dataset Paths

## Overview
This document lists all dataset paths required for the phi2FM downstream tasks on the Lustre HPC environment.

**Base Path**: `/lustre/projects/1001/gdaga/home/`

---

## Required Datasets

### 1. Anomaly Detection (Marine Area)
- **Local Path**: `/Data/anomaly_detection/marine_area_dataset.zarr`
- **Lustre Path**: `/lustre/projects/1001/gdaga/home/anomaly_detection.zarr`
- **Task**: `anomaly_detection`
- **Output Channels**: 9 classes
- **Description**: Marine area anomaly detection dataset

---

### 2. Fire Detection
- **Local Path**: `/Data/fire_dataset/fire_dataset.zarr`
- **Lustre Path**: `/lustre/projects/1001/gdaga/home/fire.zarr`
- **Task**: `fire`
- **Output Channels**: 4 classes
- **Description**: Fire detection classification dataset

---

### 3. Burned Area (LPL)
- **Local Path**: `/Data/lpl_burned_area/burned.zarr`
- **Lustre Path**: `/lustre/projects/1001/gdaga/home/lpl_burned_area.zarr`
- **Task**: `lpl_burned_area`
- **Output Channels**: 4 classes
- **Description**: Burned area segmentation dataset

---

### 4. Land Cover (PhilEO-Bench)
- **Local Path**: `/Data/phisatnet/phileo-bench_lc.zarr`
- **Lustre Path**: `/lustre/projects/1001/gdaga/home/phileo-bench_lc.zarr`
- **Task**: `lc` (phileo_bench-lc)
- **Output Channels**: 11 classes
- **Description**: Multi-class land cover segmentation dataset

---

### 5. Cloud Segmentation
- **Local Path**: `/Data/phisatnet_clouds.zarr`
- **Lustre Path**: `/lustre/projects/1001/gdaga/home/phisatnet_clouds.zarr`
- **Task**: `clouds`
- **Output Channels**: 5 classes
- **Description**: Cloud segmentation dataset

---

### 6. Roads Detection
- **Local Path**: `/Data/phisatnet_dataset/phileo-bench_roads.zarr`
- **Lustre Path**: `/lustre/projects/1001/gdaga/home/phileo-bench_roads.zarr`
- **Task**: `roads`
- **Output Channels**: 1 class
- **Description**: Road segmentation dataset

---

### 7. Worldfloods
- **Local Path**: `/Data/worldfloods/worldfloods.zarr`
- **Lustre Path**: `/lustre/projects/1001/gdaga/home/worldfloods.zarr`
- **Task**: `worldfloods`
- **Output Channels**: 3 classes
- **Description**: Flood detection and segmentation dataset

---

## Pretrained Model Weights

### PhiSatNet Models
- **Local Path**: `/home/gdaga/pretrained_weights/phisat2net_geoaware_best.pt`
- **Lustre Path**: `/lustre/projects/1001/gdaga/home/pretrained_weights/phisat2net_geoaware_best.pt`

### UNet Myriad2 Baseline
- **Local Path**: `/home/gdaga/hydranet/experiments/distillation_production_20251113_001051/baseline/config_baseline/best_downstream_model.pt`
- **Lustre Path**: `/lustre/projects/1001/gdaga/home/pretrained_weights/unet_myriad2_baseline.pt`

### SatMAE
- **Local Path**: `/home/gdaga/pretrained_weights/satmae-vitlarge-800.pth`
- **Lustre Path**: `/lustre/projects/1001/gdaga/home/pretrained_weights/satmae-vitlarge-800.pth`

### Prithvi
- **Local Path**: `/home/gdaga/pretrained_weights/Prithvi_100M.pt`
- **Lustre Path**: `/lustre/projects/1001/gdaga/home/pretrained_weights/Prithvi_100M.pt`

### SeasonalContrast (SECO)
- **Local Path**: `/home/gdaga/pretrained_weights/seco_resnet50_1m.ckpt`
- **Lustre Path**: `/lustre/projects/1001/gdaga/home/pretrained_weights/seco_resnet50_1m.ckpt`

### SSL4EO - MoCo
- **Local Path**: `/home/gdaga/pretrained_weights/moco_resnet50_200ep.pth.tar`
- **Lustre Path**: `/lustre/projects/1001/gdaga/home/pretrained_weights/moco_resnet50_200ep.pth.tar`

### SSL4EO - DINO
- **Local Path**: `/home/gdaga/pretrained_weights/dino_resnet50_ep200.pth`
- **Lustre Path**: `/lustre/projects/1001/gdaga/home/pretrained_weights/dino_resnet50_ep200.pth`

### SSL4EO - GASSL
- **Local Path**: `/home/gdaga/pretrained_weights/gassl_resnet50.pth`
- **Lustre Path**: `/lustre/projects/1001/gdaga/home/pretrained_weights/gassl_resnet50.pth`

### SSL4EO - CaCo
- **Local Path**: `/home/gdaga/pretrained_weights/caco_resnet50.pth`
- **Lustre Path**: `/lustre/projects/1001/gdaga/home/pretrained_weights/caco_resnet50.pth`

### GeoAware
- **Local Path**: `/home/gdaga/pretrained_weights/geoaware_encoder_nano_mse.pt`
- **Lustre Path**: `/lustre/projects/1001/gdaga/home/pretrained_weights/geoaware_encoder_nano_mse.pt`

### ViT
- **Local Path**: `/home/gdaga/pretrained_weights/vit_large_gc_pretrained.pt`
- **Lustre Path**: `/lustre/projects/1001/gdaga/home/pretrained_weights/vit_large_gc_pretrained.pt`

### UniPhi (Phileo Precursor)
- **Local Path**: `/home/gdaga/pretrained_weights/phileo_precursor.ckpt`
- **Lustre Path**: `/lustre/projects/1001/gdaga/home/pretrained_weights/phileo_precursor.ckpt`

---

## Output Paths

### Model Outputs
- **Local Path**: `/Data/phi2FM_n_shot` or `/Data/phi2FM_models`
- **Lustre Path**: `/lustre/projects/1001/gdaga/home/phi2FM_models`

---

## Quick Reference: All Dataset Paths

```bash
# Datasets
/lustre/projects/1001/gdaga/home/anomaly_detection.zarr
/lustre/projects/1001/gdaga/home/fire.zarr
/lustre/projects/1001/gdaga/home/lpl_burned_area.zarr
/lustre/projects/1001/gdaga/home/phileo-bench_lc.zarr
/lustre/projects/1001/gdaga/home/phisatnet_clouds.zarr
/lustre/projects/1001/gdaga/home/phileo-bench_roads.zarr
/lustre/projects/1001/gdaga/home/worldfloods.zarr

# Pretrained Weights Directory
/lustre/projects/1001/gdaga/home/pretrained_weights/

# Output Directory
/lustre/projects/1001/gdaga/home/phi2FM_models/
```

---

## Data Transfer Commands

### Transfer Datasets to HPC

```bash
# Set up variables
LOCAL_DATA="/Data"
HPC_USER="your_username"
HPC_HOST="your_hpc_hostname"
HPC_BASE="/lustre/projects/1001/gdaga/home"

# Transfer datasets
scp -r ${LOCAL_DATA}/anomaly_detection/marine_area_dataset.zarr \
    ${HPC_USER}@${HPC_HOST}:${HPC_BASE}/anomaly_detection.zarr

scp -r ${LOCAL_DATA}/fire_dataset/fire_dataset.zarr \
    ${HPC_USER}@${HPC_HOST}:${HPC_BASE}/fire.zarr

scp -r ${LOCAL_DATA}/lpl_burned_area/burned.zarr \
    ${HPC_USER}@${HPC_HOST}:${HPC_BASE}/lpl_burned_area.zarr

scp -r ${LOCAL_DATA}/phisatnet/phileo-bench_lc.zarr \
    ${HPC_USER}@${HPC_HOST}:${HPC_BASE}/phileo-bench_lc.zarr

scp -r ${LOCAL_DATA}/phisatnet_clouds.zarr \
    ${HPC_USER}@${HPC_HOST}:${HPC_BASE}/phisatnet_clouds.zarr

scp -r ${LOCAL_DATA}/phisatnet_dataset/phileo-bench_roads.zarr \
    ${HPC_USER}@${HPC_HOST}:${HPC_BASE}/phileo-bench_roads.zarr

scp -r ${LOCAL_DATA}/worldfloods/worldfloods.zarr \
    ${HPC_USER}@${HPC_HOST}:${HPC_BASE}/worldfloods.zarr
```

### Transfer Pretrained Weights to HPC

```bash
# Create pretrained_weights directory on HPC
ssh ${HPC_USER}@${HPC_HOST} "mkdir -p ${HPC_BASE}/pretrained_weights"

# Transfer all pretrained weights
scp /home/gdaga/pretrained_weights/phisat2net_geoaware_best.pt \
    ${HPC_USER}@${HPC_HOST}:${HPC_BASE}/pretrained_weights/

scp /home/gdaga/hydranet/experiments/distillation_production_20251113_001051/baseline/config_baseline/best_downstream_model.pt \
    ${HPC_USER}@${HPC_HOST}:${HPC_BASE}/pretrained_weights/unet_myriad2_baseline.pt

scp /home/gdaga/pretrained_weights/satmae-vitlarge-800.pth \
    ${HPC_USER}@${HPC_HOST}:${HPC_BASE}/pretrained_weights/

scp /home/gdaga/pretrained_weights/Prithvi_100M.pt \
    ${HPC_USER}@${HPC_HOST}:${HPC_BASE}/pretrained_weights/

scp /home/gdaga/pretrained_weights/seco_resnet50_1m.ckpt \
    ${HPC_USER}@${HPC_HOST}:${HPC_BASE}/pretrained_weights/

scp /home/gdaga/pretrained_weights/moco_resnet50_200ep.pth.tar \
    ${HPC_USER}@${HPC_HOST}:${HPC_BASE}/pretrained_weights/

scp /home/gdaga/pretrained_weights/dino_resnet50_ep200.pth \
    ${HPC_USER}@${HPC_HOST}:${HPC_BASE}/pretrained_weights/

scp /home/gdaga/pretrained_weights/gassl_resnet50.pth \
    ${HPC_USER}@${HPC_HOST}:${HPC_BASE}/pretrained_weights/

scp /home/gdaga/pretrained_weights/caco_resnet50.pth \
    ${HPC_USER}@${HPC_HOST}:${HPC_BASE}/pretrained_weights/

scp /home/gdaga/pretrained_weights/geoaware_encoder_nano_mse.pt \
    ${HPC_USER}@${HPC_HOST}:${HPC_BASE}/pretrained_weights/

scp /home/gdaga/pretrained_weights/vit_large_gc_pretrained.pt \
    ${HPC_USER}@${HPC_HOST}:${HPC_BASE}/pretrained_weights/

scp /home/gdaga/pretrained_weights/phileo_precursor.ckpt \
    ${HPC_USER}@${HPC_HOST}:${HPC_BASE}/pretrained_weights/
```

---

## Directory Structure on HPC

```
/lustre/projects/1001/gdaga/home/
├── anomaly_detection.zarr/
├── fire.zarr/
├── lpl_burned_area.zarr/
├── phileo-bench_lc.zarr/
├── phisatnet_clouds.zarr/
├── phileo-bench_roads.zarr/
├── worldfloods.zarr/
├── pretrained_weights/
│   ├── phisat2net_geoaware_best.pt
│   ├── unet_myriad2_baseline.pt
│   ├── satmae-vitlarge-800.pth
│   ├── Prithvi_100M.pt
│   ├── seco_resnet50_1m.ckpt
│   ├── moco_resnet50_200ep.pth.tar
│   ├── dino_resnet50_ep200.pth
│   ├── gassl_resnet50.pth
│   ├── caco_resnet50.pth
│   ├── geoaware_encoder_nano_mse.pt
│   ├── vit_large_gc_pretrained.pt
│   └── phileo_precursor.ckpt
└── phi2FM_models/
    └── (output directory for trained models)
```

---

## Verification Commands

After transferring data, verify the structure on HPC:

```bash
# Check datasets
ssh ${HPC_USER}@${HPC_HOST} "ls -lh ${HPC_BASE}/*.zarr"

# Check pretrained weights
ssh ${HPC_USER}@${HPC_HOST} "ls -lh ${HPC_BASE}/pretrained_weights/"

# Check output directory
ssh ${HPC_USER}@${HPC_HOST} "ls -lh ${HPC_BASE}/phi2FM_models/"
```

---

## Dataset Sizes (Estimated)

| Dataset | Approximate Size | Description |
|---------|-----------------|-------------|
| anomaly_detection.zarr | ~XX GB | Marine area dataset |
| fire.zarr | ~XX GB | Fire detection |
| lpl_burned_area.zarr | ~XX GB | Burned area |
| phileo-bench_lc.zarr | ~XX GB | Land cover (11 classes) |
| phisatnet_clouds.zarr | ~XX GB | Cloud segmentation |
| phileo-bench_roads.zarr | ~XX GB | Road detection |
| worldfloods.zarr | ~XX GB | Flood detection |

**Note**: Update sizes after checking actual dataset sizes.

---

## Important Notes

1. **Zarr Format**: All datasets use Zarr format for efficient chunked storage
2. **Permissions**: Ensure proper read/write permissions on HPC directories
3. **Disk Quota**: Check your HPC disk quota before transferring large datasets
4. **Backup**: Keep local backups of all datasets and weights
5. **Path Consistency**: All configuration files in `args/lustre_expanded/` use these paths

---

## Configuration Usage

All expanded configuration files in `downstream/args/lustre_expanded/` reference these Lustre paths. No manual path editing is required when submitting jobs on the HPC cluster.

Example from a config file:
```yaml
data_path_128_10m: /lustre/projects/1001/gdaga/home/lpl_burned_area.zarr
data_path_224_10m: /lustre/projects/1001/gdaga/home/lpl_burned_area.zarr
pretrained_model_path: /lustre/projects/1001/gdaga/home/pretrained_weights/unet_myriad2_baseline.pt
output_path: /lustre/projects/1001/gdaga/home/phi2FM_models
```
