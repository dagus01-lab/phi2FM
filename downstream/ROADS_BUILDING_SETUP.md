# Roads and Building Configurations - Setup Complete

## Summary

Successfully created configuration files for **all models** in the `roads` and `building` tasks, matching the structure of other downstream tasks like worldfloods.

## What Was Created

### 1. Original Lustre Configs (`args/lustre/`)

**Roads** - 12 configurations:
- caco, seco, phisatnet, dino, moco, gassl, geoaware, prithvi, satmae, uniphi, vit, terramind

**Building** - 11 configurations:
- caco, seco, phisatnet, dino, moco, gassl, geoaware, prithvi, satmae, uniphi, vit

### 2. Expanded Configs (`args/lustre_expanded/`)

**Roads** - 60 configurations:
- 11 models × 5 n_shot values + 1 terramind × 5 n_shot values = 60

**Building** - 55 configurations:
- 11 models × 5 n_shot values = 55

### 3. PBS Scripts (`pbs_scripts_expanded/`)

**Roads** - 60 PBS scripts
**Building** - 55 PBS scripts

## Model Parameters Used

Each model configuration includes:

| Model | Model Name | Input Channels | Pad Bands | Pretrained Weight |
|-------|-----------|----------------|-----------|-------------------|
| caco | caco | 4 | 4 | resnet50_seco_geo_1m_200.pth |
| seco | seasonal_contrast | 10 | 10 | seco_resnet50_1m.ckpt |
| phisatnet | phisatnet | 8 | None | phisat2net_geoaware_best.pt |
| dino | dino | 13 | 13 | B13_rn50_dino_0099.pth |
| moco | moco | 13 | 13 | B13_rn50_moco_0099.pth |
| gassl | gassl | 13 | 3 | B13_rn50_gassl_0099.pth |
| geoaware | GeoAware_mh_pred_core_nano | 13 | 13 | GeoAware_mh_pred_core_nano_2000ep_s2_rn50.pth |
| prithvi | prithvi | 6 | 6 | Prithvi_100M.pt |
| satmae | SatMAE | 10 | 10 | pretrain-vit-large-e199.pth |
| uniphi | phileo_precursor | 12 | 12 | phileo_precursor_swinb_si_ckpt.pth |
| vit | vit_cnn_gc | 13 | 13 | ckpt_ViT.pth |

## Task-Specific Parameters

### Roads
- **Downstream Task**: roads
- **Output Channels**: 1 (regression)
- **Batch Size**: 32
- **Dataset**: `/lustre/projects/1001/gdaga/home/phileo-bench_roads.zarr`
- **N-Shot Values**: 50, 100, 500, 1000, 5000

### Building
- **Downstream Task**: building
- **Output Channels**: 1 (regression)
- **Batch Size**: 32
- **Dataset**: `/lustre/projects/1001/gdaga/home/phileo-bench_building.zarr`
- **N-Shot Values**: 50, 100, 500, 1000, 5000

## Total Configuration Summary

| Task | Original Configs | Expanded Configs | PBS Scripts |
|------|-----------------|------------------|-------------|
| worldfloods | 12 | 60 | 60 |
| phisatnet_clouds | 11 | 55 | 55 |
| **roads** | **12** | **60** | **60** |
| **building** | **11** | **55** | **55** |
| lpl_burned_area | 13 | 61 | 61 |
| fire | 11 | 55 | 55 |
| anomaly_detection | 11 | 55 | 55 |
| **Total** | **81** | **401** | **401** |

## Usage Examples

### Submit all roads experiments:
```bash
./submit_jobs.sh --task roads          # 60 jobs
```

### Submit all building experiments:
```bash
./submit_jobs.sh --task building       # 55 jobs
```

### Submit specific n_shot for roads:
```bash
# Submit only n_shot=50 for roads
for script in pbs_scripts_expanded/roads_*_nshot50_*.sh; do
    qsub "$script"
done
```

### Submit all caco experiments (including roads and building):
```bash
./submit_jobs.sh --model caco          # Now includes roads & building
```

### Check available scripts:
```bash
./submit_jobs.sh --count               # Shows 401 total scripts
./submit_jobs.sh --list                # Shows samples including roads & building
```

## Files Created

### Helper Script
- `create_roads_building_configs.py` - Script that generated the model configs

### Configuration Files
```
args/lustre/
├── roads/          # 12 model configs
│   ├── caco.yml
│   ├── seco.yml
│   ├── phisatnet.yml
│   ├── dino.yml
│   ├── moco.yml
│   ├── gassl.yml
│   ├── geoaware.yml
│   ├── prithvi.yml
│   ├── satmae.yml
│   ├── uniphi.yml
│   ├── vit.yml
│   └── terramind.yml
│
└── building/       # 11 model configs
    ├── caco.yml
    ├── seco.yml
    ├── phisatnet.yml
    ├── dino.yml
    ├── moco.yml
    ├── gassl.yml
    ├── geoaware.yml
    ├── prithvi.yml
    ├── satmae.yml
    ├── uniphi.yml
    └── vit.yml
```

### Expanded Configs
```
args/lustre_expanded/
├── roads/          # 60 expanded configs (11 models × 5 n_shot + terramind × 5)
└── building/       # 55 expanded configs (11 models × 5 n_shot)
```

### PBS Scripts
```
pbs_scripts_expanded/
├── roads_*.sh          # 60 PBS scripts
└── building_*.sh       # 55 PBS scripts
```

## Verification

Sample configuration check:
```bash
# View roads caco config
cat args/lustre/roads/caco.yml

# View building phisatnet config
cat args/lustre/building/phisatnet.yml

# View expanded config
cat args/lustre_expanded/roads/caco_nshot50_frozen.yml

# View PBS script
cat pbs_scripts_expanded/building_seco_nshot100_frozen.sh
```

## Notes

1. ✅ All model configurations created for roads and building
2. ✅ Model parameters (model_name, pretrained_model_path, pad_bands) match worldfloods
3. ✅ Task-specific parameters (dataset paths, output_channels) preserved from original roads/building configs
4. ✅ Expanded configs generated with single n_shot values
5. ✅ PBS scripts generated for all configurations
6. ✅ Submit script updated to include roads and building tasks
7. ✅ Total system now has 401 configurations and PBS scripts

---

**Created**: November 13, 2025  
**Total Original Configs**: 81 (up from 60)  
**Total Expanded Configs**: 401 (up from 296)  
**Total PBS Scripts**: 401 (up from 296)  
**New Tasks Completed**: roads (12 models), building (11 models)
