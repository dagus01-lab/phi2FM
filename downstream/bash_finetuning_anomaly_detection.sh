#!/bin/bash

# Define the list of configuration files

configs=(
    "args/finetune_FMs/anomaly_detection/geoaware.yml"
    "args/finetune_FMs/anomaly_detection/moco.yml"
    "args/finetune_FMs/anomaly_detection/phisatnet.yml"
    "args/finetune_FMs/anomaly_detection/dino.yml"
    "args/finetune_FMs/anomaly_detection/seco.yml"
    #"args/finetune_FMs/anomaly_detection/uniphi.yml"
    "args/finetune_FMs/anomaly_detection/gassl.yml"
    "args/finetune_FMs/anomaly_detection/caco.yml"
    #"args/finetune_FMs/anomaly_detection/vit.yml"
    "args/finetune_FMs/anomaly_detection/prithvi.yml"
    "args/finetune_FMs/anomaly_detection/satmae.yml"
)

# Loop through each config file and execute the training script sequentially
for config in "${configs[@]}"; do
    echo "Running training with config: $config"
    python training_script.py -r "$config"
    if [ $? -ne 0 ]; then
        echo "Error encountered in training with $config. Exiting."
        exit 1
    fi
    echo "Finished training with config: $config"
done

echo "All training scripts completed successfully."
