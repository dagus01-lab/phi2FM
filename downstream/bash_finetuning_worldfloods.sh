#!/bin/bash

# Define the list of configuration files

configs=(
    #"args/finetune_FMs/worldfloods/geoaware.yml"
    #"args/finetune_FMs/worldfloods/moco.yml"
    #"args/finetune_FMs/worldfloods/phisatnet.yml"
    "args/finetune_FMs/worldfloods/dino.yml"
    "args/finetune_FMs/worldfloods/seco.yml"
    #"args/finetune_FMs/worldfloods/uniphi.yml"
    "args/finetune_FMs/worldfloods/gassl.yml"
    "args/finetune_FMs/worldfloods/caco.yml"
    #"args/finetune_FMs/worldfloods/vit.yml"
    "args/finetune_FMs/worldfloods/prithvi.yml"
    "args/finetune_FMs/worldfloods/satmae.yml"
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
