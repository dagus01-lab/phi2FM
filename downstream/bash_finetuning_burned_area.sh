#!/bin/bash

# Define the list of configuration files

configs=(
    "args/finetune_FMs/lpl_burned_area/geoaware.yml"
    "args/finetune_FMs/lpl_burned_area/moco.yml"
    "args/finetune_FMs/lpl_burned_area/phisatnet.yml"
    "args/finetune_FMs/lpl_burned_area/dino.yml"
    "args/finetune_FMs/lpl_burned_area/seco.yml"
    #"args/finetune_FMs/lpl_burned_area/uniphi.yml"
    "args/finetune_FMs/lpl_burned_area/gassl.yml"
    "args/finetune_FMs/lpl_burned_area/caco.yml"
    #"args/finetune_FMs/lpl_burned_area/vit.yml"
    "args/finetune_FMs/lpl_burned_area/prithvi.yml"
    "args/finetune_FMs/lpl_burned_area/satmae.yml"
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
