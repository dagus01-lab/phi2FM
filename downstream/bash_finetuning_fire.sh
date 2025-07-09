#!/bin/bash

# Define the list of configuration files
configs=(
    "args/finetune_FMs/fire/geoaware.yml"
    "args/finetune_FMs/fire/moco.yml"
    "args/finetune_FMs/fire/phisatnet.yml"
    "args/finetune_FMs/fire/dino.yml"
    "args/finetune_FMs/fire/seco.yml"
    #"args/finetune_FMs/fire/uniphi.yml"
    "args/finetune_FMs/fire/gassl.yml" 
    "args/finetune_FMs/fire/caco.yml" 
    ##"args/finetune_FMs/vit.yml" -> fix
    "args/finetune_FMs/fire/satmae.yml" 
    "args/finetune_FMs/fire/prithvi.yml"
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
