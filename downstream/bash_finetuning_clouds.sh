#!/bin/bash

# Define the list of configuration files

configs=(
    "args/finetune_FMs/phisatnet_clouds/geoaware.yml"
    "args/finetune_FMs/phisatnet_clouds/moco.yml"
    "args/finetune_FMs/phisatnet_clouds/phisatnet.yml"
    "args/finetune_FMs/phisatnet_clouds/dino.yml"
    "args/finetune_FMs/phisatnet_clouds/seco.yml"
    #"args/finetune_FMs/phisatnet_clouds/uniphi.yml"
    "args/finetune_FMs/phisatnet_clouds/gassl.yml"
    "args/finetune_FMs/phisatnet_clouds/caco.yml"
    #"args/finetune_FMs/phisatnet_clouds/vit.yml"
    "args/finetune_FMs/phisatnet_clouds/prithvi.yml"
    "args/finetune_FMs/phisatnet_clouds/satmae.yml"
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
