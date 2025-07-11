#!/bin/bash

# Define the list of configuration files
configs=(
    # "args/phisat2/geoaware.yml"
    # "args/phisat2/prithvi.yml"
    # "args/phisat2/satmae.yml"
    # "args/phisat2/moco.yml"
    "args/phisat2/phisatnet.yml"
    # "args/phisat2/dino.yml"
    # "args/phisat2/seco.yml"
    # "args/phisat2/uniphi.yml"
    # "args/phisat2/gassl.yml"
    # "args/phisat2/caco.yml"
    # "args/phisat2/vit.yml"
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
