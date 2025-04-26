#!/bin/bash

# Define the list of configuration files
configs=(
    # "args/ccollado/geoaware.yml"
    # "args/ccollado/prithvi.yml"
    # "args/ccollado/satmae.yml"
    # "args/ccollado/moco.yml"
    # "args/ccollado/dino.yml"
    "args/ccollado/phisatnet.yml"
    # "args/ccollado/seco.yml"
    # "args/ccollado/uniphi.yml"
    # "args/ccollado/vit.yml"
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
