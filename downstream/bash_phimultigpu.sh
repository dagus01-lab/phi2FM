#!/bin/bash

# Define the list of configuration files
configs=(
    "args/phimultigpu/gassl.yml"
    "args/phimultigpu/caco.yml"
    # "args/phimultigpu/geoaware.yml"
    # "args/phimultigpu/moco.yml"
    # "args/phimultigpu/dino.yml"
    # "args/phimultigpu/seco.yml"
    # "args/phimultigpu/prithvi.yml"
    # "args/phimultigpu/satmae.yml"
    # "args/phimultigpu/uniphi.yml"
    # "args/phimultigpu/vit.yml"
)

# Loop through each config file and execute the training script using DDP
for config in "${configs[@]}"; do
    echo "Running training with config: $config"

    # Run training with torchrun for DDP
    python training_script.py -r "$config"

    if [ $? -ne 0 ]; then
        echo "Error encountered in training with $config. Exiting."
        exit 1
    fi

    echo "Finished training with config: $config"
done

echo "All training scripts completed successfully."
