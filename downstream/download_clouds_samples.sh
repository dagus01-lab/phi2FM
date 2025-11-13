#!/bin/bash
# Auto-generated download script for clouds samples
# Generated from clouds_sample_id_mapping.json

# Configuration - UPDATE THESE VALUES
REMOTE_USER="your_username"
REMOTE_HOST="your_remote_host"
REMOTE_BASE_PATH="/Data/phisatnet_clouds.zarr/trainval"
LOCAL_DEST_DIR="./downloaded_clouds_samples"

# Zarr Sample IDs (52 samples)
SAMPLE_IDS=(
    "00528"
    "01143"
    "01178"
    "01183"
    "01519"
    "01605"
    "01760"
    "01767"
    "01799"
    "01983"
    "02132"
    "02447"
    "02503"
    "02587"
    "02913"
    "03097"
    "03167"
    "03168"
    "03263"
    "03434"
    "03570"
    "03863"
    "03880"
    "04124"
    "04205"
    "04223"
    "04228"
    "04540"
    "04957"
    "06119"
    "06161"
    "06223"
    "06350"
    "06369"
    "06688"
    "06749"
    "06841"
    "06954"
    "06980"
    "07178"
    "07293"
    "07330"
    "07403"
    "07733"
    "08066"
    "08252"
    "08336"
    "08926"
    "08933"
    "09068"
    "09174"
    "09692"
)

# Create local destination directory
mkdir -p "$LOCAL_DEST_DIR"

echo "Starting download of ${#SAMPLE_IDS[@]} samples..."
echo "Remote: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE_PATH}"
echo "Local destination: $LOCAL_DEST_DIR"
echo "----------------------------------------"

# Function to download a single sample
download_sample() {
    local sample_id="$1"
    local remote_path="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE_PATH}/${sample_id}"
    
    echo "Downloading sample ${sample_id}..."
    scp -r "$remote_path" "$LOCAL_DEST_DIR/"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully downloaded ${sample_id}"
    else
        echo "✗ Failed to download ${sample_id}"
    fi
}

# Download all samples
for sample_id in "${SAMPLE_IDS[@]}"; do
    download_sample "$sample_id"
done

echo "----------------------------------------"
echo "Download complete!"
echo "Samples saved to: $LOCAL_DEST_DIR"
