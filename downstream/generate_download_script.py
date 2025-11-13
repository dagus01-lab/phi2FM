#!/usr/bin/env python3
"""
Generate bash script to download samples using zarr sample IDs.
"""

import json
import os

def generate_download_script_from_mapping(
    mapping_file="clouds_sample_id_mapping.json",
    output_script="download_clouds_samples.sh",
    remote_user="your_username",
    remote_host="your_remote_host",
    remote_base_path="/Data/phisatnet_clouds.zarr/trainval"
):
    """Generate bash script from the sample ID mapping."""
    
    if not os.path.exists(mapping_file):
        print(f"Error: Mapping file not found: {mapping_file}")
        print("Please run 'python map_sample_ids.py' first.")
        return
    
    try:
        with open(mapping_file, 'r') as f:
            mapping_data = json.load(f)
        
        # Extract unique zarr sample IDs
        zarr_ids = []
        for file_info in mapping_data['mapping'].values():
            zarr_id = file_info.get('zarr_sample_id')
            if zarr_id is not None:
                zarr_ids.append(zarr_id)
        
        # Remove duplicates and sort
        zarr_ids = sorted(list(set(zarr_ids)))
        
        if not zarr_ids:
            print("No zarr sample IDs found in mapping file.")
            return
        
        # Generate bash script
        script_content = f'''#!/bin/bash
# Auto-generated download script for clouds samples
# Generated from {mapping_file}

# Configuration - UPDATE THESE VALUES
REMOTE_USER="{remote_user}"
REMOTE_HOST="{remote_host}"
REMOTE_BASE_PATH="{remote_base_path}"
LOCAL_DEST_DIR="./downloaded_clouds_samples"

# Zarr Sample IDs ({len(zarr_ids)} samples)
SAMPLE_IDS=(
{chr(10).join(f'    "{sample_id}"' for sample_id in zarr_ids)}
)

# Create local destination directory
mkdir -p "$LOCAL_DEST_DIR"

echo "Starting download of ${{#SAMPLE_IDS[@]}} samples..."
echo "Remote: ${{REMOTE_USER}}@${{REMOTE_HOST}}:${{REMOTE_BASE_PATH}}"
echo "Local destination: $LOCAL_DEST_DIR"
echo "----------------------------------------"

# Function to download a single sample
download_sample() {{
    local sample_id="$1"
    local remote_path="${{REMOTE_USER}}@${{REMOTE_HOST}}:${{REMOTE_BASE_PATH}}/${{sample_id}}"
    
    echo "Downloading sample ${{sample_id}}..."
    scp -r "$remote_path" "$LOCAL_DEST_DIR/"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully downloaded ${{sample_id}}"
    else
        echo "✗ Failed to download ${{sample_id}}"
    fi
}}

# Download all samples
for sample_id in "${{SAMPLE_IDS[@]}}"; do
    download_sample "$sample_id"
done

echo "----------------------------------------"
echo "Download complete!"
echo "Samples saved to: $LOCAL_DEST_DIR"
'''
        
        # Write script to file
        with open(output_script, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(output_script, 0o755)
        
        print(f"Download script generated: {output_script}")
        print(f"Number of samples: {len(zarr_ids)}")
        print(f"First few sample IDs: {zarr_ids[:5] if len(zarr_ids) >= 5 else zarr_ids}")
        print()
        print("To use the script:")
        print(f"1. Edit {output_script} to set your remote credentials")
        print(f"2. Run: ./{output_script}")
        
    except Exception as e:
        print(f"Error generating download script: {e}")

def main():
    print("Download Script Generator")
    print("=" * 30)
    
    # Check if mapping file exists
    mapping_file = "clouds_sample_id_mapping.json"
    if not os.path.exists(mapping_file):
        print(f"Mapping file {mapping_file} not found.")
        print("Please run 'python map_sample_ids.py' first to create the mapping.")
        return
    
    # Generate download script
    generate_download_script_from_mapping()

if __name__ == "__main__":
    main()