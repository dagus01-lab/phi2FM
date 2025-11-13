import os
import re
import json
from pathlib import Path

def extract_sample_ids_from_mapping(mapping_file="clouds_sample_id_mapping.json"):
    """
    Extract zarr sample IDs from the mapping file created by map_sample_ids.py
    
    Args:
        mapping_file (str): Path to the mapping JSON file
        
    Returns:
        list: List of zarr sample IDs (as strings)
    """
    if not os.path.exists(mapping_file):
        print(f"Error: Mapping file not found: {mapping_file}")
        print("Please run map_sample_ids.py first to create the mapping.")
        return []
    
    try:
        with open(mapping_file, 'r') as f:
            mapping_data = json.load(f)
        
        zarr_ids = []
        for file_info in mapping_data['mapping'].values():
            zarr_id = file_info.get('zarr_sample_id')
            if zarr_id is not None:
                zarr_ids.append(zarr_id)
        
        # Remove duplicates and sort
        zarr_ids = sorted(list(set(zarr_ids)))
        return zarr_ids
        
    except Exception as e:
        print(f"Error reading mapping file: {e}")
        return []

def extract_sample_ids_from_filenames(folder_path):
    """
    Extract sample IDs from image names in the format: clouds_sample_XXXXX.png
    This returns the file indices, not the zarr sample IDs.
    
    Args:
        folder_path (str): Path to the folder containing images
        
    Returns:
        list: List of file indices (as strings)
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Error: Folder {folder_path} does not exist")
        return []
    
    sample_ids = []
    
    # Pattern to match: clouds_sample_XXXXX.png
    pattern = r'clouds_sample_(\d{5})\.png'
    
    # Get all files in the directory
    for filename in os.listdir(folder_path):
        match = re.match(pattern, filename)
        if match:
            sample_id = match.group(1)  # Extract the numeric part
            sample_ids.append(sample_id)
    
    # Sort the sample IDs
    sample_ids.sort()
    
    return sample_ids

def main():
    # Path to the folder containing images
    img_folder = "downstream/extracted_samples/clouds/img/"
    
    print("Getting zarr sample IDs from mapping file...")
    
    # First try to get zarr sample IDs from mapping file
    zarr_sample_ids = extract_sample_ids_from_mapping()
    
    if zarr_sample_ids:
        print(f"Found {len(zarr_sample_ids)} zarr sample IDs from mapping")
        print("Zarr sample IDs:")
        for sample_id in zarr_sample_ids:
            print(f'"{sample_id}"')
        
        # Save zarr IDs to file
        output_file = "zarr_sample_ids.txt"
        with open(output_file, 'w') as f:
            for sample_id in zarr_sample_ids:
                f.write(f"{sample_id}\n")
        
        print(f"\nZarr sample IDs saved to {output_file}")
        
    else:
        print("Could not get zarr sample IDs from mapping. Falling back to file indices.")
        
        # Fallback: Extract file indices from filenames
        file_indices = extract_sample_ids_from_filenames(img_folder)
        
        print(f"Found {len(file_indices)} file indices")
        print("File indices (NOT zarr sample IDs):")
        for sample_id in file_indices:
            print(f'"{sample_id}"')
        
        # Save file indices to file
        output_file = "file_indices.txt"
        with open(output_file, 'w') as f:
            for sample_id in file_indices:
                f.write(f"{sample_id}\n")
        
        print(f"\nFile indices saved to {output_file}")
        print("\nWARNING: These are file indices, not zarr sample IDs!")
        print("Run 'python map_sample_ids.py' first to create the proper mapping.")

if __name__ == "__main__":
    main()