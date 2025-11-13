#!/usr/bin/env python3
"""
Script to map extracted sample indices to actual zarr sample IDs.
This will help establish the correspondence between the files in 
downstream/extracted_samples/clouds/img/ and the actual sample IDs in the zarr archive.
"""

import os
import sys
import re
import json
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
parent_dir = str(Path(__file__).resolve().parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.load_data import load_data
from utils.training_utils import read_yaml

def extract_index_from_filename(filename):
    """
    Extract the index from filename like 'clouds_sample_00042.png'
    Returns the numeric index as integer.
    """
    pattern = r'clouds_sample_(\d{5})\.png'
    match = re.match(pattern, filename)
    if match:
        return int(match.group(1))
    return None

def load_clouds_dataset():
    """
    Load the clouds dataset using the same configuration as the extraction script.
    """
    print("Loading clouds dataset...")
    
    # Configuration from extract_samples_simple.py
    config_file = "args/finetune_FMs/phisatnet_clouds/seco.yml"
    
    if not os.path.exists(config_file):
        print(f"Error: Config file not found: {config_file}")
        return None
    
    # Read configuration
    args = read_yaml(config_file)
    args.n_shot = 5000  # Use 5000 n-shot configuration
    batch_size = 1
    
    # Determine dataset path based on model
    model_name = args.model_name
    if hasattr(args, 'data_path_224_30m') and model_name in getattr(args, 'models_224_r30', []):
        dataset_folder = args.data_path_224_30m
    elif hasattr(args, 'data_path_224_10m'):
        dataset_folder = args.data_path_224_10m
    else:
        dataset_folder = args.data_path_128_10m
    
    # Set other parameters
    crop_images = model_name in ['phileo_precursor', 'phileo_precursor_classifier']
    
    try:
        # Load data
        weights, pos_weight, _, dl_test, _, _ = load_data(
            dataset_folder,
            with_augmentations=False,
            num_workers=4,
            batch_size=batch_size,
            downstream_task="clouds",
            model_name=model_name.split('_')[0],
            device='cpu',
            pad_bands=getattr(args, 'pad_bands', 10),
            crop_images=crop_images,
            num_classes=5,  # clouds has 2 output channels after aggregation
            n=5000,  # Use same as in extraction script
            weights_dir="clouds",
            patch_size=None
        )
        
        return dl_test
        
    except Exception as e:
        import traceback
        print(f"Error loading dataset: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return None

def debug_dataset_structure(dl_test):
    """
    Debug function to understand the dataset structure and sample ID mapping.
    """
    print(f"\n=== Debugging dataset structure ===")
    
    dataset = dl_test.dataset
    print(f"Dataset type: {type(dataset)}")
    print(f"Dataset size: {len(dataset)}")
    
    # Check for sample ID attributes
    sample_id_attrs = ['sample_ids', 'ids', 'keys', 'samples', 'file_list', 'sample_names']
    found_attrs = []
    
    for attr in sample_id_attrs:
        if hasattr(dataset, attr):
            attr_value = getattr(dataset, attr)
            print(f"Found attribute '{attr}': {type(attr_value)}")
            if hasattr(attr_value, '__len__'):
                print(f"  Length: {len(attr_value)}")
                if len(attr_value) > 0:
                    print(f"  First few values: {attr_value[:5] if len(attr_value) >= 5 else attr_value}")
                    found_attrs.append((attr, attr_value))
    
    # Check first sample structure
    if len(dataset) > 0:
        first_sample = dataset[0]
        print(f"First sample type: {type(first_sample)}")
        if isinstance(first_sample, dict):
            print(f"First sample keys: {list(first_sample.keys())}")
            
            # Look for sample ID in the sample itself
            id_keys = ['sample_id', 'id', 'idx', 'key', 'name', 'filename']
            for key in id_keys:
                if key in first_sample:
                    print(f"Found sample ID key '{key}': {first_sample[key]}")
    
    print("=" * 50)
    return found_attrs

def create_index_to_id_mapping(dl_test, num_samples=200):
    """
    Create mapping from dataset indices to actual sample IDs.
    """
    print(f"\nCreating index to ID mapping for {num_samples} samples...")
    
    dataset = dl_test.dataset
    dataset_size = len(dataset)
    
    # Use the same sampling strategy as extract_samples_simple.py
    if num_samples >= dataset_size:
        sample_indices = list(range(dataset_size))
    else:
        # Use evenly spaced samples
        step = max(1, dataset_size // num_samples)
        sample_indices = list(range(0, dataset_size, step))[:num_samples]
    
    index_to_id_map = {}
    
    print(f"Processing {len(sample_indices)} sample indices...")
    
    for i, sample_idx in enumerate(tqdm(sample_indices, desc="Mapping indices")):
        try:
            sample = dataset[sample_idx]
            
            # Try to get the actual sample ID from the dataset
            actual_sample_id = None
            
            # Method 1: Check dataset attributes
            if hasattr(dataset, 'sample_ids') and sample_idx < len(dataset.sample_ids):
                actual_sample_id = dataset.sample_ids[sample_idx]
            elif hasattr(dataset, 'ids') and sample_idx < len(dataset.ids):
                actual_sample_id = dataset.ids[sample_idx]
            elif hasattr(dataset, 'keys') and sample_idx < len(dataset.keys):
                actual_sample_id = dataset.keys[sample_idx]
            elif hasattr(dataset, 'sample_names') and sample_idx < len(dataset.sample_names):
                actual_sample_id = dataset.sample_names[sample_idx]
            
            # Method 2: Check sample dictionary
            elif isinstance(sample, dict):
                id_keys = ['sample_id', 'id', 'idx', 'key', 'name', 'filename']
                for key in id_keys:
                    if key in sample:
                        actual_sample_id = sample[key]
                        break
            
            # Convert to string if it's not already
            if actual_sample_id is not None:
                if isinstance(actual_sample_id, (int, float)):
                    actual_sample_id = f"{int(actual_sample_id):05d}"
                elif isinstance(actual_sample_id, bytes):
                    actual_sample_id = actual_sample_id.decode('utf-8')
                else:
                    actual_sample_id = str(actual_sample_id)
                
                # Remove any file extensions
                if '.' in actual_sample_id:
                    actual_sample_id = actual_sample_id.split('.')[0]
            
            index_to_id_map[sample_idx] = actual_sample_id
            
            # Debug: Print first few mappings
            if i < 10:
                print(f"  Index {sample_idx} -> Sample ID {actual_sample_id}")
                
        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")
            index_to_id_map[sample_idx] = None
            continue
    
    return index_to_id_map, sample_indices

def map_extracted_files_to_zarr_ids(extracted_dir, index_to_id_map):
    """
    Map the extracted files to their corresponding zarr sample IDs.
    """
    print(f"\nMapping extracted files to zarr sample IDs...")
    
    img_dir = Path(extracted_dir) / "img"
    
    if not img_dir.exists():
        print(f"Error: Directory not found: {img_dir}")
        return {}
    
    file_to_zarr_map = {}
    
    # Get all PNG files
    png_files = list(img_dir.glob("*.png"))
    print(f"Found {len(png_files)} PNG files")
    
    for png_file in png_files:
        # Extract index from filename
        file_index = extract_index_from_filename(png_file.name)
        
        if file_index is not None:
            # Look up the corresponding zarr sample ID
            zarr_sample_id = index_to_id_map.get(file_index)
            
            file_to_zarr_map[png_file.name] = {
                'file_index': file_index,
                'zarr_sample_id': zarr_sample_id,
                'file_path': str(png_file)
            }
        else:
            print(f"Warning: Could not extract index from filename: {png_file.name}")
    
    return file_to_zarr_map

def save_mapping_results(file_to_zarr_map, output_file="clouds_sample_id_mapping.json"):
    """
    Save the mapping results to a JSON file.
    """
    print(f"\nSaving mapping results to {output_file}...")
    
    # Create a summary
    summary = {
        'total_files': len(file_to_zarr_map),
        'files_with_zarr_ids': sum(1 for v in file_to_zarr_map.values() if v['zarr_sample_id'] is not None),
        'files_without_zarr_ids': sum(1 for v in file_to_zarr_map.values() if v['zarr_sample_id'] is None),
        'mapping': file_to_zarr_map
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Mapping saved to {output_file}")
    print(f"Summary:")
    print(f"  Total files: {summary['total_files']}")
    print(f"  Files with zarr IDs: {summary['files_with_zarr_ids']}")
    print(f"  Files without zarr IDs: {summary['files_without_zarr_ids']}")
    
    return summary

def create_zarr_id_list(file_to_zarr_map, output_file="clouds_zarr_sample_ids.txt"):
    """
    Create a simple text file with just the zarr sample IDs.
    """
    print(f"\nCreating zarr sample ID list...")
    
    zarr_ids = []
    for file_info in file_to_zarr_map.values():
        zarr_id = file_info['zarr_sample_id']
        if zarr_id is not None:
            zarr_ids.append(zarr_id)
    
    # Remove duplicates and sort
    zarr_ids = sorted(list(set(zarr_ids)))
    
    with open(output_file, 'w') as f:
        for zarr_id in zarr_ids:
            f.write(f'"{zarr_id}"\n')
    
    print(f"Zarr sample IDs saved to {output_file}")
    print(f"Total unique zarr IDs: {len(zarr_ids)}")
    
    if len(zarr_ids) > 0:
        print(f"First few zarr IDs: {zarr_ids[:5]}")
    
    return zarr_ids

def main():
    """
    Main function to create the mapping between extracted files and zarr sample IDs.
    """
    print("Sample ID Mapping Script")
    print("=" * 50)
    
    # Configuration
    extracted_dir = "extracted_samples/clouds"
    
    # Step 1: Load the dataset
    dl_test = load_clouds_dataset()
    if dl_test is None:
        print("Failed to load dataset. Exiting.")
        return
    
    # Step 2: Debug dataset structure
    found_attrs = debug_dataset_structure(dl_test)
    
    # Step 3: Create index to ID mapping
    index_to_id_map, sample_indices = create_index_to_id_mapping(dl_test, num_samples=200)
    
    # Step 4: Map extracted files to zarr IDs
    file_to_zarr_map = map_extracted_files_to_zarr_ids(extracted_dir, index_to_id_map)
    
    # Step 5: Save results
    summary = save_mapping_results(file_to_zarr_map)
    zarr_ids = create_zarr_id_list(file_to_zarr_map)
    
    print("\n" + "=" * 50)
    print("MAPPING COMPLETE")
    print("=" * 50)
    
    # Print some examples
    if file_to_zarr_map:
        print("\nExample mappings:")
        for i, (filename, info) in enumerate(list(file_to_zarr_map.items())[:5]):
            print(f"  {filename} -> zarr ID: {info['zarr_sample_id']}")
    
    print(f"\nResults saved to:")
    print(f"  - clouds_sample_id_mapping.json (detailed mapping)")
    print(f"  - clouds_zarr_sample_ids.txt (list of zarr IDs)")

if __name__ == "__main__":
    main()