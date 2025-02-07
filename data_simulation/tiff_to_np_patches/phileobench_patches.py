import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import buteo as beo
import json
from osgeo import gdal
from glob import glob
import os
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
import time
import time
import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
import json
import rasterio
from concurrent.futures import ProcessPoolExecutor, as_completed

def phisat2_array_crop(phisat2_tiff, phisat2_dataset, crop_size, read_first=False):
    if crop_size is not None:
        if phisat2_dataset.width < crop_size or phisat2_dataset.height < crop_size:
            # print(f"The TIFF file {phisat2_tiff} must be at least {crop_size}x{crop_size} pixels in size.")
            if read_first:
                phisat2_array = phisat2_dataset.read(1)
            else:
                phisat2_array = phisat2_dataset.read()
        else:
            start_x = (phisat2_dataset.width - crop_size) // 2
            start_y = (phisat2_dataset.height - crop_size) // 2

            # Read all PHISAT2 bands
            window = rasterio.windows.Window(start_x, start_y, crop_size, crop_size)
            if read_first:
                phisat2_array = phisat2_dataset.read(1, window=window)
            else:
                phisat2_array = phisat2_dataset.read(window=window)
    else:
        # Read all PHISAT2 bands
        if read_first:
            phisat2_array = phisat2_dataset.read(1)
        else:
            phisat2_array = phisat2_dataset.read()
    
    return phisat2_array

# FUNCTION TO CREATE PATCHES FROM TIFF FILE, AND SPLIT INTO TRAIN AND VAL SETS
def create_patches_from_tiffs(
    phisat2_tiff: str,
    cloud_tiff: str,
    world_cover_tiff: str = None,
    buildings_tiff: str = None,
    roads_tiff: str = None,
    patch_size: int = 128,
    crop_size: int = 1280,
    overlaps: int = 0,
    max_cloud_prob: float = 0.05,
    downstream_task: str = "lc",  # lc, building, roads, all
    val_ratio: float = 0.2,
    random_seed: int = 42,
):

    # Read PHISAT2 bands
    with rasterio.open(phisat2_tiff, mmap=True) as phisat2_dataset:
        if phisat2_dataset.width < patch_size or phisat2_dataset.height < patch_size:
            print(f"The TIFF file {phisat2_tiff} must be at least {patch_size}x{patch_size} pixels in size.")
            return None, None, None, None

        # Read all PHISAT2 bands
        phisat2_array = phisat2_array_crop(phisat2_tiff, phisat2_dataset, crop_size, read_first=False)
        # Transpose the array to (height, width, bands)
        phisat2_array = np.transpose(phisat2_array, (1, 2, 0))

    # Read CLOUD_PROB band
    with rasterio.open(cloud_tiff, mmap=True) as cloud_dataset:
        # cloud_array = cloud_dataset.read(1)
        cloud_array = phisat2_array_crop(cloud_tiff, cloud_dataset, crop_size, read_first=True)
        cloud_array = np.expand_dims(cloud_array, axis=2)
        cloud_array = cloud_array / 255.0

    # Ensure that the spatial dimensions match
    if phisat2_array.shape[:2] != cloud_array.shape[:2]:
        print("Spatial dimensions of PHISAT2 and CLOUD_PROB images do not match.")
        return None, None, None, None

    # Process CLOUD_PROB band
    patches_cloud = beo.array_to_patches(
        cloud_array, tile_size=patch_size, n_offsets=overlaps
    )
    indices_ok_cloud = np.where(
        np.mean(patches_cloud, axis=(1, 2, 3)) < max_cloud_prob
    )[0]

    # Create patches for PHISAT2 data
    patches_phisat2 = beo.array_to_patches(
        phisat2_array, tile_size=patch_size, n_offsets=overlaps
    )
    patches_phisat2 = patches_phisat2[indices_ok_cloud]
    if patches_phisat2.shape[0] == 0:
        print(f"No valid patches found for max_cloud_prob {max_cloud_prob}")
        return None, None, None, None

    mask = (patches_phisat2 == 65535)
    indices_not_nan = np.where(~mask.any(axis=(1, 2, 3)))[0]
    # indices_not_nan = np.where(~np.isnan(patches_phisat2).any(axis=(1, 2, 3)))[0]
    patches_phisat2 = patches_phisat2[indices_not_nan]

    if patches_phisat2.shape[0] == 0:
        print(f"No valid patches found after removing NaN values")
        return None, None, None, None

    # Initialize labels dictionary
    patches_labels = {}

    # Handle labels for the downstream task
    if downstream_task in ['all', 'lc'] and world_cover_tiff and os.path.exists(world_cover_tiff):
        with rasterio.open(world_cover_tiff, mmap=True) as wc_dataset:
            # world_cover_array = wc_dataset.read(1)
            world_cover_array = phisat2_array_crop(world_cover_tiff, wc_dataset, crop_size, read_first=True)
            world_cover_array = np.expand_dims(world_cover_array, axis=2)
            patches_world_cover = beo.array_to_patches(
                world_cover_array, tile_size=patch_size, n_offsets=overlaps
            )
            patches_world_cover = patches_world_cover[indices_ok_cloud]
            patches_world_cover = patches_world_cover[indices_not_nan]
            patches_labels["lc"] = patches_world_cover

    if downstream_task in ['all', 'building'] and buildings_tiff and os.path.exists(buildings_tiff):
        with rasterio.open(buildings_tiff, mmap=True) as buildings_dataset:
            # buildings_array = buildings_dataset.read(1)
            buildings_array = phisat2_array_crop(buildings_tiff, buildings_dataset, crop_size, read_first=True)
            buildings_array = np.expand_dims(buildings_array, axis=2)
            patches_buildings = beo.array_to_patches(
                buildings_array, tile_size=patch_size, n_offsets=overlaps
            )
            patches_buildings = patches_buildings[indices_ok_cloud]
            patches_buildings = patches_buildings[indices_not_nan]
            patches_labels["building"] = patches_buildings

    if downstream_task in ['all', 'roads'] and roads_tiff and os.path.exists(roads_tiff):
        with rasterio.open(roads_tiff, mmap=True) as roads_dataset:
            # roads_array = roads_dataset.read(1)
            roads_array = phisat2_array_crop(roads_tiff, roads_dataset, crop_size, read_first=True)
            roads_array = np.expand_dims(roads_array, axis=2)
            patches_roads = beo.array_to_patches(
                roads_array, tile_size=patch_size, n_offsets=overlaps
            )
            patches_roads = patches_roads[indices_ok_cloud]
            patches_roads = patches_roads[indices_not_nan]
            patches_labels["roads"] = patches_roads

    # Check if labels are available
    if downstream_task != 'all':
        if downstream_task not in patches_labels:
            print(f"No labels found for the downstream task: {downstream_task}")
            patches_label = None
        else:
            patches_label = patches_labels[downstream_task]

    # Split data into training and validation sets
    indices = np.arange(len(patches_phisat2))
    if len(indices) == 1:
        train_indices = indices
        val_indices = []
    else:
        train_indices, val_indices = train_test_split(
            indices, test_size=val_ratio, random_state=random_seed
        )

    x_train = patches_phisat2[train_indices]
    x_val = patches_phisat2[val_indices]

    if downstream_task == 'all':
        y_train = {task: patches_labels[task][train_indices] for task in patches_labels}
        y_val = {task: patches_labels[task][val_indices] for task in patches_labels}
    else:
        y_train = patches_label[train_indices] if patches_label is not None else None
        y_val = patches_label[val_indices] if patches_label is not None else None

    return x_train, y_train, x_val, y_val

# ------------------------------------------------------------
# Transform Labels to Classification
# ------------------------------------------------------------
# def to_lc_class(y, class_labels = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])):
#     y_classification = np.array([np.isin(class_labels, y[i, ...]) for i in range(y.shape[0])])
#     return y_classification

def to_lc_class(y, class_labels=np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]), threshold=0.05):
    y_classification = np.array([
        [
            (np.sum(y[i, ...] == cls) / y[i, ...].size) >= threshold
            for cls in class_labels
        ]
        for i in range(y.shape[0])
    ])
    return y_classification

def to_building_class(y):
    # Calculate the mean across the spatial dimensions (height and width) for each image
    mean_values = np.mean(y > 0, axis=(1, 2, 3))
    
    # Validate that all mean values are between 0 and 1
    if np.any(mean_values < 0) or np.any(mean_values > 1):
        raise ValueError('Invalid values in building mask')
    
    # Initialize the classification array with boolean type
    y_classification = np.zeros((y.shape[0], 5), dtype=np.bool_)
    
    # Define thresholds for classification
    thresholds = [0, 0.3, 0.6, 0.9, 1.0]
    
    # Assign classes based on mean value thresholds
    for i in range(len(thresholds)-1):
        lower = thresholds[i]
        upper = thresholds[i+1]
        mask = (mean_values > lower) & (mean_values <= upper)
        y_classification[:, i] = mask
    
    # Handle the special case where mean value is exactly 0
    y_classification[:, 0] = (mean_values == 0)
    
    return y_classification

def to_roads_regres(y):
    y_regres = np.mean(y>0, axis=(1, 2, 3))
    return y_regres



# ------------------------------------------------------------
# Helper Function
# ------------------------------------------------------------
def is_already_processed(file_image, output_folder, mode='train'):
    """
    Checks if a given file_image is already processed.
    `mode` can be 'train' or 'test'.
    """
    processed_np_files = os.listdir(output_folder)
    if mode == 'train':
        processed_s2_files = [file for file in processed_np_files if file.endswith('_train_s2.npy') or file.endswith('_val_s2.npy')]
        processed_base_names = [file.replace('_train_s2.npy', '').replace('_val_s2.npy', '') for file in processed_s2_files]
    elif mode == 'test':
        processed_s2_files = [file for file in processed_np_files if file.endswith('_test_s2.npy')]
        processed_base_names = [file.replace('_test_s2.npy', '') for file in processed_s2_files]
    else:
        return False
    return file_image in processed_base_names

# ------------------------------------------------------------
# Processing Functions
# ------------------------------------------------------------
def process_train_file(file_image, base_path, tiff_folder, output_folder, max_cloud_prob, image_size, crop_size, val_ratio, random_seed):
    try:
        # Paths to input TIFF and label TIFFs
        tiff_path_phi2     = os.path.join(base_path, tiff_folder, f"{file_image}.tif")
        tiff_path_cloud    = os.path.join(base_path, "labels_tif", f"{file_image}_cloud.tif")
        tiff_path_lc       = os.path.join(base_path, "labels_tif", f"{file_image}_lc.tif")
        tiff_path_building = os.path.join(base_path, "labels_tif", f"{file_image}_building.tif")
        tiff_path_roads    = os.path.join(base_path, "labels_tif", f"{file_image}_roads.tif")

        # Create patches
        x_train, y_train, x_val, y_val = create_patches_from_tiffs(
            phisat2_tiff=tiff_path_phi2,
            cloud_tiff=tiff_path_cloud,
            world_cover_tiff=tiff_path_lc,
            buildings_tiff=tiff_path_building,
            roads_tiff=tiff_path_roads,
            patch_size=image_size,
            crop_size=crop_size,
            overlaps=0,
            max_cloud_prob=max_cloud_prob,
            downstream_task='all',
            val_ratio=val_ratio,
            random_seed=random_seed,
        )

        # Save train patches
        if x_train is not None:
            np.save(os.path.join(output_folder, f"{file_image}_train_s2.npy"), x_train)
        if x_val is not None:
            np.save(os.path.join(output_folder, f"{file_image}_val_s2.npy"), x_val)

        # Save label patches
        if y_train is not None:
            if 'lc' in y_train and y_train['lc'] is not None:
                np.save(os.path.join(output_folder, f"{file_image}_train_label_lc.npy"), y_train['lc'])
                y_classification_lc = to_lc_class(y_train['lc'])
                np.save(os.path.join(output_folder, f"{file_image}_train_label_lc_classification.npy"), y_classification_lc)
            if 'building' in y_train and y_train['building'] is not None:
                np.save(os.path.join(output_folder, f"{file_image}_train_label_building.npy"), y_train['building'])
                y_classification_building = to_building_class(y_train['building'])
                np.save(os.path.join(output_folder, f"{file_image}_train_label_building_classification.npy"), y_classification_building)
            if 'roads' in y_train and y_train['roads'] is not None:
                np.save(os.path.join(output_folder, f"{file_image}_train_label_roads.npy"), y_train['roads'])
                y_regres_roads = to_roads_regres(y_train['roads'])
                np.save(os.path.join(output_folder, f"{file_image}_train_label_roads_regression.npy"), y_regres_roads)

        if y_val is not None:
            if 'lc' in y_val and y_val['lc'] is not None:
                np.save(os.path.join(output_folder, f"{file_image}_val_label_lc.npy"), y_val['lc'])
                y_classification_lc = to_lc_class(y_val['lc'])
                np.save(os.path.join(output_folder, f"{file_image}_val_label_lc_classification.npy"), y_classification_lc)
            if 'building' in y_val and y_val['building'] is not None:
                np.save(os.path.join(output_folder, f"{file_image}_val_label_building.npy"), y_val['building'])
                y_classification_building = to_building_class(y_val['building'])
                np.save(os.path.join(output_folder, f"{file_image}_val_label_building_classification.npy"), y_classification_building)
            if 'roads' in y_val and y_val['roads'] is not None:
                np.save(os.path.join(output_folder, f"{file_image}_val_label_roads.npy"), y_val['roads'])
                y_regres_roads = to_roads_regres(y_val['roads'])
                np.save(os.path.join(output_folder, f"{file_image}_val_label_roads_regression.npy"), y_regres_roads)

    except Exception as e:
        print(f"Error processing train file {file_image}: {e}")

def process_test_file(file_image, base_path, tiff_folder, output_folder, max_cloud_prob, image_size, crop_size, val_ratio, random_seed):
    try:
        # Paths to input TIFF and label TIFFs
        tiff_path_phi2     = os.path.join(base_path, tiff_folder, f"{file_image}.tif")
        tiff_path_cloud    = os.path.join(base_path, "labels_tif", f"{file_image}_cloud.tif")
        tiff_path_lc       = os.path.join(base_path, "labels_tif", f"{file_image}_lc.tif")
        tiff_path_building = os.path.join(base_path, "labels_tif", f"{file_image}_building.tif")
        tiff_path_roads    = os.path.join(base_path, "labels_tif", f"{file_image}_roads.tif")

        # Create patches (for test set, we treat it similar to training but only save test data)
        x_test, y_test, _, _ = create_patches_from_tiffs(
            phisat2_tiff=tiff_path_phi2,
            cloud_tiff=tiff_path_cloud,
            world_cover_tiff=tiff_path_lc,
            buildings_tiff=tiff_path_building,
            roads_tiff=tiff_path_roads,
            patch_size=image_size,
            crop_size=crop_size,
            overlaps=0,
            max_cloud_prob=max_cloud_prob,
            downstream_task='all',
            val_ratio=val_ratio,  # val_ratio is not used for test
            random_seed=random_seed,
        )

        # Save test patches
        if x_test is not None:
            np.save(os.path.join(output_folder, f"{file_image}_test_s2.npy"), x_test)

        # Save label patches for test set
        if y_test is not None:
            if 'lc' in y_test and y_test['lc'] is not None:
                np.save(os.path.join(output_folder, f"{file_image}_test_label_lc.npy"), y_test['lc'])
                y_classification_lc = to_lc_class(y_test['lc'])
                np.save(os.path.join(output_folder, f"{file_image}_test_label_lc_classification.npy"), y_classification_lc)
            if 'building' in y_test and y_test['building'] is not None:
                np.save(os.path.join(output_folder, f"{file_image}_test_label_building.npy"), y_test['building'])
                y_classification_building = to_building_class(y_test['building'])
                np.save(os.path.join(output_folder, f"{file_image}_test_label_building_classification.npy"), y_classification_building)
            if 'roads' in y_test and y_test['roads'] is not None:
                np.save(os.path.join(output_folder, f"{file_image}_test_label_roads.npy"), y_test['roads'])
                y_regres_roads = to_roads_regres(y_test['roads'])
                np.save(os.path.join(output_folder, f"{file_image}_test_label_roads_regression.npy"), y_regres_roads)
    except Exception as e:
        print(f"Error processing test file {file_image}: {e}")

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == '__main__':
    # ------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------
    product_type = 'L1C'
    image_size = 224
    crop_size = None
    max_cloud_probability = 0.03
    val_ratio = 0.2
    random_seed = 42
    num_workers = os.cpu_count() - 1 or 1  # Leave one core free

    # ------------------------------------------------------------
    # Paths Setup
    # ------------------------------------------------------------
    base_path = '/home/ccollado/phileo_phisat2'
    tiff_folder = f'{product_type}/tiff_files'
    output_folder = f'{product_type}/np_patches_{image_size}'
    full_tiff_path = os.path.join(base_path, tiff_folder)
    output_full_path = os.path.join(base_path, output_folder)
    os.makedirs(output_full_path, exist_ok=True)

    # Path to train/test location JSON
    train_test_files_path = '/home/ccollado/phileo_NFS/phileo_data/aux_data/train_test_location_all.json'

    # ------------------------------------------------------------
    # Load Train / Test File Lists
    # ------------------------------------------------------------
    with open(train_test_files_path, 'r') as file:
        train_test_files = json.load(file)

    # All available TIFF files (without extension)
    all_image_files = [f.split('.')[0] for f in os.listdir(full_tiff_path)]

    train_files = train_test_files['train_locations']
    test_files  = train_test_files['test_locations']

    # Filter only existing TIFF files that match train/test sets
    real_train_files = [f for f in all_image_files if f in train_files]
    real_test_files  = [f for f in all_image_files if f in test_files]

    print(f'Total processed files (train + test): {len(real_train_files) + len(real_test_files)}')

    # ------------------------------------------------------------
    # Parallel Processing Setup
    # ------------------------------------------------------------
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # ---------------------
        # Process Training Files
        # ---------------------
        train_futures = []
        for i, file_image in enumerate(real_train_files):
            if is_already_processed(file_image, output_full_path, mode='train'):
                print(f"Already processed (train/val): {file_image}")
                continue
            train_futures.append(
                executor.submit(
                    process_train_file,
                    file_image,
                    base_path,
                    tiff_folder,
                    output_full_path,
                    max_cloud_probability,
                    image_size,
                    crop_size,
                    val_ratio,
                    random_seed
                )
            )

        # Progress bar for training
        for _ in tqdm(as_completed(train_futures), total=len(train_futures), desc="Processing Train Files"):
            pass  # Results are handled inside the processing function

        # ---------------------
        # Process Testing Files
        # ---------------------
        test_futures = []
        for i, file_image in enumerate(real_test_files):
            if is_already_processed(file_image, output_full_path, mode='test'):
                print(f"Already processed (test): {file_image}")
                continue
            test_futures.append(
                executor.submit(
                    process_test_file,
                    file_image,
                    base_path,
                    tiff_folder,
                    output_full_path,
                    max_cloud_probability,
                    image_size,
                    crop_size,
                    val_ratio,
                    random_seed
                )
            )

        # Progress bar for testing
        for _ in tqdm(as_completed(test_futures), total=len(test_futures), desc="Processing Test Files"):
            pass  # Results are handled inside the processing function

    print("Processing complete.")
