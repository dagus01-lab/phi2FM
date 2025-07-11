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
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------------------------------------------
# FUNCTION TO CREATE PATCHES FROM TIFF FILE, AND SPLIT INTO TRAIN AND VAL SETS
import rasterio
import buteo as beo
from sklearn.model_selection import train_test_split
from pyproj import Transformer


def encode_latitude(lat):
    """ 
    Encode latitude in the range [-90, 90] using the WRAP approach.
    Normalizes to [-1, 1] and computes sine and cosine.
    """
    # Normalize latitude to [-1, 1]
    lat_normalized = lat / 90.0
    encoded_sin = np.sin(np.pi * lat_normalized)
    encoded_cos = np.cos(np.pi * lat_normalized)
    # Returns shape (2, N)
    return np.array([encoded_sin, encoded_cos], dtype=np.float32)

def encode_longitude(lng):
    """
    Encode longitude in the range [-180, 180] using the WRAP approach.
    Normalizes to [-1, 1] and computes sine and cosine.
    """
    # Normalize longitude to [-1, 1]
    lng_normalized = lng / 180.0
    encoded_sin = np.sin(np.pi * lng_normalized)
    encoded_cos = np.cos(np.pi * lng_normalized)
    # Returns shape (2, N)
    return np.array([encoded_sin, encoded_cos], dtype=np.float32)

def encode_coordinates(coords):
    """
    Expects coords to be a 2D numpy array of shape (N, 2) where each row is [lat, lon].
    Returns a numpy array of shape (N, 4) with columns:
    [sin(lat_encoded), cos(lat_encoded), sin(lon_encoded), cos(lon_encoded)].
    """
    lat_values = coords[:, 0]  # shape (N,)
    lon_values = coords[:, 1]  # shape (N,)
    
    encoded_lat = encode_latitude(lat_values)  # shape (2, N)
    encoded_lon = encode_longitude(lon_values)   # shape (2, N)
    
    # Concatenate along the first axis -> shape becomes (4, N)
    encoded_coords = np.concatenate((encoded_lat, encoded_lon), axis=0)
    # Transpose to shape (N, 4)
    encoded_coords = np.transpose(encoded_coords)
    return encoded_coords


def create_patches_from_tiffs(
    phisat2_tiff: str,
    patch_size: int = 128,
    crop_size: int = 1280,
    overlaps: int = 0,
    max_cloud_prob: float = 0.05,
):

    # -------------------------------------------------
    # 1. READ TIFF FILE
    # -------------------------------------------------
    with rasterio.open(phisat2_tiff, mmap=True) as phisat2_dataset:

        if crop_size is not None:

            if phisat2_dataset.width < crop_size or phisat2_dataset.height < crop_size:
                print(f"The TIFF file {phisat2_tiff} must be at least {crop_size}x{crop_size} pixels in size.")
                return None, None, None

            start_x = (phisat2_dataset.width - crop_size) // 2
            start_y = (phisat2_dataset.height - crop_size) // 2

            # Read all PHISAT2 bands
            window = rasterio.windows.Window(start_x, start_y, crop_size, crop_size)
            phisat2_array = phisat2_dataset.read(window=window)
        else:
            phisat2_array = phisat2_dataset.read()
        meta = phisat2_dataset.tags()
    
    # -------------------------------------------------
    # 2. EXTRACT CLOUD BAND
    # -------------------------------------------------
    cloud_array = phisat2_array[8]
    cloud_array = np.where(cloud_array == 10000, 0.25, cloud_array)
    cloud_array = np.where(cloud_array == 20000, 0.5, cloud_array)
    cloud_array = np.where(cloud_array == 30000, 1, cloud_array)
    cloud_array = np.expand_dims(cloud_array, axis=-1)

    # -------------------------------------------------
    # 3. EXTRACT CLIMATE ZONES BAND
    # -------------------------------------------------
    climate_array = phisat2_array[9]
    climate_array = np.expand_dims(climate_array, axis=-1)

    # -------------------------------------------------
    # 4. EXTRACT PHISAT2 BANDS
    # -------------------------------------------------
    phisat2_array = phisat2_array[:8]
    
    # Transpose the array to (height, width, bands)
    phisat2_array = np.transpose(phisat2_array, (1, 2, 0))

    # Ensure that the spatial dimensions match
    if phisat2_array.shape[:2] != cloud_array.shape[:2]:
        print("Spatial dimensions of PHISAT2 and CLOUD_PROB images do not match.")
        return None, None, None

    # -------------------------------------------------
    # 5. CREATE PATCHES
    # -------------------------------------------------

    # CLOUD -------------------------------------------
    patches_cloud = beo.array_to_patches(
        cloud_array, tile_size=patch_size, n_offsets=overlaps
    )
    indices_ok_cloud = np.where(
        np.mean(patches_cloud, axis=(1, 2, 3)) < max_cloud_prob
    )[0]

    # PHISAT2 -------------------------------------------
    patches_phisat2 = beo.array_to_patches(
        phisat2_array, tile_size=patch_size, n_offsets=overlaps
    )
    patches_phisat2 = patches_phisat2[indices_ok_cloud]
    if patches_phisat2.shape[0] == 0:
        print(f"No valid patches found for max_cloud_prob {max_cloud_prob}")
        return None, None, None

    mask = (patches_phisat2 == 65535)
    indices_not_nan = np.where(~mask.any(axis=(1, 2, 3)))[0]
    # indices_not_nan = np.where(~np.isnan(patches_phisat2).any(axis=(1, 2, 3)))[0]
    patches_phisat2 = patches_phisat2[indices_not_nan]

    if patches_phisat2.shape[0] == 0:
        print(f"No valid patches found after removing NaN values")
        return None, None, None
    
    # CLIMATE ZONES -------------------------------------------
    patches_climate = beo.array_to_patches(
        climate_array, tile_size=patch_size, n_offsets=overlaps
    )
    patches_climate = patches_climate[indices_ok_cloud]
    patches_climate = patches_climate[indices_not_nan]


    # -------------------------------------------------
    # 6. CREATE COORDINATES LABELS
    # -------------------------------------------------

    spatial_resolution = 4.75
    image_width = phisat2_array.shape[1]
    image_height = phisat2_array.shape[0]

    # Create transformers
    wgs84_to_utm = Transformer.from_crs('EPSG:4326', meta['crs'], always_xy=True)
    utm_to_wgs84 = Transformer.from_crs(meta['crs'], 'EPSG:4326', always_xy=True)

    # Coordinate Array from Center Coordinates
    # ----------------------------------------
    # Convert center coordinates to UTM
    x_center, y_center = wgs84_to_utm.transform(meta['centre_lon'], meta['centre_lat'])

    # Calculate bounding coordinates
    total_extent_m = spatial_resolution * image_width
    half_extent = total_extent_m / 2

    x_min_center = x_center - half_extent
    y_max_center = y_center + half_extent

    # Generate UTM coordinate arrays
    x_indices = np.arange(image_width)
    y_indices = np.arange(image_height)

    x_coords_center = x_min_center + (x_indices + 0.5) * spatial_resolution
    y_coords_center = y_max_center - (y_indices + 0.5) * spatial_resolution  # Inverted y-axis

    x_grid_center, y_grid_center = np.meshgrid(x_coords_center, y_coords_center)

    # Convert UTM to lat/lon
    lon_grid_center, lat_grid_center = utm_to_wgs84.transform(x_grid_center, y_grid_center)
    coods_array = np.stack((lat_grid_center, lon_grid_center), axis=-1)
    
    # -------------------------------------------------
    # 6. CREATE PATCHES FOR LABELS TOO
    # -------------------------------------------------
    patches_coords = beo.array_to_patches(
        coods_array, tile_size=patch_size, n_offsets=overlaps
    )
    patches_coords = patches_coords[indices_ok_cloud]
    patches_coords = patches_coords[indices_not_nan]
    
    patches_coords = np.mean(patches_coords, axis=(1, 2))
    patches_coords = encode_coordinates(patches_coords)
    
    # -------------------------------------------------
    # 7. SPLIT INTO TRAIN AND VAL SETS
    # -------------------------------------------------
    
    return patches_phisat2, patches_coords, patches_climate



def process_file_image(file_image, index, output_dir, patch_size, crop_size, phi2_tif_path):
    
    processed_np_files = os.listdir(output_dir)
    s2_files = [file for file in processed_np_files if file.endswith('_s2.npy')]
    processed_np_files = [file.replace('_train_s2.npy', '').replace('_val_s2.npy', '').replace('_test_s2.npy', '') for file in s2_files]

    if file_image in processed_np_files:
        return

    try:
        tiff_path_phi2 = f'{phi2_tif_path}/{file_image}.tif'
        
        partition = 'train'
        index_mod = index % 10
        
        if index_mod in [3, 6]:
            partition = 'val'
        if index_mod == 9:
            partition = 'test'

        patches_phisat2, patches_coords, patches_climate = create_patches_from_tiffs(
            phisat2_tiff=tiff_path_phi2,
            patch_size=patch_size,
            crop_size=crop_size,
            overlaps=0,
            max_cloud_prob=0.05,
        )

        if patches_phisat2 is not None:
            np.save(f'{output_dir}/{file_image}_{partition}_s2.npy', patches_phisat2.astype(np.uint16))
        if patches_coords is not None:
            np.save(f'{output_dir}/{file_image}_{partition}_label_coords.npy', patches_coords.astype(np.float64))
        if patches_climate is not None:
            np.save(f'{output_dir}/{file_image}_{partition}_label_climate.npy', patches_climate.astype(np.uint8))
        
        return
    
    except Exception as e:
        print(f'Error processing {file_image}: {e}')
        return


# ----------------------------------------------------------------
# ----------------------------- MAIN -----------------------------
# ----------------------------------------------------------------

if __name__ == "__main__":

    # Parameters
    patch_size = 256
    crop_size = 1536
    parallel_processing = True

    phi2_tif_path = '/home/ccollado/phileo_phisat2/MajorTOM/tiff_files'
    output_folder = f'MajorTOM/np_patches_{patch_size}_crop_{crop_size}'
    output_dir = f'/home/ccollado/phileo_phisat2/{output_folder}'
    df_simulated = pd.read_csv('/home/ccollado/1_simulate_data/Major-TOM/df_existing.csv') # Which images to process
    os.makedirs(output_dir, exist_ok=True)

    # Determine the number of workers
    max_workers = min(16, os.cpu_count() + 4)  # Adjust as needed

    # Prepare the list of tasks
    tasks = list(zip(df_simulated['unique_identifier'], range(len(df_simulated)), [output_dir]*len(df_simulated)))

    if parallel_processing:
        # Initialize the progress bar and execute in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_file_image, file_image, idx, out_dir, patch_size, crop_size, phi2_tif_path): idx for file_image, idx, out_dir in tasks}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Images"):
                idx = futures[future]
                try:
                    future.result()  # Retrieve the result to catch any exceptions
                except Exception as e:
                    print(f"Error processing image at index {idx}: {e}")
    else:
        # Sequential processing
        for file_image, idx, out_dir in tqdm(tasks, total=len(tasks), desc="Processing Images"):
            try:
                process_file_image(file_image, idx, out_dir, patch_size, crop_size, phi2_tif_path)  # Process each task sequentially
            except Exception as e:
                print(f"Error processing image at index {idx}: {e}")

print('Done')