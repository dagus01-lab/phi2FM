from osgeo import gdal
gdal.DontUseExceptions()
import os

import rasterio
from rasterio.windows import Window
import numpy as np

from pyproj import Transformer
from scipy.ndimage import zoom
from tqdm import tqdm

def calculate_coordinates(metadata):
    # Parse numerical values from metadata
    x_min, y_min, x_max, y_max = map(float, metadata['bbox'].split(','))
    centre_lat = float(metadata['centre_lat'])
    centre_lon = float(metadata['centre_lon'])
    topleft_lat = float(metadata['topleft_max_lat'])
    topleft_lon = float(metadata['topleft_min_lon'])
    size_x = int(metadata['size_x'])
    size_y = int(metadata['size_y'])
    
    # Create transformers
    transform_to_wgs84 = Transformer.from_crs(metadata['crs'], "EPSG:4326", always_xy=True)
    transform_from_wgs84 = Transformer.from_crs("EPSG:4326", metadata['crs'], always_xy=True)

    # Method 1: Directly from bbox using the given CRS
    top_left_lon_1, top_left_lat_1 = transform_to_wgs84.transform(x_min, y_max)
    bottom_right_lon_1, bottom_right_lat_1 = transform_to_wgs84.transform(x_max, y_min)

    # Method 2: Using top-left and center coordinates
    lon_diff = topleft_lon - centre_lon
    lat_diff = topleft_lat - centre_lat
    bottom_right_lon_2 = centre_lon - lon_diff
    bottom_right_lat_2 = centre_lat - lat_diff
    top_left_lon_2 = topleft_lon
    top_left_lat_2 = topleft_lat

    # Method 3: Using top-left, size and resolution
    resolution = 10  # Assume resolution from metadata if variable
    total_width_m = size_x * resolution
    total_height_m = size_y * resolution
    top_left_x_utm, top_left_y_utm = transform_from_wgs84.transform(topleft_lon, topleft_lat)
    bottom_right_x_utm = top_left_x_utm + total_width_m
    bottom_right_y_utm = top_left_y_utm - total_height_m
    bottom_right_lon_3, bottom_right_lat_3 = transform_to_wgs84.transform(bottom_right_x_utm, bottom_right_y_utm)

    # Calculate mean coordinates
    mean_top_left_lon = (top_left_lon_1 + top_left_lon_2 + topleft_lon) / 3
    mean_top_left_lat = (top_left_lat_1 + top_left_lat_2 + topleft_lat) / 3
    mean_bottom_right_lon = (bottom_right_lon_1 + bottom_right_lon_2 + bottom_right_lon_3) / 3
    mean_bottom_right_lat = (bottom_right_lat_1 + bottom_right_lat_2 + bottom_right_lat_3) / 3

    # Check for significant differences
    threshold = 1e-1
    lons = [top_left_lon_1, top_left_lon_2, topleft_lon]
    lats = [top_left_lat_1, top_left_lat_2, topleft_lat]
    br_lons = [bottom_right_lon_1, bottom_right_lon_2, bottom_right_lon_3]
    br_lats = [bottom_right_lat_1, bottom_right_lat_2, bottom_right_lat_3]

    if max(lons) - min(lons) > threshold or max(lats) - min(lats) > threshold or \
       max(br_lons) - min(br_lons) > threshold or max(br_lats) - min(br_lats) > threshold:
        raise ValueError(f"Differences between methods exceed the acceptable threshold - differences: {max(lons) - min(lons), max(lats) - min(lats), max(br_lons) - min(br_lons), max(br_lats) - min(br_lats)}")

    return (mean_top_left_lon, mean_top_left_lat), (mean_bottom_right_lon, mean_bottom_right_lat)



def extract_subarray(tiff_path, top_left, bot_right, output_npy=None):
    """
    Extract a subarray from a GeoTIFF given geographic bounding coordinates.
    
    Parameters
    ----------
    tiff_path : str
        Path to the input GeoTIFF file.
    top_left_lon : float
        Longitude of the top-left corner.
    top_left_lat : float
        Latitude of the top-left corner.
    bottom_right_lon : float
        Longitude of the bottom-right corner.
    bottom_right_lat : float
        Latitude of the bottom-right corner.
    output_npy : str, optional
        If provided, saves the extracted array as a NumPy file.
        
    Returns
    -------
    subarray : numpy.ndarray
        Extracted portion of the raster data as a NumPy array.
    """
    top_left_lon, top_left_lat = top_left
    bottom_right_lon, bottom_right_lat = bot_right
    # Open the raster file
    with rasterio.open(tiff_path) as src:
        # Confirm the CRS and that coordinates match
        # (Optional: If you're unsure, you might reproject or check CRS.)
        
        # Convert geospatial coordinates to pixel indices.
        # rasterio's index() method converts (lon, lat) to (row, col)
        row_min, col_min = src.index(top_left_lon, top_left_lat)
        row_max, col_max = src.index(bottom_right_lon, bottom_right_lat)
        
        # Ensure row_min/row_max and col_min/col_max define the upper-left and lower-right corners correctly.
        # Note that latitudes decrease as we go down, so we might need to reorder:
        if row_min > row_max:
            row_min, row_max = row_max, row_min
        if col_min > col_max:
            col_min, col_max = col_max, col_min

        # Define the window of interest.
        # Note: width = col_max - col_min, height = row_max - row_min
        window = Window.from_slices((row_min, row_max), (col_min, col_max))
        
        # Read the data from the window
        subarray = src.read(1, window=window)  # Reads the first band; adjust if multi-band.

    # Optionally save as a NumPy binary file
    if output_npy:
        np.save(output_npy, subarray)

    return subarray


def add_climate_zone_band(original_path, temp_path, climate_zones, descriptor='CLIMATE_ZONES'):
    """
    Adds a new band to an existing TIFF file representing climate zones.

    Parameters:
        original_path (str): Path to the original TIFF file.
        temp_path (str): Path for temporarily storing the modified TIFF file.
        climate_zones (ndarray): Array containing the climate zone data.
        descriptor (str): Description for the new band.
    """
    # Open the original TIFF file
    dataset = gdal.Open(original_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Failed to open the file: {original_path}")

    # Read the original data
    arr = dataset.ReadAsArray()

    # Metadata and band names
    metadata = dataset.GetMetadata()
    band_names = [dataset.GetRasterBand(i+1).GetDescription() for i in range(dataset.RasterCount)]
    band_names.append(descriptor)

    # Create a new TIFF file with an additional band
    driver = gdal.GetDriverByName('GTiff')
    new_dataset = driver.Create(temp_path, dataset.RasterXSize, dataset.RasterYSize, dataset.RasterCount + 1, gdal.GDT_Float32)
    if new_dataset is None:
        raise Exception(f"Failed to create the file: {temp_path}")

    # Copy metadata to the new file
    new_dataset.SetMetadata(metadata)

    # Write data from the original bands
    for i in range(dataset.RasterCount):
        new_band = new_dataset.GetRasterBand(i+1)
        new_band.WriteArray(arr[i])
        new_band.SetDescription(band_names[i])
        new_band.FlushCache()

    # Add the new climate zones band
    new_band = new_dataset.GetRasterBand(dataset.RasterCount + 1)
    new_band.WriteArray(climate_zones)
    new_band.SetDescription(descriptor)
    new_band.FlushCache()

    # Close datasets to release file locks
    dataset = None
    new_dataset = None

    # Replace the original file with the new file
    os.remove(original_path)  # Remove the original file
    os.rename(temp_path, original_path)  # Rename the new file to the original file name




def process_tiff_file(tif_path, file0, temp_tif_name):
    # Open the first file in the folder
    with gdal.Open(os.path.join(tif_path, file0)) as ds:
        if ds.RasterCount == 10:
            print(f'Already processed {file0}')
            return
        metadata = ds.GetMetadata()

    # Calculate top left and bottom right coordinates
    top_left, bot_right = calculate_coordinates(metadata)

    # Path to the climate zone TIFF
    tiff_climate = "/home/ccollado/2_phileo_fm/pretrain/foundation_uniphi/climate_4326.tif"
    
    # Extract subarray from climate zone TIFF
    extracted_data = extract_subarray(tiff_climate, top_left, bot_right, output_npy=None)

    # Calculate zoom factors and rescale
    if extracted_data.size != 0:
        zoom_factor_y = 2248 / extracted_data.shape[0]
        zoom_factor_x = 2248 / extracted_data.shape[1]
        climate_zones = zoom(extracted_data, (zoom_factor_y, zoom_factor_x), order=0)
    else:
        climate_zones = np.zeros((2248, 2248), dtype=np.uint8)

    # Paths for original and temporary files
    original_path = os.path.join(tif_path, file0)
    temp_path = os.path.join(tif_path, temp_tif_name)

    # Add climate zone band to original TIFF
    add_climate_zone_band(original_path, temp_path, climate_zones, descriptor='CLIMATE_ZONES')



if __name__ == '__main__':
    import os
    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm
    import functools

    tif_path = '/home/ccollado/phileo_phisat2/MajorTOM/tiff_files'
    tif_files = sorted([f for f in os.listdir(tif_path) if f.endswith('.tif')])

    def parallel_wrapper(file_name):
        # Create a unique temp file name for each original TIFF file
        base_name = os.path.splitext(file_name)[0]
        temp_tif_name = f"{base_name}_temp.tif"
        process_tiff_file(tif_path, file_name, temp_tif_name)

    # Use a Pool to process files in parallel
    with Pool(cpu_count()) as p:
        # Use tqdm to show progress while mapping the function
        list(tqdm(p.imap(parallel_wrapper, tif_files), total=len(tif_files)))
