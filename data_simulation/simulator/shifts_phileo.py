# --------------------------------------------
# IMPORTS
# --------------------------------------------

import datetime
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

import cv2
import numpy as np
import geopandas as gpd
import rasterio

from eolearn.core import (
    EOTask, 
    EOPatch,
    EOWorkflow,
    FeatureType,
    MapFeatureTask,
    RemoveFeatureTask,
    linearly_connect_tasks,
    EOExecutor,
)
from eolearn.features import SimpleFilterTask
from eolearn.io import SentinelHubInputTask
from eolearn.features.utils import spatially_resize_image as resize_images
from sentinelhub import (
    BBox,
    DataCollection,
    SHConfig,
    get_utm_crs,
    wgs84_to_utm,
    SentinelHubRequest, 
    MimeType, 
    CRS, 
    bbox_to_dimensions
)

from sentinelhub.exceptions import SHDeprecationWarning
from tqdm.auto import tqdm

import cupy as cp
from cupyx.scipy.fft import fft2, ifft2
from pyproj import Proj, Transformer


from phisat2_constants import (
    S2_BANDS,
    S2_RESOLUTION,
    BBOX_SIZE,
    PHISAT2_RESOLUTION,
    ProcessingLevels,
    WORLD_GDF,
)
from phisat2_utils import (
    AddPANBandTask,
    AddMetadataTask,
    CalculateRadianceTask,
    CalculateReflectanceTask,
    SCLCloudTask,
    BandMisalignmentTask,
    PhisatCalculationTask,
    AlternativePhisatCalculationTask,
    CropTask,
    GriddingTask,
    ExportGridToTiff,
    get_extent,
)

from utils import *

gdal.DontUseExceptions()


# filter out some SHDeprecationWarnings
import warnings

warnings.filterwarnings("ignore", category=SHDeprecationWarning)




# --------------------------------------------
# CUSTOM FUNCTIONS
# --------------------------------------------


def shift_array(arr, vertical_shift, horizontal_shift):
    # Determine whether the input array is a CuPy array or a NumPy array
    if isinstance(arr, cp.ndarray):
        xp = cp  # Use CuPy for GPU arrays
    else:
        xp = np  # Use NumPy for CPU arrays
    
    # Shift the array using the appropriate roll function
    shifted_array = xp.roll(arr, shift=(vertical_shift, horizontal_shift), axis=(0, 1))
    
    # Zero padding for vertical shifts
    if vertical_shift > 0:
        shifted_array[:vertical_shift, :] = 0
    elif vertical_shift < 0:
        shifted_array[vertical_shift:, :] = 0
    
    # Zero padding for horizontal shifts
    if horizontal_shift > 0:
        shifted_array[:, :horizontal_shift] = 0
    elif horizontal_shift < 0:
        shifted_array[:, horizontal_shift:] = 0
    
    return shifted_array

def find_best_shift_fft(arr1, arr2):
    # Compute the cross-correlation in the frequency domain using FFT
    f1 = fft2(arr1)
    f2 = fft2(arr2)
    
    # Compute the cross-correlation using the conjugate of f2
    cross_corr = ifft2(f1 * cp.conj(f2))
    
    # Find the location of the peak in the cross-correlation array
    max_idx = cp.unravel_index(cp.argmax(cp.abs(cross_corr)), cross_corr.shape)
    
    # Calculate the shifts based on the peak position
    shift_y = max_idx[0] - arr1.shape[0] if max_idx[0] >= arr1.shape[0] // 2 else max_idx[0]
    shift_x = max_idx[1] - arr1.shape[1] if max_idx[1] >= arr1.shape[1] // 2 else max_idx[1]
    
    shift_y = -shift_y
    shift_x = -shift_x

    # Calculate the max similarity and normalize between 0 and 1 using norms
    shifted_array = shift_array(arr1, shift_y.get(), shift_x.get())
    similarity = float(cp.mean(shifted_array == arr2))
    
    return shift_y.get(), shift_x.get(), similarity

def shifted_coords(lat, lon, dx, dy, resolution=10):
    """
    Calculate new coordinates given a shift in pixels using pyproj.
    
    Parameters:
    - lat (float): Original latitude.
    - lon (float): Original longitude.
    - dx (int): Shift in pixels to the right (positive) or left (negative).
    - dy (int): Shift in pixels up (positive) or down (negative).
    - resolution (float): Spatial resolution in meters/pixel.
    
    Returns:
    - (float, float): New latitude and longitude.
    """
    # Define a projection transformer that projects geographic coordinates to meters and vice versa
    # EPSG:4326 is the code for WGS 84
    # EPSG:3857 is the World Mercator projection commonly used in web mapping applications
    transformer_to_meters = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    transformer_to_latlon = Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True)

    # Convert original latitude and longitude to meters
    x, y = transformer_to_meters.transform(lon, lat)

    # Calculate new coordinates in meters
    x_new = x + dx * resolution
    y_new = y + dy * resolution

    # Convert back to latitude and longitude
    new_lon, new_lat = transformer_to_latlon.transform(x_new, y_new)

    return new_lat, new_lon


# --------------------------------------------
# --------------------------------------------
# MAIN FUNCTION
# --------------------------------------------
# --------------------------------------------


def find_best_shift(sh_config, image_example):
    # --------------------------------------------
    # COORDINATES
    # --------------------------------------------

    labels_df_path = '../data_info/labels_df.csv'
    locations_df_path = '../data_info/locations_df.csv'

    labels_df = pd.read_csv(labels_df_path, index_col=0)
    labels_df = labels_df.where(pd.notnull(labels_df), None)
    locations_df = pd.read_csv(locations_df_path, index_col=0)

    lat_topleft, lon_topleft, lat_botright, lon_botright, height_pixels, width_pixels = locations_df.loc[image_example]
    height_pixels, width_pixels = int(height_pixels), int(width_pixels)

    if height_pixels > 2500 or width_pixels > 2500:
        raise ValueError(f"Image {image_example} is too large: {height_pixels}x{width_pixels}")

    # Get the bounding box based on the top-left corner and pixel size
    bbox = get_utm_bbox_from_top_left_and_size(lat_topleft, lon_topleft, width_pixels, height_pixels)

    # --------------------------------------------
    # GET ESA WORLDCOVER
    # --------------------------------------------

    # GET FROM PHILEO - ESA WORLDCOVER
    phileo_lc_path = f'/home/ccollado/phileo_NFS/phileo_data/downstream/downstream_dataset_tifs/{dict(labels_df.loc[image_example])["_label_lc.tif"]}'
    dataset = gdal.Open(phileo_lc_path)
    phileo_lc = src_band(dataset, 'all')[0]
    dataset = None

    # --------------------------------------------
    # GET BEST SHIFT
    # --------------------------------------------

    hub_wc_new = phileo_lc
    lat = lat_topleft
    lon = lon_topleft

    for i in range(50):
        best_shift_y, best_shift_x, similarity = find_best_shift_fft(cp.array(hub_wc_new), cp.array(phileo_lc))
        # print(f"Best shift: ({best_shift_y}, {best_shift_x}), Similarity: {similarity}")
        lat, lon = shifted_coords(lat, lon, -best_shift_x, best_shift_y)
        bbox_new = get_utm_bbox_from_top_left_and_size(lat, lon, width_pixels, height_pixels)
        hub_wc_new = get_worldcover(sh_config, bbox_new)
        
        if best_shift_x == 0 and best_shift_y == 0 and i>0:
            break
        
    else:  # This 'else' is executed only if the for loop exits normally without a break
        print(f"{image_example}: Warning: No correct shift found")
        lat, lon = None, None

    final_lat, final_lon = lat, lon
    
    return final_lat, final_lon, similarity


























