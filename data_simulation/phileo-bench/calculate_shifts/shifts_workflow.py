# %%
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
from shifts_phileo import find_best_shift, get_worldcover

gdal.DontUseExceptions()


# filter out some SHDeprecationWarnings
import warnings

warnings.filterwarnings("ignore", category=SHDeprecationWarning)

# %% [markdown]
# ## Requirements

# %%
sh_config = SHConfig()
sh_config.sh_client_id = "aa0e165a-f9a4-49fa-9b21-1944e145f561"
sh_config.sh_client_secret = "N3RBoykGJj58mLcIEgbKBlFpzS4vsSOG"

PROCESSING_LEVEL = ProcessingLevels.L1C

# %%
labels_df_path = '../data_info/labels_names.csv'
locations_df_path = '../data_info/old_locations.csv'

labels_df = pd.read_csv(labels_df_path, index_col=0)
labels_df = labels_df.where(pd.notnull(labels_df), None)
locations_df = pd.read_csv(locations_df_path, index_col=0)

unique_images = labels_df.index

new_locations = pd.read_csv('../data_info/new_locations.csv', index_col=0)

# %%
image_example = 'denmark-1_1_0'

phileo_lc_path = f'/home/ccollado/phileo_NFS/phileo_data/downstream/downstream_dataset_tifs/{dict(labels_df.loc[image_example])["_label_lc.tif"]}'
dataset = gdal.Open(phileo_lc_path)
phileo_lc = src_band(dataset, 'all')[0]
dataset = None

lat_topleft, lon_topleft, lat_botright, lon_botright, height_pixels, width_pixels = locations_df.loc[image_example]
lat_topleft_corr, lon_topleft_corr, similarity = new_locations.loc[image_example]
bbox_downloaded = get_utm_bbox_from_top_left_and_size(lat_topleft_corr, lon_topleft_corr, width_pixels, height_pixels)
wc_downloaded = get_worldcover(sh_config, bbox_downloaded)

print(f'Similarity: {np.mean(wc_downloaded == phileo_lc)}')

# %%
if True:
    for i, image_example in enumerate(tqdm(unique_images)):
        print(f'Processing image {image_example}')
        # Check if the image_example is already in the DataFrame
        if image_example not in new_locations.index:
            lat_topleft, lon_topleft, lat_botright, lon_botright, height_pixels, width_pixels = locations_df.loc[image_example]

            height_pixels, width_pixels = int(height_pixels), int(width_pixels)

            if height_pixels > 2500 or width_pixels > 2500:
                print(f'Image {image_example} is too large, skipping')
                continue

            final_lat, final_lon, similarity = find_best_shift(sh_config, image_example)
            new_locations.loc[image_example] = [final_lat, final_lon, similarity]
        
        # Save progress intermittently
        if i % 2 == 0:
            new_locations.to_csv('../data_info/new_locations.csv')
            print(f'{i} - Saved progress - last image: {image_example}')

        # Additional safety save every 100 iterations
        if i % 100 == 0:
            backup_path = f'../data_info/backups/new_loc_{i}.csv'
            new_locations.to_csv(backup_path)
            print(f'{i} - Saved backup to {backup_path}')



