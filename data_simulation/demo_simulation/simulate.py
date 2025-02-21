from pathlib import Path
from shapely.geometry import box
import rasterio
from data_simulation.pretraining.src import *
from data_simulation import climate_tif_path

# SETUP MAJORTOM DATASET

SOURCE_DATASET = 'Major-TOM/Core-S2L1C' # Identify HF Dataset
DATASET_DIR = Path('./data/Major-TOM/')
DATASET_DIR.mkdir(exist_ok=True, parents=True)
ACCESS_URL = 'https://huggingface.co/datasets/{}/resolve/main/metadata.parquet?download=true'.format(SOURCE_DATASET)
LOCAL_URL = DATASET_DIR / '{}.parquet'.format(ACCESS_URL.split('.parquet')[0].split('/')[-1])

local_dir = './data/'

gdf = metadata_from_url(ACCESS_URL, LOCAL_URL)


# Define the center of Lausanne
lausanne_lat, lausanne_lon = 46.519630, 6.632130

# Define the number of kilometers to extend in each direction
km = 20

# Approximate conversions
km_per_degree_lat = 111  # 1 degree latitude ≈ 111 km
km_per_degree_lon = 85   # 1 degree longitude ≈ 85 km (varies by latitude)

# Compute shifts in degrees
lat_shift = km / km_per_degree_lat
lon_shift = km / km_per_degree_lon

# Define bounding box
left, bottom, right, top = (
    lausanne_lon - lon_shift,
    lausanne_lat - lat_shift,
    lausanne_lon + lon_shift,
    lausanne_lat + lat_shift
)

region = box(left, bottom, right, top)


# FILTER AND FORMAT THE DATAFRAME
filtered_df = filter_metadata(gdf,
                              cloud_cover = (0,5), # cloud cover between 0% and 10%
                              region = region,
                              daterange=('2020-01-01', '2025-01-01'),
                              nodata=(0.0,0.0) # only 0% of no data allowed
                              )

print(f'Number of rows: {len(filtered_df)}')


filter_download(filtered_df, local_dir=local_dir, source_name='L1C', by_row=True)
filter_download(filtered_df, local_dir=local_dir, source_name='L2A', by_row=True)
import pdb; pdb.set_trace()

# 3. Simulate
import os
from osgeo import gdal
gdal.DontUseExceptions()

import pandas as pd
from tqdm import tqdm

from data_simulation import local_directory_majortom
from data_simulation.pretraining.workflow_majortom_phisat2 import phisat2simulation
from data_simulation import sh_config

from multiprocessing import Pool
from functools import partial

from data_simulation import df_to_simulate_path



tif_bands = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'cloud_mask']
ds = MajorTOM(df_simulation, os.path.join(local_dir, 'L1C'), tif_bands=tif_bands, combine_bands=True)

output_path_tifs = os.path.join(local_dir, 'tiff_files')
os.makedirs(output_path_tifs, exist_ok=True)

rows = df_simulation.to_dict('records')


process_partial = partial(
    process_row,
    tif_bands=tif_bands,
    sh_client_id=sh_config.sh_client_id,
    sh_client_secret=sh_config.sh_client_secret,
    output_path_tifs=output_path_tifs
)

for row in tqdm(rows, total=len(rows)):
    process_partial(row)



import pdb; pdb.set_trace()
