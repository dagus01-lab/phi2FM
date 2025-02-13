from pathlib import Path
from shapely.geometry import box
import rasterio
from src import *
from data_simulation import climate_tif_path

# SETUP MAJORTOM DATASET

SOURCE_DATASET = 'Major-TOM/Core-S2L1C' # Identify HF Dataset
DATASET_DIR = Path('./data/Major-TOM/')
DATASET_DIR.mkdir(exist_ok=True, parents=True)
ACCESS_URL = 'https://huggingface.co/datasets/{}/resolve/main/metadata.parquet?download=true'.format(SOURCE_DATASET)
LOCAL_URL = DATASET_DIR / '{}.parquet'.format(ACCESS_URL.split('.parquet')[0].split('/')[-1])

gdf = metadata_from_url(ACCESS_URL, LOCAL_URL)


# DEFINE THE REGION TO DOWNLOAD

with rasterio.open(climate_tif_path) as src:
    bounds = src.bounds
left, bottom, right, top = bounds

# Adjust the bounds by the margin
margin = 1
adjusted_left = left + margin
adjusted_bottom = bottom + margin
adjusted_right = right - margin
adjusted_top = top - margin

region = box(adjusted_left, adjusted_bottom, adjusted_right, adjusted_top)


# FILTER AND FORMAT THE DATAFRAME
filtered_df = filter_metadata(gdf,
                              cloud_cover = (0,5), # cloud cover between 0% and 10%
                              region = region,
                              daterange=('2020-01-01', '2025-01-01'),
                              nodata=(0.0,0.0) # only 0% of no data allowed
                              )

reduced_df = filtered_df[::77] # take every 77th row to sample uniformly across the world
reduced_df.to_csv('df_to_download.csv', index=False)