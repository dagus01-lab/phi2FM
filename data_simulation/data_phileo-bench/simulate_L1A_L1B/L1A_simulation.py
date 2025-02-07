import os
import re
import json

cwd = os.getcwd()
print(cwd)

import sys
sys.path.append('/home/ccollado/1_simulate_data')
sys.path.append('/home/ccollado/1_simulate_data/phisat_2')


from osgeo import gdal
gdal.DontUseExceptions()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patches
import random

from tqdm import tqdm

from phisat_2.utils import tiff2array, plot_array_bands, get_corner_coordinates, get_centroid_coordinates, rgb_bands, src_band, stats_array, get_band_names

from phisat_2.workflow import phisat2simulation

from sentinelhub import SHConfig

os.chdir(cwd)

print(os.getcwd())

# ------------------------
# DATA
# ------------------------

dates_df = pd.read_csv('/home/ccollado/1_simulate_data/data_info/final_dates.csv', index_col=1)

final_locs_df = pd.read_csv('/home/ccollado/1_simulate_data/data_info/final_locations.csv', index_col=0)
final_locs_df = final_locs_df.where(pd.notna(final_locs_df), other=None)
sorted_df = final_locs_df.sort_values(by='image_size', ascending=False)


# ------------------------
# UTILS
# ------------------------

def check_and_modify_image_name(image_name):
    # Check if the image name ends with '_0'
    if image_name.endswith('_0'):
        # Remove the '_0' from the image name
        return image_name[:-2]  # Remove the last two characters ('_0')
    else:
        # Raise an error if it does not end with '_0'
        raise ValueError(f"Error: The image name '{image_name}' does not end in '_0'.")

sh_config = SHConfig()
sh_config.sh_client_id = "2dc91a34-ed44-485b-9994-636223d83a63"
sh_config.sh_client_secret = "zjwugSRMlAqaWYCRqnAS1wFhB1Bvit3u"

output_path_tifs = '/home/ccollado/phileo_phisat2/L1A/tiff_files'


# ------------------------
# RUN WORKFLOW
# ------------------------

for i, image_name in enumerate(tqdm(sorted_df.index)):

    output_file_name = f"{check_and_modify_image_name(image_name)}.tif"
    time_interval = dates_df.loc[output_file_name, 'date']

    if output_file_name not in os.listdir(output_path_tifs):
        print(f'{i+1}/{len(final_locs_df)}: {image_name}')
        
        lat_topleft, lon_topleft, height_pixels, width_pixels, roads_file, buildings_file = final_locs_df.loc[image_name][['lat_topleft', 'lon_topleft', 
                                                                                                                            'height', 'width', 
                                                                                                                            'road_label', 'building_label']]
        try:
            # Try to run the workflow with maxcc=0.05
            workflow = phisat2simulation(lat_topleft=lat_topleft, lon_topleft=lon_topleft, width_pixels=width_pixels, height_pixels=height_pixels, 
                                         time_interval=time_interval, output_file_name=output_file_name, roads_file=roads_file, 
                                         buildings_file=buildings_file, maxcc=0.95, threshold_cloudless=0.95, process_level='L1A',
                                         threshold_snow=0.4, output_path_tifs=output_path_tifs, 
                                         sh_client_id=sh_config.sh_client_id, sh_client_secret=sh_config.sh_client_secret,
                                         plot_results=False)
            workflow.direct_download()
        
        except Exception as retry_error:
            raise retry_error
