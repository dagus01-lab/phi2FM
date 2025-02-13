# %%
import os
import json

from osgeo import gdal
gdal.DontUseExceptions()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patches
from s2cloudless import S2PixelCloudDetector

from tqdm import tqdm

from data_simulation.simulator.workflow import phisat2simulation

# %%
import os

import numpy as np

from sentinelhub import (
    BBox,
    DataCollection,
    SHConfig,
    get_utm_crs,
    wgs84_to_utm,
)
from sentinelhub.exceptions import SHDeprecationWarning
from tqdm.auto import tqdm


from data_simulation.simulator.phisat2_constants import ProcessingLevels

from data_simulation import label_names_path, old_locations_path, new_locations_path, sh_config

# from data_simulation.simulator.utils import None

gdal.DontUseExceptions()


# %%
ProcessingLevels._member_names_

# %%
PROCESSING_LEVEL = ProcessingLevels.L1C
PROCESSING_LEVEL

# %% [markdown]
# ## Locations

# %%
image_example = 'uganda-1_3_0'

# %%
labels_df = pd.read_csv(label_names_path, index_col=0)
labels_df = labels_df.where(pd.notnull(labels_df), None)

locations_df = pd.read_csv(old_locations_path, index_col=0)

new_locs_df = pd.read_csv(new_locations_path, index_col=0)
new_locs_df.columns = new_locs_df.columns.str.strip()


# %%
new_locs_df

# %%
min_sim = 0.9
new_locs_df_90 = new_locs_df[new_locs_df['similarity'] > min_sim]
new_locs_df_drop = new_locs_df.drop(new_locs_df_90.index)
print(f'There are {len(new_locs_df_90)} locations with similarity greater than 0.9, and {len(new_locs_df)} locations in total.')

# %%
locations_df_2500 = locations_df[(locations_df['width'] <= 2500) & (locations_df['height'] <= 2500)].copy()
locations_df_2500.drop(columns=['lat_botright', 'lon_botright'], inplace=True)
print(f'There are {len(locations_df_2500)} locations with width and height less than 2500 pixels, and {len(locations_df)} locations in total.')

locations_df_2500.loc[:, 'building_label'] = labels_df['_label_building.tif']
locations_df_2500.loc[:, 'road_label'] = labels_df['_label_roads.tif']
locations_df_2500.loc[:, 'lc_label'] = True
locations_df_2500.loc[:, 'similarity'] = new_locs_df['similarity'] 
locations_df_2500.sample(5)

# %%
final_locs_df = locations_df_2500.copy()
final_locs_df.update(new_locs_df_90[['lat_topleft', 'lon_topleft']])

final_locs_df.loc[new_locs_df_drop.index, ['building_label', 'road_label']] = False
final_locs_df.replace({False: None, np.nan: None}, inplace=True)

non_zero_indices = final_locs_df.index[final_locs_df.index.str[-1] != '0']
final_locs_df.drop(non_zero_indices, inplace=True)

# %%
final_locs_df['road_label']

# %% [markdown]
# ## Why less files than in train + test labels of data?
# 
# Well, because we drop images that are too large

# %%
import json
from data_simulation import train_test_files_path

with open(train_test_files_path, 'r') as file:
    train_test_files = json.load(file)

tt_files = train_test_files['train_locations'] + train_test_files['test_locations']
print(f'There are {len(tt_files)} locations in the train and test sets.')

missing_items = [name for name in tt_files if name not in final_locs_df.index.str[:-2].tolist()]

print(f'There are {len(missing_items)} missing items.')

# %%
len(locations_df[(locations_df['width'] > 2500) | (locations_df['height'] > 2500)])

# %% [markdown]
# ### Visualize Results

# %%
# Function to extract only the country name
def extract_country_name(index):
    return index.split('_')[0]

countries_df = final_locs_df.copy()
countries_df['country_name'] = countries_df.index.map(extract_country_name)

filtered_df = countries_df[countries_df['similarity'] > min_sim]

country_counts = filtered_df['country_name'].value_counts()
total_counts = countries_df['country_name'].value_counts()

proportion_df = pd.DataFrame({
    'count': country_counts,
    'proportion': country_counts / total_counts,
    'total': total_counts
}).fillna(0)

proportion_df

# %% [markdown]
# # INPUTS

# %%
def check_and_modify_image_name(image_name):
    # Check if the image name ends with '_0'
    if image_name.endswith('_0'):
        # Remove the '_0' from the image name
        return image_name[:-2]  # Remove the last two characters ('_0')
    else:
        # Raise an error if it does not end with '_0'
        raise ValueError(f"Error: The image name '{image_name}' does not end in '_0'.")

# %%
lat_topleft, lon_topleft, height_pixels, width_pixels, roads_file, buildings_file = final_locs_df.loc[image_example][['lat_topleft', 'lon_topleft', 
                                                                                                                      'height', 'width', 
                                                                                                                      'road_label', 'building_label']]

output_file_name = f"{check_and_modify_image_name(image_example)}.tif"

# %%
time_interval = ("2021-01-01", "2021-12-31")
from data_simulation import output_path_tifs_phileobench

# %%
final_locs_df['image_size'] = final_locs_df['height'] * final_locs_df['width']

# Sort the DataFrame by the calculated image size in ascending order
sorted_df = final_locs_df.sort_values(by='image_size', ascending=True)

# sorted_df[(sorted_df['height'] < 128) | (sorted_df['width'] < 128)].sort_index()

# %%
import random
shuffled_indices = final_locs_df.index.tolist()
random.shuffle(shuffled_indices)

# Iterate through the shuffled indices
# for i, image_name in enumerate(tqdm(shuffled_indices)):
for i, image_name in enumerate(tqdm(sorted_df.index)):

    print(f'{i+1}/{len(final_locs_df)}: {image_name}')

    output_file_name = f"{check_and_modify_image_name(image_name)}.tif"

    if output_file_name not in os.listdir(output_path_tifs_phileobench):

        lat_topleft, lon_topleft, height_pixels, width_pixels, roads_file, buildings_file = final_locs_df.loc[image_name][['lat_topleft', 'lon_topleft', 
                                                                                                                            'height', 'width', 
                                                                                                                            'road_label', 'building_label']]
        try:
            # Try to run the workflow with maxcc=0.05
            workflow = phisat2simulation(lat_topleft=lat_topleft, lon_topleft=lon_topleft, width_pixels=width_pixels, height_pixels=height_pixels, 
                                         time_interval=time_interval, output_file_name=output_file_name, roads_file=roads_file, 
                                         buildings_file=buildings_file, maxcc=0.05, threshold_cloudless=0.05, 
                                         threshold_snow=0.4, output_path_tifs=output_path_tifs_phileobench,
                                         plot_results=False)
            workflow.run()
        
        except Exception as e:
            print(f"Error encountered for {image_name} with maxcc=0.05: {e}")
            print(f"Retrying with maxcc=0.1 for {image_name}")
            
            # If an error occurs, retry with maxcc=0.1
            try:
                workflow = phisat2simulation(lat_topleft=lat_topleft, lon_topleft=lon_topleft, width_pixels=width_pixels, height_pixels=height_pixels, 
                                             time_interval=time_interval, output_file_name=output_file_name, roads_file=roads_file, 
                                             buildings_file=buildings_file, maxcc=0.1, threshold_cloudless=0.05, 
                                             threshold_snow=0.4, output_path_tifs=output_path_tifs_phileobench,
                                             plot_results=False)
                workflow.run()

            except Exception as retry_error:
                raise retry_error
