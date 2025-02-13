import numpy as np
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta

import os
import json
from osgeo import gdal
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

from s2cloudless import S2PixelCloudDetector
from copy import deepcopy
import random
from pyproj import Transformer
from scipy.ndimage import zoom

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

from eolearn.core import EOTask

import warnings




def tiff2array(file_path):
    dataset = gdal.Open(file_path)
    metadata = dataset.GetMetadata()
    
    num_bands = dataset.RasterCount

    # Initialize an empty list to store arrays for each band
    bands_data = []

    # Loop through each band and read it as an array
    for i in range(1, num_bands + 1):  # Band index starts at 1 in GDAL
        band = dataset.GetRasterBand(i)
        band_array = band.ReadAsArray()
        bands_data.append(band_array)

    # Stack all the band arrays into a single 3D array
    stacked_array = np.stack(bands_data, axis=0)
    new_array = np.expand_dims(stacked_array, axis=-1)  # Adds new dimension at the end


    # Close the dataset (good practice)
    dataset = None
    
    return new_array, metadata


def plot_array_bands(array, title='RGB Image (B04=Red, B03=Green, B02=Blue)', figsize=(10, 10), plot_array = True, return_norm = False):

    # Extract the RGB bands
    red_band = array[3]  # B04
    green_band = array[2]  # B03
    blue_band = array[1]  # B02

    # Log-transform the bands because max is too high compared to mean
    red_band_plot = np.log1p(red_band)
    green_band_plot = np.log1p(green_band)
    blue_band_plot = np.log1p(blue_band)

    # Stack the bands to form an RGB image
    rgb_image = np.stack((red_band_plot, green_band_plot, blue_band_plot), axis=-1)

    # Normalize the RGB image for display purposes
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

    if plot_array:
        # Plot the RGB image
        plt.figure(figsize=figsize)
        plt.imshow(rgb_image)
        plt.title(title)
        plt.axis('off')
        plt.show()
        
    if return_norm:
        return rgb_image
    
    return np.stack((red_band, green_band, blue_band), axis=-1)


def decode_lat_lng(encoded_coords):

    # Decode latitude
    lat = 90 - (encoded_coords[0] * 180)
    
    # Decode longitude
    lng_sin = 2 * encoded_coords[1] - 1
    lng_cos = 2 * encoded_coords[2] - 1
    
    # Use atan2 to get the longitude in degrees
    lng = np.degrees(np.arctan2(lng_sin, lng_cos))
    
    return lat, lng


def get_corner_coordinates(stacked_array):
    # First (top-left) point
    top_left_coords = decode_lat_lng(stacked_array[:, 0, 0])
    
    # Last (bottom-right) point
    bottom_right_coords = decode_lat_lng(stacked_array[:, -1, -1])
    
    return top_left_coords, bottom_right_coords

def get_centroid_coordinates(staked_array):
    # Get the corner coordinates
    top_left_coords, bottom_right_coords = get_corner_coordinates(staked_array)
    
    # Calculate the centroid coordinates
    centroid_lat = (top_left_coords[0] + bottom_right_coords[0]) / 2
    centroid_lng = (top_left_coords[1] + bottom_right_coords[1]) / 2
    
    return centroid_lat, centroid_lng

def get_utm_bbox_from_corners(lat_top_left: float, lon_top_left: float, lat_bottom_right: float, lon_bottom_right: float):
    """
    Returns a bounding box given the top-left and bottom-right coordinates of the area-of-interest in WGS84
    """
    
    # Convert top-left and bottom-right corners from WGS84 to UTM coordinates
    east_top_left, north_top_left = wgs84_to_utm(lon_top_left, lat_top_left)
    east_bottom_right, north_bottom_right = wgs84_to_utm(lon_bottom_right, lat_bottom_right)

    # Round the coordinates to the nearest multiple of 10
    east_top_left, north_top_left = 10 * int(east_top_left / 10), 10 * int(north_top_left / 10)
    east_bottom_right, north_bottom_right = 10 * int(east_bottom_right / 10), 10 * int(north_bottom_right / 10)
    
    # Determine the CRS based on the top-left corner (you can change this logic if necessary)
    crs = get_utm_crs(lon_top_left, lat_top_left)

    return BBox(
        bbox=(
            (east_top_left, north_top_left),  # Top-left corner in UTM
            (east_bottom_right, north_bottom_right),  # Bottom-right corner in UTM
        ),
        crs=crs,
    )

def get_utm_bbox_from_top_left_and_size(lat_top_left: float, lon_top_left: float, width_pixels: int, height_pixels: int, resolution: float = 10):
    # Convert top-left corner from WGS84 to UTM coordinates
    east_top_left, north_top_left = wgs84_to_utm(lon_top_left, lat_top_left)
    
    # Calculate the east and north coordinates of the bottom-right corner
    east_bottom_right = east_top_left + width_pixels * resolution
    north_bottom_right = north_top_left - height_pixels * resolution
    
    # Determine the CRS based on the top-left corner
    crs = get_utm_crs(lon_top_left, lat_top_left)
    
    return BBox(
        bbox=(
            (east_top_left, north_top_left),
            (east_bottom_right, north_bottom_right),
        ),
        crs=crs,
    )


def src_band(dataset, identifier, warn=True):
    if identifier == 'bands':
        identifier = list(range(1, 9))
    if identifier == 'labels':
        identifier = list(range(9, dataset.RasterCount + 1))
    if identifier == 'all':
        identifier = list(range(1, dataset.RasterCount + 1))
    if identifier == 'rgb':
        return rgb_bands(dataset, normalize=False)
    if identifier == 'rgb_norm':
        return rgb_bands(dataset, normalize=True)
    if identifier == 'bands_names':
        bands = [dataset.GetRasterBand(i).GetDescription() for i in range(1, dataset.RasterCount + 1)]
        datatypes = [gdal.GetDataTypeName(dataset.GetRasterBand(i).DataType) for i in range(1, dataset.RasterCount + 1)]
        bands_dict = dict(zip(bands, datatypes))
        return bands_dict

    # Helper function to get a band by description or index
    def get_band_by_identifier(dataset, ident):
        if isinstance(ident, str):  # Search by description if it's a string                
            for i in range(dataset.RasterCount):
                band_index = i + 1  # GDAL band indices are 1-based
                band = dataset.GetRasterBand(band_index)
                description = band.GetDescription()
                if description == ident:
                    return band.ReadAsArray()  # Return the band data
            if warn:
                warnings.warn(f"Band with description '{ident}' not found.")
            return None
        elif isinstance(ident, int):  # Read directly if it's an integer
            if 1 <= ident <= dataset.RasterCount:  # Ensure index is within valid range
                band = dataset.GetRasterBand(ident)
                return band.ReadAsArray()  # Return the band data
            else:
                if warn:
                    warnings.warn(f"Band index {ident} is out of range.")
                return None
        else:
            if warn:
                warnings.warn(f"Identifier must be a string (description) or an integer (index).")
            return None
    
    # If the identifier is a list or tuple, retrieve multiple bands
    if isinstance(identifier, (list, tuple)):
        bands = [get_band_by_identifier(dataset, ident) for ident in identifier]
        return np.stack(bands, axis=0)  # Stack the bands into a single array
    
    # If the identifier is a single value (string or integer)
    else:
        return get_band_by_identifier(dataset, identifier)

def rgb_bands(dataset, normalize=False):
    # Select the RGB bands by their descriptions
    rgb_image = src_band(dataset, ["PHISAT2-B04", "PHISAT2-B03", "PHISAT2-B02"], warn=False)
    if rgb_image[0] is None:
        print("RGB bands not found. Using default bands 3, 2, 1.")
        rgb_image = src_band(dataset, [4,3,2])  # Indices start at 1 instead of 0
    # Apply normalization if needed
    print(rgb_image.shape)
    if normalize:
        if np.max(rgb_image) > 100:
            rgb_image = np.log1p(rgb_image)
        else:
            rgb_image = np.clip(3.5 * rgb_image, 0, 1)

        for i in range(rgb_image.shape[0]):  # Loop over the bands (axis 0)
            rgb_image[i] = (rgb_image[i] - rgb_image[i].min()) / (rgb_image[i].max() - rgb_image[i].min())
    
    # Transpose the bands to [height, width, channels]
    rgb_array = np.transpose(rgb_image, [1, 2, 0])
    
    return rgb_array

def export_eop(eop, filename='eopatch_data.tif', include_metadata=True, metadata_option='full', include_masks=True, include_labels=False,
               include_clouds=True, lat_topleft=None, lon_topleft=None):

    from phisat2_constants import (
        S2_BANDS,
        S2_RESOLUTION,
        BBOX_SIZE,
        PHISAT2_RESOLUTION,
        ProcessingLevels,
        WORLD_GDF,
    )

    bands_to_include = [
        eop.data['PHISAT2-BANDS'][0]  # Always include PHISAT2-BANDS
    ]

    band_names = [
        'PHISAT2-B02', ' PHISAT2-B03', 'PHISAT2-B04', 'PHISAT2-PAN', 'PHISAT2-B08', 'PHISAT2-B05', 'PHISAT2-B06', 'PHISAT2-B07'
    ]

    if include_clouds:
        bands_to_include.append(eop.data['CLOUD_PROB_RES'][0])
        band_names.append('CLOUD_PROB')
        
        bands_to_include.append(eop.data['CLIMATE_ZONES'][0])
        band_names.append('CLIMATE_ZONES')

    # Include labels if specified
    if include_labels:
        bands_to_include.append(eop.data['WORLD_COVER_RES'][0])
        band_names.append('WORLD_COVER')
        if 'ROADS_RES'in eop.data.keys():
            bands_to_include.append(eop.data['ROADS_RES'][0])
            band_names.append('ROADS')

        if 'BUILDINGS_RES'in eop.data.keys():
            bands_to_include.append(eop.data['BUILDINGS_RES'][0])
            band_names.append('BUILDINGS')

    # Include masks if specified
    if include_masks:
        bands_to_include.append(eop.mask['SCL_CIRRUS_RES'][0])
        band_names.append('SCL_CIRRUS')
        bands_to_include.append(eop.mask['SCL_CLOUD_RES'][0])
        band_names.append('SCL_CLOUD')
        bands_to_include.append(eop.mask['SCL_CLOUD_SHADOW_RES'][0])
        band_names.append('SCL_CLOUD_SHADOW')
        bands_to_include.append(eop.mask['dataMask_RES'][0])
        band_names.append('dataMask')

    # Concatenate all selected bands along the last axis
    all_bands = np.concatenate(bands_to_include, axis=-1)

    # Scale the data and convert to uint16
    for i, band_name in enumerate(band_names):
        if band_name != 'CLIMATE_ZONES':
            all_bands[..., i] = all_bands[..., i] * 10000
    all_bands = np.clip(all_bands, 0, 65534)
    all_bands = all_bands.astype(np.uint16)

    # Define the spatial transform and CRS
    transform = from_origin(eop.bbox.min_x, eop.bbox.max_y, PHISAT2_RESOLUTION, PHISAT2_RESOLUTION)  # Top-left corner and pixel size
    crs = f"EPSG:{eop.bbox._crs.value}"

    # Save to GeoTIFF
    with rasterio.open(
        filename, 'w', driver='GTiff',
        height=all_bands.shape[0], width=all_bands.shape[1],
        count=all_bands.shape[2], dtype='uint16',
        crs=crs, transform=transform, compress='lzw'
    ) as dst:
        for i in range(all_bands.shape[2]):
            dst.write(all_bands[:, :, i], i+1)
            dst.set_band_description(i+1, band_names[i])

        # Conditionally save metadata as tags
        # raise NotImplementedError("Metadata saving is not implemented yet.\n\n\n eop:\n\n\n", eop)
        if include_metadata:
            if metadata_option == 'full':
                dst.update_tags(
                    earth_sun_dist=float(eop.scalar['earth_sun_dist'].item()),  # Extract scalar value using .item()
                    sol_irr_B02=float(eop.scalar['sol_irr_B02'].item()),        # Extract scalar value using .item()
                    sol_irr_B03=float(eop.scalar['sol_irr_B03'].item()),        # Extract scalar value using .item()
                    sol_irr_B04=float(eop.scalar['sol_irr_B04'].item()),        # Extract scalar value using .item()
                    sol_irr_B05=float(eop.scalar['sol_irr_B05'].item()),        # Extract scalar value using .item()
                    sol_irr_B06=float(eop.scalar['sol_irr_B06'].item()),        # Extract scalar value using .item()
                    sol_irr_B07=float(eop.scalar['sol_irr_B07'].item()),        # Extract scalar value using .item()
                    sol_irr_B08=float(eop.scalar['sol_irr_B08'].item()),        # Extract scalar value using .item()
                    # sol_irr_PAN=float(eop.scalar['sol_irr_PAN'].item()),        # Extract scalar value using .item()
                    maxcc=float(eop.meta_info['maxcc']),                        # Assuming this is already a scalar
                    real_cloud_cover = float(eop.meta_info['cloud_prob_s2_cloudless']),
                    snow_coverage = float(eop.meta_info['snow_coverage']),
                    lat_topleft = lat_topleft,
                    lon_topleft = lon_topleft,
                    size_x=int(eop.meta_info['size_x']),                        # Convert to integer if needed
                    size_y=int(eop.meta_info['size_y']),                        # Convert to integer if needed
                    time_difference=str(eop.meta_info['time_difference']),      # Convert to string if datetime/timedelta
                    time_interval=str(eop.meta_info['time_interval']),          # Convert to string if datetime/timedelta
                    bbox=str(eop.bbox),                                         # Convert to string
                    timestamp=str(eop.timestamp[0])                             # Convert to string
                )
                
                if 'sol_irr_PAN' in eop.scalar.keys():
                    dst.update_tags(
                        sol_irr_PAN=float(eop.scalar['sol_irr_PAN'].item()),        # Extract scalar value using .item()
                    )
            elif metadata_option == 'minimal':
                dst.update_tags(
                    bbox=str(eop.bbox),                                     # Convert to string
                    timestamp=str(eop.timestamp[0])                            # Convert to string
                )
                
            elif metadata_option == 'none':
                pass
            
            elif metadata_option == 'local_l1c':
                dst.update_tags(
                    earth_sun_dist=float(eop.scalar['earth_sun_dist'].item()),  # Extract scalar value using .item()
                    sol_irr_B02=float(eop.scalar['sol_irr_B02'].item()),        # Extract scalar value using .item()
                    sol_irr_B03=float(eop.scalar['sol_irr_B03'].item()),        # Extract scalar value using .item()
                    sol_irr_B04=float(eop.scalar['sol_irr_B04'].item()),        # Extract scalar value using .item()
                    sol_irr_B05=float(eop.scalar['sol_irr_B05'].item()),        # Extract scalar value using .item()
                    sol_irr_B06=float(eop.scalar['sol_irr_B06'].item()),        # Extract scalar value using .item()
                    sol_irr_B07=float(eop.scalar['sol_irr_B07'].item()),        # Extract scalar value using .item()
                    sol_irr_B08=float(eop.scalar['sol_irr_B08'].item()),        # Extract scalar value using .item()
                    # sol_irr_PAN=float(eop.scalar['sol_irr_PAN'].item()),       # Extract scalar value using .item()
                    maxcc=float(eop.meta_info['maxcc']),                        # Assuming this is already a scalar
                    size_x=int(eop.meta_info['size_x']),                        # Convert to integer if needed
                    size_y=int(eop.meta_info['size_y']),                        # Convert to integer if needed
                    bbox=str(eop.bbox),                                         # Convert to string
                    timestamp=str(eop.timestamp[0]),                            # Convert to string
                    topleft_min_lon=str(eop.meta_info['topleft_min_lon']),
                    topleft_max_lat=str(eop.meta_info['topleft_max_lat']),                    
                )
                
                for meta_key in eop.meta_info['meta_list']:
                    if meta_key in eop.meta_info:  # Ensure the key exists in meta_info
                        dst.update_tags(**{meta_key: eop.meta_info[meta_key]})

                
                if 'sol_irr_PAN' in eop.scalar.keys():
                    dst.update_tags(
                        sol_irr_PAN=float(eop.scalar['sol_irr_PAN'].item()),        # Extract scalar value using .item()
                    )




# def export_eop_local_l1c(eop, filename='eopatch_data.tif', include_metadata=True, metadata_option='full', include_masks=True, include_labels=False,
#                lat_topleft=None, lon_topleft=None):

#     from phisat2_constants import (
#         S2_BANDS,
#         S2_RESOLUTION,
#         BBOX_SIZE,
#         PHISAT2_RESOLUTION,
#         ProcessingLevels,
#         WORLD_GDF,
#     )

#     bands_to_include = [
#         eop.data['PHISAT2-BANDS'][0]  # Always include PHISAT2-BANDS
#     ]

#     band_names = [
#         'PHISAT2-B02', ' PHISAT2-B03', 'PHISAT2-B04', 'PHISAT2-PAN', 'PHISAT2-B08', 'PHISAT2-B05', 'PHISAT2-B06', 'PHISAT2-B07'
#     ]

#     # Include labels if specified
#     if include_labels:
#         bands_to_include.append(eop.data['WORLD_COVER_RES'][0])
#         band_names.append('WORLD_COVER')
#         bands_to_include.append(eop.data['CLOUD_PROB_RES'][0])
#         band_names.append('CLOUD_PROB')
#         if 'ROADS_RES'in eop.data.keys():
#             bands_to_include.append(eop.data['ROADS_RES'][0])
#             band_names.append('ROADS')

#         if 'BUILDINGS_RES'in eop.data.keys():
#             bands_to_include.append(eop.data['BUILDINGS_RES'][0])
#             band_names.append('BUILDINGS')

#     # Include masks if specified
#     if include_masks:
#         bands_to_include.append(eop.mask['SCL_CIRRUS_RES'][0])
#         band_names.append('SCL_CIRRUS')
#         bands_to_include.append(eop.mask['SCL_CLOUD_RES'][0])
#         band_names.append('SCL_CLOUD')
#         bands_to_include.append(eop.mask['SCL_CLOUD_SHADOW_RES'][0])
#         band_names.append('SCL_CLOUD_SHADOW')
#         bands_to_include.append(eop.mask['dataMask_RES'][0])
#         band_names.append('dataMask')

#     # Concatenate all selected bands along the last axis
#     all_bands = np.concatenate(bands_to_include, axis=-1)

#     # Define the spatial transform and CRS
#     transform = from_origin(eop.bbox.min_x, eop.bbox.max_y, PHISAT2_RESOLUTION, PHISAT2_RESOLUTION)  # Top-left corner and pixel size
#     crs = f"EPSG:{eop.bbox._crs.value}"

#     # Save to GeoTIFF
#     with rasterio.open(
#         filename, 'w', driver='GTiff',
#         height=all_bands.shape[0], width=all_bands.shape[1],
#         count=all_bands.shape[2], dtype='float32',
#         crs=crs, transform=transform, compress='lzw'
#     ) as dst:
#         for i in range(all_bands.shape[2]):
#             dst.write(all_bands[:, :, i], i+1)
#             dst.set_band_description(i+1, band_names[i])

#         # Conditionally save metadata as tags
#         # raise NotImplementedError("Metadata saving is not implemented yet.\n\n\n eop:\n\n\n", eop)
#         if include_metadata:
#             if metadata_option == 'full':
#                 dst.update_tags(
#                     earth_sun_dist=float(eop.scalar['earth_sun_dist'].item()),  # Extract scalar value using .item()
#                     sol_irr_B02=float(eop.scalar['sol_irr_B02'].item()),        # Extract scalar value using .item()
#                     sol_irr_B03=float(eop.scalar['sol_irr_B03'].item()),        # Extract scalar value using .item()
#                     sol_irr_B04=float(eop.scalar['sol_irr_B04'].item()),        # Extract scalar value using .item()
#                     sol_irr_B05=float(eop.scalar['sol_irr_B05'].item()),        # Extract scalar value using .item()
#                     sol_irr_B06=float(eop.scalar['sol_irr_B06'].item()),        # Extract scalar value using .item()
#                     sol_irr_B07=float(eop.scalar['sol_irr_B07'].item()),        # Extract scalar value using .item()
#                     sol_irr_B08=float(eop.scalar['sol_irr_B08'].item()),        # Extract scalar value using .item()
#                     # sol_irr_PAN=float(eop.scalar['sol_irr_PAN'].item()),        # Extract scalar value using .item()
#                     maxcc=float(eop.meta_info['maxcc']),                        # Assuming this is already a scalar
#                     real_cloud_cover = float(eop.meta_info['cloud_prob_s2_cloudless']),
#                     snow_coverage = float(eop.meta_info['snow_coverage']),
#                     lat_topleft = lat_topleft,
#                     lon_topleft = lon_topleft,
#                     size_x=int(eop.meta_info['size_x']),                        # Convert to integer if needed
#                     size_y=int(eop.meta_info['size_y']),                        # Convert to integer if needed
#                     time_difference=str(eop.meta_info['time_difference']),      # Convert to string if datetime/timedelta
#                     time_interval=str(eop.meta_info['time_interval']),          # Convert to string if datetime/timedelta
#                     bbox=str(eop.bbox),                                         # Convert to string
#                     timestamp=str(eop.timestamp[0])                             # Convert to string
#                 )
                
#                 if 'sol_irr_PAN' in eop.scalar.keys():
#                     dst.update_tags(
#                         sol_irr_PAN=float(eop.scalar['sol_irr_PAN'].item()),        # Extract scalar value using .item()
#                     )
#             elif metadata_option == 'minimal':
#                 dst.update_tags(
#                     bbox=str(eop.bbox),                                     # Convert to string
#                     timestamp=str(eop.timestamp[0])                            # Convert to string
#                 )





def get_worldcover(sh_config, bbox):
  # Calculate image dimensions based on bbox and resolution
  s2_res = 10
  size = bbox_to_dimensions(bbox, resolution=s2_res)

  # --------------------------- Evalscript ----------------------------

  evalscript = """
  //VERSION=3

  function setup() {
    return {
      input: ["Map"],
      output: {
        bands: 1,
        sampleType: "UINT8"
      }
    }
  }

  function evaluatePixel(sample) {
    return [sample.Map];
  }
  """

  # ------------------------ Data Collection --------------------------

  # Define the BYOC collection using its collection ID
  collection_id = '0b940c63-45dd-4e6b-8019-c3660b81b884'
  data_collection = DataCollection.define_byoc(collection_id)

  # -------------------------- API Request ----------------------------

  request = SentinelHubRequest(
      evalscript=evalscript,
      input_data=[SentinelHubRequest.input_data(data_collection=data_collection)],
      responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
      bbox=bbox,
      size=size,
      config=sh_config,
  )

  # -------------------------- Get Data -------------------------------

  data = request.get_data()
  hub_wc = data[0]  # Extract the image array from the list
  return hub_wc




class AddLabelsTask(EOTask):
    def __init__(self, roads_file, buildings_file, folder_path_tifs, sh_config):
        
        self.sh_config = sh_config

        if roads_file:
            self.label_roads_path = os.path.join(folder_path_tifs, roads_file)
        else:
            self.label_roads_path = None
            
        if buildings_file:
            self.label_buildings_path = os.path.join(folder_path_tifs, buildings_file)
        else:
            self.label_buildings_path = None
    
    def execute(self, eopatch):
        
        hub_wc = get_worldcover(self.sh_config, eopatch.bbox)
        eopatch.data['WORLD_COVER'] = hub_wc[np.newaxis, ..., np.newaxis]

        if self.label_roads_path:
            label_roads_array, _ = tiff2array(self.label_roads_path)
            eopatch.data['ROADS'] = label_roads_array
            eopatch.scalar['ROADS_IN_DATA'] = np.array([[True]])
        else:
            eopatch.scalar['ROADS_IN_DATA'] = np.array([[False]])

        if self.label_buildings_path:
            label_buildings_array, _ = tiff2array(self.label_buildings_path)
            eopatch.data['BUILDINGS'] = label_buildings_array
            eopatch.scalar['BUILDINGS_IN_DATA'] = np.array([[True]])
        else:
            eopatch.scalar['BUILDINGS_IN_DATA'] = np.array([[False]])


        return eopatch

class ReplaceDataWithPhilEO(EOTask):
    def __init__(self, folder_path_tifs, file_names, replace_with_phileo):
        self.folder_path = folder_path_tifs
        self.file_name = file_names['image_file']
        self.replace_with_phileo = replace_with_phileo
        
    def execute(self, eopatch):
        if self.replace_with_phileo:
            # Load the image data
            image_array, _ = tiff2array(os.path.join(self.folder_path, self.file_name))
            image_bands = image_array[1:8]
            eopatch.data['BANDS'] = image_bands.transpose(3, 1, 2, 0) / 10000 # Divide to calculate reflectance
        return eopatch



class ExportEOPatchTask(EOTask):
    def __init__(self, folder_path, filename='eopatch_min.tif', include_metadata=True, metadata_option='full', lat_topleft=None, lon_topleft=None,
                 include_masks=False, include_labels=True, use_local_l1c=False):
        self.folder_path = folder_path
        self.filename = filename
        self.include_metadata = include_metadata
        self.metadata_option = metadata_option
        self.include_masks = include_masks
        self.include_labels = include_labels
        self.lat_topleft = lat_topleft
        self.lon_topleft = lon_topleft
        self.use_local_l1c = use_local_l1c

    def execute(self, eopatch):
        export_path = os.path.join(self.folder_path, self.filename)

        # Export the EOPatch to a GeoTIFF file with all arguments explicitly named
        if self.use_local_l1c:
            export_eop(eopatch, filename=export_path, 
                    include_metadata=self.include_metadata, 
                    metadata_option='local_l1c', 
                    include_masks=False,
                    include_labels=False,
                    lat_topleft=self.lat_topleft, lon_topleft=self.lon_topleft)
        else:
            export_eop(eopatch, filename=export_path, 
                    include_metadata=self.include_metadata, 
                    metadata_option=self.metadata_option, 
                    include_masks=self.include_masks, 
                    include_labels=self.include_labels,
                    lat_topleft=self.lat_topleft, lon_topleft=self.lon_topleft)

        return eopatch

class PlotResultsTask(EOTask):
    def __init__(self, plot_results=True):
        self.plot_results = plot_results
        
    def plot_images(self, eopatch):
        plot_data = np.clip(6 * eopatch.data['BANDS'][0][..., [2, 1, 0]], 0, 1)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(plot_data)
        plt.title(f"Date: {eopatch.timestamp[0]}")
        plt.axis('off')  # Turn off axis numbering and labels
        plt.show()

    def execute(self, eopatch):
        if self.plot_results:
            self.plot_images(eopatch)
        return eopatch

def stats_array(array):
    stats = {
        'Mean': np.mean(array, axis=(1, 2)),
        'Min': np.min(array, axis=(1, 2)),
        'Max': np.max(array, axis=(1, 2)),
        'Std Dev': np.std(array, axis=(1, 2)),
        'Variance': np.var(array, axis=(1, 2)),
        'Median': np.median(array, axis=(1, 2)),
    }

    # Creating a DataFrame
    band_stats_df = pd.DataFrame(stats, index=[f'Band {i+1}' for i in range(array.shape[0])])
    return band_stats_df

def get_band_names(dataset):
    band_names = []
    for i in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(i)
        band_names.append(band.GetDescription())
    return band_names


def create_time_intervals(start_date, end_date, num_intervals=2):
    # Convert input strings to datetime objects
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    
    # Generate a list of tuples for each custom interval
    current = start
    intervals = []
    
    while current < end:
        # Calculate the length of each interval within the month
        next_month = current + relativedelta(months=1)
        month_days = (next_month - current).days
        interval_length = month_days / num_intervals

        # Create intervals for the current month
        for i in range(num_intervals):
            interval_start = current
            interval_end = interval_start + relativedelta(days=(interval_length * (i + 1)))
            
            # Adjust the end of the last interval to not exceed the month or the overall end date
            if i == num_intervals - 1 or interval_end >= next_month:
                interval_end = min(next_month, end)
            
            intervals.append((interval_start.strftime("%Y-%m-%d"), interval_end.strftime("%Y-%m-%d")))
            
            # Update current to the end of this interval for the next iteration
            current = interval_end
            if current >= end:
                break

        # Move to the start of the next month if not yet at the end date
        if current < end and current < next_month:
            current = next_month
    
    return intervals




class DownloadWOCloudSnow(EOTask):
    def __init__(self, 
                 csv_file_path,
                 threshold_cloudless=0.05, 
                 threshold_snow=0.4,
                 cloud_detector_config=None,
                 input_task_all_bands=None, 
                 scl_download_task=None,
                 scl_cloud_snow_task=None, 
                 ):
        """
        Parameters:
        - threshold_cloudless (float): Maximum average cloud coverage allowed (0 to 1).
        - threshold_snow (float): Maximum average snow coverage allowed (0 to 1).
        - cloud_detector_config (dict): Configuration for S2PixelCloudDetector.
        - input_task_all_bands (EOTask): Task to process all bands.
        - scl_download_task (EOTask): Task to download SCL masks.
        - scl_cloud_snow_task (EOTask): Task to process cloud masks.
        - csv_file_path (str): Path to the CSV file storing monthly counts.
        """
        self.threshold_cloudless = threshold_cloudless
        self.threshold_snow = threshold_snow
        self.input_task_all_bands = input_task_all_bands
        self.scl_download_task = scl_download_task
        self.scl_cloud_snow_task = scl_cloud_snow_task
        self.csv_file_path = csv_file_path

        # Initialize the cloud detector
        if cloud_detector_config is None:
            self.cloud_detector = S2PixelCloudDetector(
                threshold=0.4, 
                average_over=4, 
                dilation_size=2, 
                all_bands=True
            )
        else:
            self.cloud_detector = S2PixelCloudDetector(**cloud_detector_config)

    def execute(self, eopatch):
        """
        Executes the task on the given EOPatch.

        Parameters:
        - eopatch (EOPatch): The input EOPatch containing data and masks.

        Returns:
        - EOPatch: The processed EOPatch with the selected date.
        """
        # Ensure required masks are present
        if "SCL_CLOUD" not in eopatch.mask or "SCL_SNOW" not in eopatch.mask:
            raise ValueError("EOPatch must contain 'SCL_CLOUD' and 'SCL_SNOW' masks.")

        self.bbox = eopatch.bbox

        # Calculate snow cover
        snow_data = eopatch.mask["SCL_SNOW"][..., 0]    # Assuming binary mask: 1 for snow, 0 for no snow
        avg_snow_coverage = snow_data.mean(axis=(1, 2))    # Shape: (num_timestamps,)

        # Filter timestamps based on cloud and snow thresholds
        indices_ok = np.where((avg_snow_coverage < self.threshold_snow))[0]
        ok_timestamps = [eopatch.timestamp[i] for i in indices_ok]
        if type(ok_timestamps) != list:
            ok_timestamps = [ok_timestamps]

        while not ok_timestamps:
            raise ValueError("No suitable dates found that meet cloud and snow cover thresholds.")
        
        # Read counts from the CSV file
        counts_df = self._read_counts_csv()
        probabilities = counts_df['Prob'].values

        # Combine timestamps, weights, and indices into a list of tuples
        timestamp_data = list(zip(ok_timestamps, indices_ok))

        # Initialize variables for retry mechanism
        current_threshold = self.threshold_cloudless
        max_retries = 19  # Define a maximum number of retries to prevent infinite loops
        retry_count = 0

        while retry_count <= max_retries:
            # Copy the timestamp data for modification in this retry iteration
            temp_timestamp_data = timestamp_data.copy()

            while temp_timestamp_data:
                # Unzip the data to get lists of timestamps and weights
                temp_ok_timestamps, temp_indices = zip(*temp_timestamp_data)

                # Select a timestamp based on weights
                selected_timestamp = None
                while not selected_timestamp:
                    selected_month = random.choices(counts_df['Month'], weights=probabilities, k=1)[0]
                    matched_dates = [date for date in temp_ok_timestamps if date.month == selected_month]
                    if matched_dates:
                        selected_timestamp = random.choice(matched_dates)

                # Find the index of the selected timestamp in temp_ok_timestamps
                idx_in_temp = temp_ok_timestamps.index(selected_timestamp)

                # Get the corresponding weight and index
                corresponding_index = temp_indices[idx_in_temp]

                # Proceed with downloading and processing
                eop = self.scl_download_task(bbox=self.bbox, time_interval=selected_timestamp)
                eop = self.scl_cloud_snow_task(eop)
                eop = self.input_task_all_bands(eopatch=deepcopy(eop))
                s2_cloud_array = self.cloud_detector.get_cloud_probability_maps(eop.data["BANDS"])
                cloud_probability = s2_cloud_array.mean()

                if cloud_probability < current_threshold:
                    eop.data["BANDS"] = eop.data['BANDS'][:, :, :, [1, 2, 3, 7, 4, 5, 6]]  # Reorder bands
                    eop.data["CLOUD_PROB"] = s2_cloud_array[..., np.newaxis]
                    eop.meta_info['cloud_prob_s2_cloudless'] = cloud_probability
                    eop.meta_info['snow_coverage'] = avg_snow_coverage[corresponding_index]
                    self._update_counts_csv(selected_timestamp.month, counts_df)
                    return eop  # Return the processed EOPatch for the selected date

                # If the selected timestamp does not meet the cloud threshold, remove it from the list
                temp_timestamp_data.pop(idx_in_temp)

            # If no suitable date is found, increase the cloudless threshold and retry
            retry_count += 1
            current_threshold += 0.05
            print(f"No suitable dates found with cloud threshold {current_threshold - 0.05}. Increasing threshold to {current_threshold} and retrying.")

        # After max retries, raise an error
        raise ValueError(f"No suitable dates found after increasing cloud threshold up to {current_threshold}.")

    def _read_counts_csv(self):
        """
        Reads the monthly counts from the CSV file. Initializes the file if it doesn't exist.

        Returns:
        - pandas.DataFrame: DataFrame with 'Month' and 'Count' columns.
        """
        if os.path.exists(self.csv_file_path):
            counts_df = pd.read_csv(self.csv_file_path)
            # Validate the CSV structure
            if not set(['Month', 'Count']).issubset(counts_df.columns):
                raise ValueError("CSV file must contain 'Month' and 'Count' columns.")
        else:
            # Initialize counts to zero for each month
            # print('Creating new CSV file')
            counts_df = pd.DataFrame({
                'Month': list(range(1, 13)),
                'Count': [1] * 12 # Initialize counts to 1 to avoid division by zero
            })
            counts_df.to_csv(self.csv_file_path, index=False)
        
        counts_df['Prob'] = 1 / counts_df['Count']
        counts_df['Prob'] = counts_df['Prob'].values/counts_df['Prob'].sum()
        
        return counts_df


    def _update_counts_csv(self, selected_month, counts_df):
        """
        Updates the counts CSV by incrementing the count for the selected month.

        Parameters:
        - selected_month (int): The month number (1-12) of the selected timestamp.
        - counts_df (pandas.DataFrame): DataFrame with current counts per month.
        """
        # Increment the count for the selected month
        # print('Incrementing count for month', counts_df)
        counts_df.loc[counts_df['Month'] == selected_month, 'Count'] += 1
        counts_df = counts_df[['Month', 'Count']]
        # print('Incremented count for month', counts_df)

        # Save the updated counts back to the CSV file
        counts_df.to_csv(self.csv_file_path, index=False)







class AddDummyData(EOTask):
    """
    An EOTask that adds a dummy data layer to an EOPatch. The data layer will be filled with zeros,
    with the shape derived from the 'BANDS' data layer of the EOPatch.
    """
    def __init__(self, layer_name):
        self.layer_name = layer_name

    def execute(self, eopatch):
        """
        Executes the task on the given EOPatch.

        Parameters:
        - eopatch (EOPatch): The input EOPatch to which the dummy data will be added.

        Returns:
        - EOPatch: The EOPatch with the added dummy data layer.
        """
        # Calculate the shape based on the 'BANDS' data layer
        if 'SCL_CLOUD' in eopatch.mask:
            shape = eopatch.mask['SCL_CLOUD'].shape
        else:
            raise ValueError("EOPatch does not contain a 'SCL_CLOUD' mask layer.")

        if 'BANDS' in eopatch.data:
            clouddtype = eopatch.data['BANDS'].dtype
        else:
            clouddtype = np.float32
            # raise ValueError("EOPatch does not contain a 'BANDS' data layer. \n\n\n eopatch:\n\n\n", eopatch)

        # Initialize the data layer with zeros
        dummy_data = np.zeros(shape, dtype=clouddtype)

        # Add the dummy data layer to the EOPatch
        eopatch.data[self.layer_name] = dummy_data
        eopatch.meta_info['cloud_prob_s2_cloudless'] = 0.0
        eopatch.meta_info['snow_coverage'] = 0.0

        return eopatch

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


class AddClimateZones(EOTask):
    def __init__(self, tiff_climate, layer_name='CLIMATE_ZONES'):
        self.layer_name = layer_name
        self.tiff_climate = tiff_climate

    def execute(self, eopatch):
        metadata = dict(eopatch.meta_info)
        metadata['bbox'] = str(eopatch.bbox)
        top_left, bot_right = calculate_coordinates(metadata)
        extracted_data = extract_subarray(self.tiff_climate, top_left, bot_right, output_npy=None)

        # Calculate zoom factors and rescale
        if extracted_data.size != 0:
            zoom_factor_y = 2248 / extracted_data.shape[0]
            zoom_factor_x = 2248 / extracted_data.shape[1]
            climate_zones = zoom(extracted_data, (zoom_factor_y, zoom_factor_x), order=0)
        else:
            climate_zones = np.ones((2248, 2248), dtype=np.uint8) * 255
        
        eopatch.data[self.layer_name] = climate_zones[np.newaxis, ..., np.newaxis]
        
        return eopatch




def calculate_bbox(data, size_x, size_y):
    meta = data['meta']
    centre_lat = meta['centre_lat']
    centre_lon = meta['centre_lon']
    crs = meta['crs']  # e.g., 'EPSG:32632'

    # Transform center coordinates from lat/lon to the image CRS
    transformer = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    center_x, center_y = transformer.transform(centre_lon, centre_lat)

    # Compute half extents
    resolution = S2_RESOLUTION  # meters per pixel
    half_extent_x = (size_x * resolution) / 2
    half_extent_y = (size_y * resolution) / 2

    # Compute bbox coordinates
    min_x = center_x - half_extent_x
    max_x = center_x + half_extent_x
    min_y = center_y - half_extent_y
    max_y = center_y + half_extent_y
    bbox_coordinates = ((min_x, min_y), (max_x, max_y))
    return BBox(bbox_coordinates, CRS(crs))

class FillEOPatchTask(EOTask):
    def __init__(self, ds_point):
        self.ds_point = ds_point

    def execute(self, eop):
        # Data
        eop.data['BANDS'] = ( self.ds_point['bands'].numpy().transpose(1, 2, 0)[np.newaxis, ...] - 1000 ) / 10000
        eop.data['CLOUD_PROB'] = self.ds_point['cloud_mask'].numpy()[np.newaxis, ..., np.newaxis]

        # Meta info
        eop.meta_info['maxcc'] = 0.1
        eop.meta_info['size_x'] = eop.data['BANDS'].shape[1]
        eop.meta_info['size_y'] = eop.data['BANDS'].shape[2]
        
        meta_list = []
        for key, value in self.ds_point['meta'].items():
            eop.meta_info[key] = str(value)
            meta_list.append(key)
        eop.meta_info['meta_list'] = meta_list

        if (eop.meta_info['size_x'] != 1068) or  (eop.meta_info['size_y'] != 1068):
            raise ValueError("The size of the image is not 1068x1068")

        # Set bbox and timestamp
        eop.bbox = calculate_bbox(self.ds_point, eop.meta_info['size_x'], eop.meta_info['size_y'])
        eop.timestamp = [pd.to_datetime(self.ds_point['meta']['timestamp']).to_pydatetime()]
        
        # Calculate min_lon and max_lat
        bbox = eop.bbox
        crs_proj = bbox.crs
        crs_geo = CRS.WGS84  # EPSG:4326
        min_x, max_y = bbox.min_x, bbox.max_y

        transformer = Transformer.from_crs(crs_proj.pyproj_crs(), crs_geo.pyproj_crs(), always_xy=True)
        min_lon, max_lat = transformer.transform(min_x, max_y)

        eop.meta_info['topleft_min_lon'] = min_lon
        eop.meta_info['topleft_max_lat'] = max_lat
        
        return eop


class ResampleZenithAngles(EOTask):
    def execute(self, eopatch):
        sun_zenith_angles = eopatch.data['sunZenithAngles']
        current_shape = sun_zenith_angles.shape[1:3]  # exclude the batch and channel dimensions
        target_shape = eopatch.data['BANDS'].shape[1:3]  # (1068, 1068)

        if current_shape != target_shape:
            scale_h = target_shape[0] / current_shape[0]
            scale_w = target_shape[1] / current_shape[1]

            resampled_data = zoom(sun_zenith_angles, (1, scale_h, scale_w, 1), order=1)  # using bilinear interpolation (order=1)
            eopatch.data['sunZenithAngles'] = resampled_data
        
        return eopatch
