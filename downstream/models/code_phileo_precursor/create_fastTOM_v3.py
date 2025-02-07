""" Dataloader for images and labels. """
import os
import json

import numpy as np  
import buteo as beo
from osgeo import gdal; gdal.PushErrorHandler('CPLQuietErrorHandler')
import geopandas as gpd
from shapely.geometry import Point
import warnings


def numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [numpy_to_list(o) for o in obj]
    elif isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, np.generic):
        return obj.item()
    elif hasattr(obj, "item"):
        return obj.item()

    return obj


def process_image(image_path, target_folder="/home/phimultigpu/fastTOM/dataset/", static_folder="/home/phimultigpu/phileo-foundation/data_static/", admin_zones=""):
    static_landcover = os.path.join(static_folder, "landcover_4326.tif")
    static_degurba = os.path.join(static_folder, "degurba_4326.tif")
    static_climate = os.path.join(static_folder, "climate_4326.tif")
    static_terrain = os.path.join(static_folder, "terrain_4326.tif")
    static_buildings = os.path.join(static_folder, "build_4326.tif")
    static_water = os.path.join(static_folder, "water_4326.tif")

    metadata_dict = {}

    offsets_10m = [
        [0, 0, 516, 516],
        [0, 516, 516, 516],
        [516, 0, 516, 516],
        [516, 516, 516, 516],
    ]

    offsets_20m = [
        [0, 0, 258, 258],
        [0, 258, 258, 258],
        [258, 0, 258, 258],
        [258, 258, 258, 258],
    ]

    offsets_60m = [
        [0, 0, 86, 86],
        [0, 86, 86, 86],
        [86, 0, 86, 86],
        [86, 86, 86, 86],
    ]

    bands_10m = ["B02", "B03", "B04", "B08"]
    bands_20m = ["B05", "B06", "B07", "B8A", "B11", "B12"]
    bands_60m = ["B01", "B09", "B10"]

    path_split = image_path["B01"].split(os.path.sep)
    grid_cell = path_split[-4]
    image_name = path_split[-2]

    dst_folder = f"{target_folder}{grid_cell}/{image_name}/"

    if os.path.exists(dst_folder):
        return False

    os.makedirs(dst_folder)
 
    cloud_mask_path = image_path["Cloud_mask"]
    offset_idx = 0
    cloud_free_percent = 0.0
    try:
        for idx, o in enumerate(offsets_10m):
            cloud_free_pixels = beo.raster_to_array(cloud_mask_path, pixel_offsets=o, filled=True, fill_value=0) == 0
            cloud_free_percent_offset = (cloud_free_pixels).sum() / cloud_free_pixels.size
            if cloud_free_percent_offset >= cloud_free_percent:
                offset_idx = idx
                cloud_free_percent = cloud_free_percent_offset
    except:
        os.removedirs(dst_folder)
        return False

    if cloud_free_percent < 0.33:
        os.removedirs(dst_folder)
        return False

    offset_10m = offsets_10m[offset_idx]
    offset_20m = offsets_20m[offset_idx]
    offset_60m = offsets_60m[offset_idx]

    try:
        reference_raster = beo.array_to_raster(beo.raster_to_array(cloud_mask_path, pixel_offsets=offset_10m), reference=cloud_mask_path, pixel_offsets=offset_10m)
        water_clipped = beo.raster_clip(static_water, reference_raster)

        water_arr = beo.raster_to_array(water_clipped, filled=True, fill_value=1)
        water_percent = water_arr.sum() / water_arr.size

        if water_percent > 0.5:
            os.removedirs(dst_folder)
            return False
        
        metadata_raster = beo.raster_to_metadata(reference_raster)
        metadata_dict["projection_wkt"] = metadata_raster["projection_wkt"]
        metadata_dict["shape"] = metadata_raster["shape"]
        metadata_dict["level"] = image_path["Level"]
        metadata_dict["bbox"] = metadata_raster["bbox"]
        metadata_dict["bbox_latlng"] = metadata_raster["bbox_latlng"]
        metadata_dict["centroid"] = metadata_raster["centroid"]
        metadata_dict["centroid_latlng"] = metadata_raster["centroid_latlng"]
        metadata_dict["grid_cell"] = grid_cell
        metadata_dict["image_name"] = image_name
        metadata_dict["latitude"] = metadata_dict["centroid_latlng"][0]
        metadata_dict["longitude"] = metadata_dict["centroid_latlng"][1]

        point = Point(metadata_dict["centroid_latlng"][1], metadata_dict["centroid_latlng"][0])

        try:
            adm = geo_adm[geo_adm.intersects(point)].iloc[0]
        except:
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=UserWarning)
                adm = geo_adm.iloc[geo_adm.distance(point).argsort()[0]]

        metadata_dict["adm_country"] = adm["NAME"]
        metadata_dict["adm_iso"]  = adm["ISO_A3"]
        metadata_dict["adm_continent"]  = adm["CONTINENT"]
        metadata_dict["adm_region"]  = adm["SUBREGION"]

        reference_raster_4326 = beo.raster_reproject(reference_raster, static_landcover)

        landcover_map = np.zeros(101, dtype=np.uint8)
        landcover_map[0] = 0 # nodata
        landcover_map[10] = 1 # trees
        landcover_map[20] = 2 # shrubs
        landcover_map[30] = 3 # grass
        landcover_map[40] = 4 # crops
        landcover_map[50] = 5 # built
        landcover_map[60] = 6 # bare
        landcover_map[70] = 7 # snow
        landcover_map[80] = 8 # water
        landcover_map[90] = 9 # wetland
        landcover_map[95] = 10 # mangrove
        landcover_map[100] = 11 # snow
        vectorized_landcover_map = np.vectorize(lambda x: landcover_map[x])

        static_landcover_clip = beo.raster_clip(static_landcover, clip_geom=reference_raster_4326, all_touch=True)
        static_landcover_aligned = beo.raster_align(static_landcover_clip, reference=reference_raster)[0]
        static_landcover_arr = vectorized_landcover_map(beo.raster_to_array(static_landcover_aligned, filled=True, fill_value=0))
        beo.array_to_raster(static_landcover_arr, reference=static_landcover_aligned, out_path=dst_folder + "static_landcover.tif")

        static_buildings_clip = beo.raster_clip(static_buildings, clip_geom=reference_raster_4326, all_touch=True)
        static_buildings_aligned = beo.raster_align(static_buildings_clip, out_path=dst_folder + "static_buildings.tif", reference=reference_raster)[0]
        static_buildings_arr = beo.raster_to_array(static_buildings_aligned, filled=True, fill_value=0)

        static_climate_clip = beo.raster_clip(static_climate, clip_geom=reference_raster_4326, all_touch=True)
        static_climate_aligned = beo.raster_align(static_climate_clip, out_path=dst_folder + "static_climate.tif", reference=reference_raster)[0]
        static_climate_arr = beo.raster_to_array(static_climate_aligned, filled=True, fill_value=0)

        degurba_map = np.zeros(31, dtype=np.uint8)
        degurba_map[0] = 0 # nodata
        degurba_map[11] = 1 # Very Low Density Rural
        degurba_map[12] = 2 # Low Density Rural
        degurba_map[13] = 3 # Rural Cluster
        degurba_map[21] = 4 # Suburban Or Peri-Urban
        degurba_map[22] = 5 # Semi-Dense Urban Cluster
        degurba_map[23] = 6 # Dense Urban Cluster
        degurba_map[30] = 7 # Urban Centre
        vectorized_degurba_map = np.vectorize(lambda x: degurba_map[x])

        static_degurba_clip = beo.raster_clip(static_degurba, clip_geom=reference_raster_4326, all_touch=True)
        static_degurba_clip_aligned = beo.raster_align(static_degurba_clip, reference=reference_raster)[0]
        static_degurba_clip_arr = vectorized_degurba_map(beo.raster_to_array(static_degurba_clip_aligned, filled=True, fill_value=0))
        beo.array_to_raster(static_degurba_clip_arr, reference=static_degurba_clip_aligned, out_path=dst_folder + "static_degurba.tif")

        static_terrain_clip = beo.raster_clip(static_terrain, clip_geom=reference_raster_4326, all_touch=True)
        static_terrain_aligned = beo.raster_align(static_terrain_clip, out_path=dst_folder + "static_terrain.tif", reference=reference_raster)[0]
        static_terrain_arr = beo.raster_to_array(static_terrain_aligned, filled=True, fill_value=0)

        landcover_non_zero = np.count_nonzero(static_landcover_arr)
        landcover_weight = landcover_non_zero / (static_landcover_arr.shape[0] * static_landcover_arr.shape[1])
        landcover_proportions = np.bincount(static_landcover_arr.ravel(), minlength=12) / landcover_non_zero

        buildings_weight = 1.0
        buildings_proportions = (static_buildings_arr / 100.0).sum() / (static_buildings_arr.shape[0] * static_buildings_arr.shape[1])

        climate_non_zero = np.count_nonzero(static_climate_arr)
        climate_weight = climate_non_zero / (static_climate_arr.shape[0] * static_climate_arr.shape[1])
        climate_proportions = np.bincount(static_climate_arr.ravel(), minlength=31) / climate_non_zero

        degurba_non_zero = np.count_nonzero(static_degurba_clip_arr)
        degurba_weight = degurba_non_zero / (static_degurba_clip_arr.shape[0] * static_degurba_clip_arr.shape[1])
        degurba_proportions = np.bincount(static_degurba_clip_arr.ravel(), minlength=8) / degurba_non_zero

        terrain_non_zero = np.count_nonzero(static_terrain_arr)
        terrain_weight = terrain_non_zero / (static_terrain_arr.shape[0] * static_terrain_arr.shape[1])
        terrain_proportions = np.bincount(static_terrain_arr.ravel(), minlength=23) / terrain_non_zero

        metadata_dict["label_landcover"] = landcover_proportions
        metadata_dict["label_landcover_weight"] = landcover_weight

        metadata_dict["label_buildings"] = buildings_proportions
        metadata_dict["label_buildings_weight"] = buildings_weight

        metadata_dict["label_climate"] = climate_proportions
        metadata_dict["label_climate_weight"] = climate_weight

        metadata_dict["label_degurba"] = degurba_proportions
        metadata_dict["label_degurba_weight"] = degurba_weight

        metadata_dict["label_terrain"] = terrain_proportions
        metadata_dict["label_terrain_weight"] = terrain_weight

        beo.delete_dataset_if_in_memory_list([
            reference_raster,
            reference_raster_4326,
            water_clipped,
            static_landcover_clip,
            static_landcover_aligned,
            static_buildings_clip,
            static_climate_clip,
            static_degurba_clip,
            static_degurba_clip_aligned,
            static_terrain_clip,
        ])

    except:
        os.removedirs(dst_folder)
        return False

    for band in ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]:
        path = image_path[band]

        if image_path["Level"] == "S2L2A" and band == "B10":
            continue

        if not os.path.exists(path):
            return False

        if band in bands_10m:
            band_offset = offset_10m
        elif band in bands_20m:
            band_offset = offset_20m
        elif band in bands_60m:
            band_offset = offset_60m
        else:
            print("WARNING.")

        arr = beo.raster_to_array(path, filled=True, fill_value=0, pixel_offsets=band_offset)
        beo.array_to_raster(arr, reference=path, out_path=f"{dst_folder}/{band}.tif", pixel_offsets=band_offset)

    arr = beo.raster_to_array(cloud_mask_path, filled=True, fill_value=4, pixel_offsets=band_offset)
    beo.array_to_raster(arr, reference=path, out_path=f"{dst_folder}/cloud_mask.tif", pixel_offsets=band_offset)

    metadata_dict = {k: numpy_to_list(v) for k, v in metadata_dict.items()}

    with open(os.path.join(dst_folder, "metadata.json"), "w") as f:
        json_str = json.dumps(metadata_dict, indent=4, allow_nan=False)
        f.write(json_str)

    return True


if __name__ == "__main__":
    from multiprocessing import Pool
    from tqdm import tqdm

    target_folder = "/home/phimultigpu/fastTOM/dataset/"
    static_folder = "/home/phimultigpu/fastTOM/data_static/"
    file_paths = "/home/phimultigpu/fastTOM/majorTOM_paths.json"
    target_folder="/home/phimultigpu/fastTOM/dataset/"

    paths_list = json.loads(open(file_paths, 'r').readline())
    geo_adm = gpd.read_file(os.path.join(static_folder, "admin_wb_4326.gpkg"))

    pool = Pool(processes=24)
    bar = tqdm(total=len(paths_list), desc="Processing labels")


    for i in reversed(range(len(paths_list))):
        path_split = paths_list[i]["B01"].split(os.path.sep)
        grid_cell = path_split[-4]
        image_name = path_split[-2]

        b1_path = f"{target_folder}{grid_cell}/{image_name}/B01.tif"

        if os.path.exists(b1_path):
            continue

        pool.apply_async(process_image, args=(paths_list[i], target_folder, static_folder), callback=lambda x: bar.update())

    pool.close()
    pool.join()
