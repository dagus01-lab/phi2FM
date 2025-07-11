""" Dataloader for images and labels. """
import os
from glob import glob
import json

import torch
import torch.nn.functional as F
import numpy as np
import buteo as beo


# lsof /dev/nvidia* | awk '{print $2}' | xargs -I {} kill {}


def encode_latitude(lat):
    """ Latitude goes from -90 to 90 """
    lat_adj = lat + 90.0
    lat_max = 180

    encoded_sin = ((np.sin(2 * np.pi * (lat_adj / lat_max)) + 1)) / 2.0
    encoded_cos = ((np.cos(2 * np.pi * (lat_adj / lat_max)) + 1)) / 2.0

    return np.array([encoded_sin, encoded_cos], dtype=np.float32)


def encode_longitude(lng):
    """ Longitude goes from -180 to 180 """
    lng_adj = lng + 180.0
    lng_max = 360

    encoded_sin = ((np.sin(2 * np.pi * (lng_adj / lng_max)) + 1)) / 2.0
    encoded_cos = ((np.cos(2 * np.pi * (lng_adj / lng_max)) + 1)) / 2.0

    return np.array([encoded_sin, encoded_cos], dtype=np.float32)


def read_raster(raster, filled=True, fill_value=0, pixel_offsets=None, channel_last=False, cast=np.float32):
    return beo.raster_to_array(beo.raster_open(raster, writeable=False), filled=filled, fill_value=fill_value, pixel_offsets=pixel_offsets, channel_last=channel_last, cast=cast)


def read_latlng_bounds(raster):
    metadata = beo.raster_to_metadata(beo.raster_open(raster, writeable=False))
    bounds_wgs84 = metadata["bbox_latlng"]
    lat, lng = metadata["centroid_latlng"]

    return lat, lng, bounds_wgs84


def create_reference(raster, pixel_offsets):
    return beo.array_to_raster(
        beo.raster_to_array(beo.raster_open(raster, writeable=False), pixel_offsets=pixel_offsets),
        reference=beo.raster_open(raster, writeable=False),
        pixel_offsets=pixel_offsets,
    )


def clip_and_read(raster, clip_geom, filled=True, fill_value=0, channel_last=False, cast=None):
    clipped = beo.raster_clip(beo.raster_open(raster, writeable=True), clip_geom=clip_geom)
    array = read_raster(clipped, filled=filled, fill_value=fill_value, channel_last=channel_last, cast=cast)
    beo.delete_dataset_if_in_memory(clipped)

    return array


def is_1_within_2(bbox1, bbox2):
    return beo.utils_bbox._check_bboxes_within(bbox1, bbox2)


def delete_memory_layer(layer):
    beo.delete_dataset_if_in_memory(layer)


def read_globals(label_path):
    paths = {
        "water": os.path.join(label_path, "water_4326.tif"),
        "terrain": os.path.join(label_path, "terrain_4326.tif"),
        "climate": os.path.join(label_path, "climate_4326.tif"),
        "landcover": os.path.join(label_path, "landcover_4326.tif"),
        "degurba": os.path.join(label_path, "degurba_4326.tif"),
        "buildings": os.path.join(label_path, "build_4326.tif"),
    }

    bboxes = {
        "water": beo.get_bbox_from_dataset(paths["water"]),
        "terrain": beo.get_bbox_from_dataset(paths["terrain"]),
        "climate": beo.get_bbox_from_dataset(paths["climate"]),
        "landcover": beo.get_bbox_from_dataset(paths["landcover"]),
        "degurba": beo.get_bbox_from_dataset(paths["degurba"]),
        "buildings": beo.get_bbox_from_dataset(paths["buildings"]),        
    }

    return paths, bboxes



class TinyTomDataset(torch.utils.data.Dataset):
    def __init__(self, tinyTOM_dataset_path, label_folder_path, patch_size=128, read_static_to_ram=False, device="cuda", transform=None):
        self.dataset_path = tinyTOM_dataset_path
        self.label_path = label_folder_path
        self.patch_size = patch_size
        self.read_static_to_ram = read_static_to_ram
        self.device = device
        self.transform = transform

        self.global_static_labels, self.global_static_bounds = read_globals(self.label_path)

        self.band_scales = {
            "10m": ["B02", "B03", "B04", "B08", "Cloud_mask"],
            "20m": ["B05", "B06", "B07", "B8A", "B11", "B12"],
            "60m": ["B01", "B09"],
        }

        self.band_sizes = {
            "10m": (1068, 1068),
            "20m": (534, 534),
            "60m": (178, 178),
        }

        self.images_large = []
        for f in glob(self.dataset_path + "/*/*/*/*/B01.tif"): # Row, Col, S2L2A, dataset, images.tif
            self.images_large.append({
                "B01": f,
                "B02": f.replace("B01.tif", "B02.tif"),
                "B03": f.replace("B01.tif", "B03.tif"),
                "B04": f.replace("B01.tif", "B04.tif"),
                "B05": f.replace("B01.tif", "B05.tif"),
                "B06": f.replace("B01.tif", "B06.tif"),
                "B07": f.replace("B01.tif", "B07.tif"),
                "B08": f.replace("B01.tif", "B08.tif"),
                "B8A": f.replace("B01.tif", "B8A.tif"),
                "B09": f.replace("B01.tif", "B09.tif"),
                "B11": f.replace("B01.tif", "B11.tif"),
                "B12": f.replace("B01.tif", "B12.tif"),
                "Cloud_mask": f.replace("B01.tif", "cloud_mask.tif"),
                "Thumbnail": f.replace("B01.tif", "thumbnail.jpg"),
            })

        self.landcover_map = np.zeros(101, dtype=np.uint8)
        self.landcover_map[0] = 0 # nodata
        self.landcover_map[10] = 1 # trees
        self.landcover_map[20] = 2 # shrubs
        self.landcover_map[30] = 3 # grass
        self.landcover_map[40] = 4 # crops
        self.landcover_map[50] = 5 # built
        self.landcover_map[60] = 6 # bare
        self.landcover_map[70] = 7 # snow
        self.landcover_map[80] = 8 # water
        self.landcover_map[90] = 9 # wetland
        self.landcover_map[95] = 10 # mangrove
        self.landcover_map[100] = 11 # snow

    def get_random_offset(self):
        pixel_20m = self.patch_size // 2
        height = np.random.randint(0, self.band_sizes["20m"][1] - pixel_20m + 1) * 2
        width = np.random.randint(0, self.band_sizes["20m"][1] - pixel_20m + 1) * 2

        return (width, height, self.patch_size, self.patch_size)


    def __len__(self):
        return len(self.images_large)


    def __getitem__(self, idx):
        paths = self.images_large[idx]
        bbox = self.get_random_offset()

        try:
            reference_raster = create_reference(paths["B05"], pixel_offsets=[v // 2 for v in bbox])
            lat, lng, bounds_wgs84 = read_latlng_bounds(reference_raster)

            bands = {}
            for band in self.band_scales["10m"]:
                bands[band] = read_raster(paths[band], pixel_offsets=bbox)

            for band in self.band_scales["20m"]:
                bands[band] = read_raster(paths[band], pixel_offsets=[v // 2 for v in bbox])

            for band in bands:
                bands[band] = torch.tensor(bands[band], dtype=torch.float32, device=self.device) / 10000.0

                if band in self.band_scales["20m"]:
                    bands[band] = F.interpolate(bands[band].unsqueeze(0), scale_factor=2, mode="bilinear").squeeze(0)

            clouds = torch.tensor([
                (bands["Cloud_mask"] == 0).sum() / bands["Cloud_mask"].numel(),
                (bands["Cloud_mask"] == 1).sum() / bands["Cloud_mask"].numel(),
                (bands["Cloud_mask"] == 2).sum() / bands["Cloud_mask"].numel(),
                (bands["Cloud_mask"] == 3).sum() / bands["Cloud_mask"].numel(),
            ], dtype=torch.float32, device=self.device)
            clouds_weight = torch.tensor(1.0, dtype=torch.float32, device=self.device)

            coords = torch.tensor(
                np.concatenate([encode_latitude(lat), encode_latitude(lng)]),
                dtype=torch.float32,
                device=self.device,
            )
            coords_weight = torch.tensor(1.0, dtype=torch.float32, device=self.device)

            if is_1_within_2(bounds_wgs84, self.global_static_bounds["water"]):
                water_array = clip_and_read(self.global_static_labels["water"], clip_geom=reference_raster)
                water_value = water_array.sum() / water_array.size
                water = torch.tensor(water_value, dtype=torch.float32, device=self.device)
                water_weight = torch.tensor(1.0, dtype=torch.float32, device=self.device)
            else:
                water = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                water_weight = torch.tensor(0.0, dtype=torch.float32, device=self.device)

            if is_1_within_2(bounds_wgs84, self.global_static_bounds["buildings"]):
                buildings_array = clip_and_read(self.global_static_labels["buildings"], clip_geom=reference_raster)
                buildings_value = (buildings_array.sum() / buildings_array.size) / 100.0
                buildings = torch.tensor(buildings_value, dtype=torch.float32, device=self.device)
                buildings_weight = torch.tensor(1.0, dtype=torch.float32, device=self.device)
            else:
                buildings = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                buildings_weight = torch.tensor(0.0, dtype=torch.float32, device=self.device)

            if is_1_within_2(bounds_wgs84, self.global_static_bounds["landcover"]):
                landcover_array = clip_and_read(self.global_static_labels["landcover"], clip_geom=reference_raster).flatten()
                landcover_non_zero = np.count_nonzero(landcover_array)
                map_landcover = np.vectorize(lambda x: self.landcover_map[x])

                if landcover_non_zero == 0.0:
                    landcover = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
                    landcover_weight = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                else:
                    mapped = map_landcover(landcover_array)
                    landcover_values = np.bincount(mapped, minlength=12)[1:] / landcover_non_zero
                    landcover = torch.tensor(landcover_values, dtype=torch.float32, device=self.device)
                    landcover_weight = torch.tensor(landcover_non_zero / landcover_array.size, dtype=torch.float32, device=self.device)
        
            else:
                landcover = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
                landcover_weight = torch.tensor(0.0, dtype=torch.float32, device=self.device)

            delete_memory_layer(reference_raster)

            x = torch.cat([
                bands["B02"],
                bands["B03"],
                bands["B04"],
                bands["B08"],
                bands["B05"],
                bands["B06"],
                bands["B07"],
                bands["B8A"],
                bands["B11"],
                bands["B12"],
            ], dim=0)

            label = {
                "coords": coords, "coords_weight": coords_weight,
                "clouds": clouds, "cloud_weight": clouds_weight,
                "water": water, "water_weight": water_weight,
                "buildings": buildings, "buildings_weight": buildings_weight,
                "landcover": landcover, "landcover_weight": landcover_weight,
            }

            if self.transform is not None:
                x = self.transform(x)

        except:
            new_idx = 0 if idx + 1 == len(self.images_large) else idx + 1
            return self.__getitem__(new_idx)

        return x, label



class MajorTomDataset(torch.utils.data.Dataset):
    def __init__(self, majorTOM_dataset1_path, majorTOM_dataset2_path, label_folder_path, patch_size=128, read_static_to_ram=False, device="cuda", transform=None, path_list="/home/phimultigpu/phileo-foundation/majorTOM_paths.json"):
        self.dataset1_path = majorTOM_dataset1_path
        self.dataset2_path = majorTOM_dataset2_path
        self.label_path = label_folder_path
        self.patch_size = patch_size
        self.read_static_to_ram = read_static_to_ram
        self.device = device
        self.transform = transform
        self.path_list = path_list

        self.global_static_labels, self.global_static_bounds = read_globals(self.label_path)

        self.band_scales = {
            "10m": ["B02", "B03", "B04", "B08", "Cloud_mask"],
            "20m": ["B05", "B06", "B07", "B8A", "B11", "B12"],
            "60m": ["B01", "B09", "B10"],
        }

        self.band_sizes = {
            "10m": (1068, 1068),
            "20m": (534, 534),
            "60m": (178, 178),
        }

        path_list = open(self.path_list, 'r')
        self.images_large = json.loads(path_list.readline())
        path_list = None

        # self.images_large = json.loads(self.path_list) if os.path.exists(self.path_list) else self.init_paths()          

        self.landcover_map = np.zeros(101, dtype=np.uint8)
        self.landcover_map[0] = 0 # nodata
        self.landcover_map[10] = 1 # trees
        self.landcover_map[20] = 2 # shrubs
        self.landcover_map[30] = 3 # grass
        self.landcover_map[40] = 4 # crops
        self.landcover_map[50] = 5 # built
        self.landcover_map[60] = 6 # bare
        self.landcover_map[70] = 7 # snow
        self.landcover_map[80] = 8 # water
        self.landcover_map[90] = 9 # wetland
        self.landcover_map[95] = 10 # mangrove
        self.landcover_map[100] = 11 # snow


    def init_paths(self):
        images_large = []
        for f in glob(self.dataset1_path + "/*/*/*/*/B01.tif") + glob(self.dataset2_path + "/*/*/*/*/B01.tif"): # Row, Col, S2L2A/S2L1C, dataset, images.tif
            images_large.append({
                "B01": f,
                "B02": f.replace("B01.tif", "B02.tif"),
                "B03": f.replace("B01.tif", "B03.tif"),
                "B04": f.replace("B01.tif", "B04.tif"),
                "B05": f.replace("B01.tif", "B05.tif"),
                "B06": f.replace("B01.tif", "B06.tif"),
                "B07": f.replace("B01.tif", "B07.tif"),
                "B08": f.replace("B01.tif", "B08.tif"),
                "B8A": f.replace("B01.tif", "B8A.tif"),
                "B09": f.replace("B01.tif", "B09.tif"),
                "B10": f.replace("B01.tif", "B10.tif"),
                "B11": f.replace("B01.tif", "B11.tif"),
                "B12": f.replace("B01.tif", "B12.tif"),
                "Cloud_mask": f.replace("B01.tif", "cloud_mask.tif"),
                "Thumbnail": f.replace("B01.tif", "thumbnail.jpg"),
                "Level": os.path.dirname(f).split(os.path.sep)[-2]
            })

        with open(self.path_list, "w") as file:
            file.write(json.dumps(images_large))

        return images_large


    def get_random_offset(self):
        pixel_20m = self.patch_size // 2
        height = np.random.randint(0, self.band_sizes["20m"][1] - pixel_20m + 1) * 2
        width = np.random.randint(0, self.band_sizes["20m"][1] - pixel_20m + 1) * 2

        return (width, height, self.patch_size, self.patch_size)


    def __len__(self):
        return len(self.images_large)


    def __getitem__(self, idx):
        paths = self.images_large[idx]
        bbox = self.get_random_offset()

        try:
            # Only bands 10 missing
            # level = paths["Level"]

            reference_raster = create_reference(paths["B05"], pixel_offsets=[v // 2 for v in bbox])
            lat, lng, bounds_wgs84 = read_latlng_bounds(reference_raster)

            bands = {}
            for band in self.band_scales["10m"]:
                bands[band] = read_raster(paths[band], pixel_offsets=bbox)

            for band in self.band_scales["20m"]:
                bands[band] = read_raster(paths[band], pixel_offsets=[v // 2 for v in bbox])

            for band in bands:
                bands[band] = torch.tensor(bands[band], dtype=torch.float32, device=self.device) / 10000.0

                if band in self.band_scales["20m"]:
                    bands[band] = F.interpolate(bands[band].unsqueeze(0), scale_factor=2, mode="bilinear").squeeze(0)

            # OBS change
            for band in self.band_scales["60m"]:
                bands[band] = torch.zeros_like(bands["B02"])

            clouds = torch.tensor([
                (bands["Cloud_mask"] == 0).sum() / bands["Cloud_mask"].numel(),
                (bands["Cloud_mask"] == 1).sum() / bands["Cloud_mask"].numel(),
                (bands["Cloud_mask"] == 2).sum() / bands["Cloud_mask"].numel(),
                (bands["Cloud_mask"] == 3).sum() / bands["Cloud_mask"].numel(),
            ], dtype=torch.float32, device=self.device)
            clouds_weight = torch.tensor(1.0, dtype=torch.float32, device=self.device)

            coords = torch.tensor(
                np.concatenate([encode_latitude(lat), encode_latitude(lng)]),
                dtype=torch.float32,
                device=self.device,
            )
            coords_weight = torch.tensor(1.0, dtype=torch.float32, device=self.device)

            if is_1_within_2(bounds_wgs84, self.global_static_bounds["water"]):
                water_array = clip_and_read(self.global_static_labels["water"], clip_geom=reference_raster)
                water_value = water_array.sum() / water_array.size
                water = torch.tensor(water_value, dtype=torch.float32, device=self.device)
                water_weight = torch.tensor(1.0, dtype=torch.float32, device=self.device)
            else:
                water = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                water_weight = torch.tensor(0.0, dtype=torch.float32, device=self.device)

            if is_1_within_2(bounds_wgs84, self.global_static_bounds["buildings"]):
                buildings_array = clip_and_read(self.global_static_labels["buildings"], clip_geom=reference_raster)
                buildings_value = (buildings_array.sum() / buildings_array.size) / 100.0
                buildings = torch.tensor(buildings_value, dtype=torch.float32, device=self.device)
                buildings_weight = torch.tensor(1.0, dtype=torch.float32, device=self.device)
            else:
                buildings = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                buildings_weight = torch.tensor(0.0, dtype=torch.float32, device=self.device)

            if is_1_within_2(bounds_wgs84, self.global_static_bounds["landcover"]):
                landcover_array = clip_and_read(self.global_static_labels["landcover"], clip_geom=reference_raster).flatten()
                landcover_non_zero = np.count_nonzero(landcover_array)
                map_landcover = np.vectorize(lambda x: self.landcover_map[x])

                if landcover_non_zero == 0.0:
                    landcover = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
                    landcover_weight = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                else:
                    mapped = map_landcover(landcover_array)
                    landcover_values = np.bincount(mapped, minlength=12)[1:] / landcover_non_zero
                    landcover = torch.tensor(landcover_values, dtype=torch.float32, device=self.device)
                    landcover_weight = torch.tensor(landcover_non_zero / landcover_array.size, dtype=torch.float32, device=self.device)
        
            else:
                landcover = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
                landcover_weight = torch.tensor(0.0, dtype=torch.float32, device=self.device)

            delete_memory_layer(reference_raster)

            x = torch.cat([
                # bands["B01"],
                bands["B02"],
                bands["B03"],
                bands["B04"],
                bands["B08"],
                bands["B05"],
                bands["B06"],
                bands["B07"],
                bands["B8A"],
                # bands["B09"],
                # bands["B10"],
                bands["B11"],
                bands["B12"],
            ], dim=0)

            label = {
                "coords": coords, "coords_weight": coords_weight,
                "clouds": clouds, "cloud_weight": clouds_weight,
                "water": water, "water_weight": water_weight,
                "buildings": buildings, "buildings_weight": buildings_weight,
                "landcover": landcover, "landcover_weight": landcover_weight,
            }

            if self.transform is not None:
                x = self.transform(x)

        except:
            new_idx = 0 if idx + 1 == len(self.images_large) else idx + 1
            return self.__getitem__(new_idx)

        return x, label



if __name__ == "__main__":
    dataset = TinyTomDataset("/home/phimultigpu/tinyTOM/dataset", "/home/phimultigpu/phileo-foundation/data_static")