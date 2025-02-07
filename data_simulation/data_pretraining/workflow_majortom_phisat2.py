# ----------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------

import datetime
import os
import re
import sys
sys.path.append('/home/ccollado/1_simulate_data/phisat_2')

os.chdir("/home/ccollado/1_simulate_data/phisat_2")

import matplotlib.pyplot as plt
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
)
from sentinelhub.exceptions import SHDeprecationWarning
from tqdm.auto import tqdm


# Import local modules

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
    SCLCloudSnowTask,
    BandMisalignmentTask,
    PhisatCalculationTask,
    AlternativePhisatCalculationTask,
    CropTask,
    GriddingTask,
    ExportGridToTiff,
    get_extent,
)
from utils import get_utm_bbox_from_top_left_and_size, AddLabelsTask, ExportEOPatchTask, PlotResultsTask, DownloadWOCloudSnow, AddDummyData, AddClimateZones

# ----------------------------------------------------------------

import warnings

warnings.filterwarnings("ignore", category=SHDeprecationWarning)



class phisat2simulation:
    def __init__(self, output_file_name, lat_topleft=None, lon_topleft=None, width_pixels=None, height_pixels=None, time_interval=None, 
                 roads_file=None, buildings_file=None,
                 maxcc=0.05, threshold_cloudless=0.05, threshold_snow = 0.4,
                 process_level='l1c',
                 folder_path_tifs='/home/ccollado/phileo_NFS/phileo_data/downstream/downstream_dataset_tifs', 
                 output_path_tifs='/home/ccollado/1_simulate_data/phisat_2/tiff_folder', 
                 exec_path="/home/ccollado/1_simulate_data/phisat_2/executables/phisat2_unix.bin",
                 sh_client_id="01b7f489-bc50-46fd-825b-460d078ca88a", 
                 sh_client_secret="cGGEjvkqmwzNhFFv9K0MGmdMhvZf1Nu6",
                 plot_results=False,
                 use_local_l1c=False,
                 ds_point=None
                 ):

        self.lat_topleft = lat_topleft
        self.lon_topleft = lon_topleft
        self.width_pixels = width_pixels
        self.height_pixels = height_pixels
        self.time_interval = time_interval
        self.roads_file = roads_file
        self.buildings_file = buildings_file
        self.output_file_name = output_file_name
        self.folder_path_tifs = folder_path_tifs
        self.output_path_tifs = output_path_tifs
        self.exec_path = exec_path
        self.maxcc = maxcc
        self.threshold_cloudless = threshold_cloudless
        self.threshold_snow = threshold_snow
        self.process_level = process_level
        self.sh_client_id = sh_client_id
        self.sh_client_secret = sh_client_secret
        self.plot_results = plot_results
        self.sh_config = self._set_sh_config()
        
        if lat_topleft and lon_topleft and width_pixels and height_pixels:
            self.bbox = self._get_bbox()
        else:
            self.bbox = None
        
        self.use_local_l1c = use_local_l1c
        self.ds_point = ds_point
        
        self.workflow, self.nodes, self.workflow_just_plot, self.nodes_just_plot, self.workflow_direct_download, self.nodes_direct_download, self.nodes_local_l1c, self.workflow_local_l1c = self._build_workflow()

    def _set_sh_config(self):
        sh_config = SHConfig()
        sh_config.sh_client_id = self.sh_client_id
        sh_config.sh_client_secret = self.sh_client_secret
        return sh_config

    def _get_bbox(self):
        return get_utm_bbox_from_top_left_and_size(self.lat_topleft, self.lon_topleft, self.width_pixels, self.height_pixels)

    def _build_workflow(self):

        aux_request_args = {"processing": {"upsampling": "BICUBIC"}}
        if self.process_level == 'L1C':
            PROCESSING_LEVEL = ProcessingLevels.L1C
        elif self.process_level == 'L1A':
            PROCESSING_LEVEL = ProcessingLevels.L1A
        elif self.process_level == 'L1B':
            PROCESSING_LEVEL = ProcessingLevels.L1B
        else:
            raise ValueError("Invalid processing level, choose between 'L1A', 'L1B', 'L1C'")
            

        # ----------------------------------------------------------------
        # 1. DATA DOWNLOAD
        # ----------------------------------------------------------------

        scl_download_task = SentinelHubInputTask(
            data_collection=DataCollection.SENTINEL2_L2A,
            resolution=S2_RESOLUTION,
            additional_data=[(FeatureType.MASK, "SCL")],
            maxcc=self.maxcc,
            aux_request_args=aux_request_args,
            config=self.sh_config,
            # cache_folder="./temp_data/",
            cache_folder=None,
            time_difference=datetime.timedelta(minutes=180),
        )

        scl_cloud_snow_task = SCLCloudSnowTask(scl_feature=(FeatureType.MASK, "SCL"))


        # ----------------------------------------------------------------
        # 3. FILTER BY CLOUDS & DOWNLOAD BANDS
        # ----------------------------------------------------------------
        additional_bands = [(FeatureType.DATA, name) for name in ["sunZenithAngles"]]
        masks = [(FeatureType.MASK, "dataMask")]
        
        input_task_all_bands = SentinelHubInputTask(
            data_collection=DataCollection.SENTINEL2_L1C,
            resolution=S2_RESOLUTION,
            bands_feature=(FeatureType.DATA, "BANDS"),
            additional_data=masks + additional_bands,
            # bands=S2_BANDS,  # note the order of these bands, where B08 follows B03
            aux_request_args=aux_request_args,
            config=self.sh_config,
            # cache_folder="./temp_data/",
            cache_folder=None,
            time_difference=datetime.timedelta(minutes=180),
        )
        
        input_task = SentinelHubInputTask(
            data_collection=DataCollection.SENTINEL2_L1C,
            resolution=S2_RESOLUTION,
            bands_feature=(FeatureType.DATA, "BANDS"),
            additional_data=masks + additional_bands,
            bands=S2_BANDS,  # note the order of these bands, where B08 follows B03
            aux_request_args=aux_request_args,
            config=self.sh_config,
            # cache_folder="./temp_data/",
            cache_folder=None,
            time_difference=datetime.timedelta(minutes=180),
        )
        
        cloud_prob_task = AddDummyData("CLOUD_PROB")

        cloud_and_download_bands_task = DownloadWOCloudSnow(threshold_cloudless = self.threshold_cloudless, threshold_snow = self.threshold_snow,
                                                         input_task_all_bands=input_task_all_bands, scl_download_task=scl_download_task,
                                                         scl_cloud_snow_task = scl_cloud_snow_task) # defaults for cloud_detector_config and csv_file_path


        # 1.1. OPTIONAL: FILTER BY VALID DATA

        def full_valid_data(array: np.array) -> bool:
            return np.mean(array) == 1.0

        filter_task = SimpleFilterTask((FeatureType.MASK, "dataMask"), full_valid_data)


        # ----------------------------------------------------------------
        # 2. OPTIONAL: PLOT RESULTS
        # ----------------------------------------------------------------

        plot_task = PlotResultsTask(self.plot_results)

        # ----------------------------------------------------------------
        # 2. METADATA DOWNLOAD
        # ----------------------------------------------------------------

        add_meta_task = AddMetadataTask(config=self.sh_config)

        # 2.1. LABELS DOWNLOAD
        labels_task = AddLabelsTask(self.roads_file, self.buildings_file, self.folder_path_tifs,
                                    self.sh_config)
        
        # ----------------------------------------------------------------
        # 2. LOCAL L1C DOWNLOAD
        # ----------------------------------------------------------------

        # ----------------------------------------------------------------
        # 3. COMPUTE RADIANCES
        # ----------------------------------------------------------------

        radiance_task = CalculateRadianceTask(
            (FeatureType.DATA, "BANDS"), (FeatureType.DATA, "BANDS-RAD")
        )

        # ----------------------------------------------------------------
        # 4. ADD PANCHROMATIC BAND
        # ----------------------------------------------------------------

        add_pan_task = AddPANBandTask(
            (FeatureType.DATA, "BANDS-RAD"), (FeatureType.DATA, "BANDS-RAD-PAN")
        )

        # ----------------------------------------------------------------
        # 5. SPATIAL RESAMPLING
        # ----------------------------------------------------------------

        # Define the features to resize
        if self.use_local_l1c:
            features_to_resize = {
                FeatureType.DATA: ["BANDS-RAD-PAN", "sunZenithAngles", "CLOUD_PROB"],
            }
            
        else:
            features_to_resize = {
                FeatureType.DATA: ["BANDS-RAD-PAN", "sunZenithAngles", "WORLD_COVER", "CLOUD_PROB"],
                FeatureType.MASK: [
                    "SCL_CLOUD",
                    "SCL_CIRRUS",
                    "SCL_CLOUD_SHADOW",
                    "dataMask",
                ],
            }

            # Conditionally add "ROADS" to the FeatureType.DATA list
            if self.roads_file:
                features_to_resize[FeatureType.DATA].append("ROADS")
            
            if self.buildings_file:
                features_to_resize[FeatureType.DATA].append("BUILDINGS")


        NEW_SIZE = tuple(int(dim * S2_RESOLUTION / PHISAT2_RESOLUTION) for dim in (self.height_pixels, self.width_pixels))
        # print(f"New size: {NEW_SIZE} -- Original size: {self.height_pixels, self.width_pixels}")

        # covert dataMask to uint8
        casting_task = MapFeatureTask(
            (FeatureType.MASK, "dataMask"), (FeatureType.MASK, "dataMask"), np.uint8
        )

        resize_task_list = []

        for feature_type in (features_to_resize.keys()):
            for feature in (features_to_resize[feature_type]):
                resize_task_list.append(
                    MapFeatureTask(
                        (feature_type, feature),
                        (feature_type, f"{feature}_RES"),
                        resize_images,
                        new_size=NEW_SIZE,
                        resize_method="nearest",
                    )
                )

        # remove the unnecessary features

        # Create the list of features to remove
        if self.use_local_l1c:
            features_to_remove = [
                (FeatureType.DATA, "BANDS"),
                (FeatureType.DATA, "BANDS-RAD"),
                (FeatureType.DATA, "BANDS-RAD-PAN"),
                (FeatureType.DATA, "sunZenithAngles"),
                (FeatureType.DATA, "CLOUD_PROB"),
            ]
            
        else:
            features_to_remove = [
                (FeatureType.DATA, "BANDS"),
                (FeatureType.DATA, "BANDS-RAD"),
                (FeatureType.DATA, "BANDS-RAD-PAN"),
                (FeatureType.DATA, "sunZenithAngles"),
                (FeatureType.DATA, "WORLD_COVER"),
                (FeatureType.DATA, "CLOUD_PROB"),
                (FeatureType.MASK, "SCL_CLOUD"),
                (FeatureType.MASK, "SCL_CLOUD_SHADOW"),
                (FeatureType.MASK, "SCL_CIRRUS"),
                (FeatureType.MASK, "dataMask"),
            ]

            # Conditionally add "ROADS" to the list
            if self.roads_file:
                features_to_remove.append((FeatureType.DATA, "ROADS"))
            
            if self.buildings_file:
                features_to_remove.append((FeatureType.DATA, "BUILDINGS"))

        # Create the RemoveFeatureTask with the updated list
        remove_feature_task1 = RemoveFeatureTask(features_to_remove)


        # ----------------------------------------------------------------
        # 6. BAND MISALIGNMENT
        # ----------------------------------------------------------------

        band_shift_task = BandMisalignmentTask(
            (FeatureType.DATA, "BANDS-RAD-PAN_RES"),
            (FeatureType.DATA, "BANDS-RAD-PAN_SHIFTED"),
            PROCESSING_LEVEL,
            std_sea=6,
            interpolation_method=cv2.INTER_NEAREST,
        )

        # remove the unnecessary features
        remove_feature_task2 = RemoveFeatureTask([(FeatureType.DATA, "BANDS-RAD-PAN_RES")])

        # Crop the image to contain only valid data after band misalignment

        # Create the list of features to crop
        if self.use_local_l1c:
            features_to_crop = [
                (FeatureType.DATA, "BANDS-RAD-PAN_SHIFTED"),
                (FeatureType.DATA, "sunZenithAngles_RES"),
                (FeatureType.DATA, "CLOUD_PROB_RES"),
            ]

        else:
            features_to_crop = [
                (FeatureType.DATA, "BANDS-RAD-PAN_SHIFTED"),
                (FeatureType.DATA, "sunZenithAngles_RES"),
                (FeatureType.DATA, "WORLD_COVER_RES"),
                (FeatureType.DATA, "CLOUD_PROB_RES"),
                (FeatureType.MASK, "SCL_CLOUD_RES"),
                (FeatureType.MASK, "SCL_CLOUD_SHADOW_RES"),
                (FeatureType.MASK, "SCL_CIRRUS_RES"),
                (FeatureType.MASK, "dataMask_RES"),
            ]

            # Conditionally add "ROADS_RES" to the list
            if self.roads_file:
                features_to_crop.append((FeatureType.DATA, "ROADS_RES"))
            
            if self.buildings_file:
                features_to_crop.append((FeatureType.DATA, "BUILDINGS_RES"))

        # Create the CropTask with the updated list
        crop_task = CropTask(features_to_crop=features_to_crop)

        # ----------------------------------------------------------------
        # 7. NOISE CALCULATION
        # ----------------------------------------------------------------

        '''
        A compiled executable will perform the SNR and PSF calculation. 

        Executables for the following Operating Systems (OSs) have been compiled:

        * Unix, `phisat2_unix.bin`
        * Windows, `phisat2_win.bin`
        * MacOS, `phisat2_osx-arm64.bin` for ARM chips, `phisat2_osx-x86_64.bin` for Intel chips
        
        Download the suitable binary for your OS from this link https://cloud.sinergise.com/s/g88Ns32rXB3AT6i.

        Once downloaded, allow the operating system to run the file and make it executable (e.g. `chmod a+x phisat2_unix.bin`). Then set the path to the executable in the cell below.
        '''

        snr_task = PhisatCalculationTask(
            input_feature=(FeatureType.DATA, "BANDS-RAD-PAN_SHIFTED"),
            output_feature=(FeatureType.DATA, "L_out_RES"),
            executable=self.exec_path,
            calculation="SNR",
        )

        # ----------------------------------------------------------------
        # 8. SIMULATE POINT SPREAD FUNCTION (PSF) FILTERING
        # ----------------------------------------------------------------

        psf_filter_task = PhisatCalculationTask(
            input_feature=(FeatureType.DATA, "L_out_RES"),
            output_feature=(FeatureType.DATA, "L_out_PSF"),
            executable=self.exec_path,
            calculation="PSF",
        )

        # ----------------------------------------------------------------
        # 9. ALTERNATIVE NOISE AND PSF CALCULATION
        # ----------------------------------------------------------------

        kernel_bands = ["B1", "B2", "B3", "B0", "B7", "B4", "B5", "B6"]
        psf_kernel = { band: np.random.random(size=(7,7)) for band in kernel_bands }

        snr_bands = ["B02", "B03", "B04", "PAN", "B08", "B05", "B06", "B07"]
        snr_values = { band: np.random.randint(20,250) for band in snr_bands }

        snr_psf_task = AlternativePhisatCalculationTask(
            input_feature=(FeatureType.DATA, "BANDS-RAD-PAN_SHIFTED"),
            snr_feature=(FeatureType.DATA, "L_out_RES"),
            psf_feature=(FeatureType.DATA, "L_out_PSF"),
            snr_values=snr_values,
            psf_kernel=psf_kernel,
            l_ref=100
        )

        # uncomment to run
        # eop = snr_psf_task(eop)


        # ----------------------------------------------------------------
        # 10. COMPUTE L1C
        # ----------------------------------------------------------------

        reflectance_task = CalculateReflectanceTask(
            input_feature=(FeatureType.DATA, "L_out_PSF"),
            output_feature=(FeatureType.DATA, "PHISAT2-BANDS"),
            processing_level=PROCESSING_LEVEL,
        )

        # remove the unnecessary features

        remove_feature_task3 = RemoveFeatureTask(
            [
                (FeatureType.DATA, "BANDS-RAD-PAN_SHIFTED"),
                (FeatureType.DATA, "L_out_PSF"),
                (FeatureType.DATA, "L_out_RES"),
            ]
        )
        
        # ----------------------------------------------------------------
        # 11. ADD CLIMATE ZONES
        # ----------------------------------------------------------------
        climate_zones_task = AddClimateZones()

        # ----------------------------------------------------------------
        # 11. SAVE IMAGE AS TIFF
        # ----------------------------------------------------------------

        export_task = ExportEOPatchTask(folder_path=self.output_path_tifs, filename=self.output_file_name,
                                        lat_topleft=self.lat_topleft, lon_topleft=self.lon_topleft,
                                        use_local_l1c=self.use_local_l1c)
        
        
        
        # ----------------------------------------------------------------
        # SPECIFICS TO LOCAL L1C
        # ----------------------------------------------------------------
        from scipy.ndimage import zoom
        from pyproj import Transformer
        import pandas as pd
        from sentinelhub import CRS

        class CreateEOPatchTask(EOTask):
            def execute(self):
                return EOPatch()

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
        
        ds_point = self.ds_point

        class FillEOPatchTask(EOTask):
            def __init__(self):
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

        create_eopatch_task = CreateEOPatchTask()
        fill_eopatch_task = FillEOPatchTask()

        sunZenithAngles = [(FeatureType.DATA, name) for name in ["sunZenithAngles"]]
        aux_request_args = {"processing": {"upsampling": "BICUBIC"}}

        input_task_zenith = SentinelHubInputTask(
            data_collection=DataCollection.SENTINEL2_L1C,
            resolution=S2_RESOLUTION,
            bands_feature=None,  # Do not request any bands
            bands=[],            # Ensure no bands are specified
            additional_data=sunZenithAngles,
            aux_request_args=aux_request_args,
            config=self.sh_config,
            # cache_folder="./temp_data/",
            time_difference=datetime.timedelta(minutes=180),
        )

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

        resample_zenith_angles = ResampleZenithAngles()


        # ----------------------------------------------------------------
        # 12. CREATE WORKFLOW
        # ----------------------------------------------------------------

        nodes = linearly_connect_tasks(
            scl_download_task,
            scl_cloud_snow_task,
            cloud_and_download_bands_task,
            # filter_task,  # Uncomment this if you want to filter by 100% of valid data
            plot_task,
            add_meta_task,
            labels_task,
            radiance_task,
            add_pan_task,
            casting_task,
            *resize_task_list,
            remove_feature_task1,
            band_shift_task,
            remove_feature_task2,
            crop_task,
            snr_task,
            psf_filter_task,
            reflectance_task,
            remove_feature_task3,
            export_task,
        )
        workflow = EOWorkflow(nodes)

        nodes_just_plot = linearly_connect_tasks(
            scl_download_task,
            scl_cloud_snow_task,
            cloud_and_download_bands_task,
            # filter_task,  # Uncomment this if you want to filter by 100% of valid data
            plot_task,
        )

        workflow_just_plot = EOWorkflow(nodes_just_plot)

        nodes_direct_download = linearly_connect_tasks(
            scl_download_task,
            scl_cloud_snow_task,
            # cloud_and_download_bands_task,
            cloud_prob_task,
            input_task,
            # filter_task,  # Uncomment this if you want to filter by 100% of valid data
            plot_task,
            add_meta_task,
            labels_task,
            radiance_task,
            add_pan_task,
            casting_task,
            *resize_task_list,
            remove_feature_task1,
            band_shift_task,
            remove_feature_task2,
            crop_task,
            snr_task,
            psf_filter_task,
            reflectance_task,
            remove_feature_task3,
            export_task,
        )

        workflow_direct_download = EOWorkflow(nodes_direct_download)

        nodes_local_l1c = linearly_connect_tasks(
            create_eopatch_task,
            fill_eopatch_task,
            input_task_zenith,
            resample_zenith_angles,
            add_meta_task,
            radiance_task,
            add_pan_task,
            # casting_task,
            *resize_task_list,
            remove_feature_task1,
            band_shift_task,
            remove_feature_task2,
            crop_task,
            snr_task,
            psf_filter_task,
            reflectance_task,
            remove_feature_task3,
            climate_zones_task,
            export_task,
        )
        
        workflow_local_l1c = EOWorkflow(nodes_local_l1c)

        
        
        return workflow, nodes, workflow_just_plot, nodes_just_plot, workflow_direct_download, nodes_direct_download, nodes_local_l1c, workflow_local_l1c

    def run(self):
        if self.use_local_l1c:
            results = self.workflow_local_l1c.execute()
        else:
            results = self.workflow.execute({self.nodes[0]: {"bbox": self.bbox, "time_interval": self.time_interval}})
        # print("Execution finished successfully")
        return results

    def plot(self):
        results = self.workflow_just_plot.execute({self.nodes_just_plot[0]: {"bbox": self.bbox, "time_interval": self.time_interval}})
        return results
    
    def direct_download(self):
        results = self.workflow_direct_download.execute({self.nodes_direct_download[0]: {"bbox": self.bbox, "time_interval": self.time_interval}})
        return results

