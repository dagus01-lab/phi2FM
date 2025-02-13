import datetime

import cv2

from eolearn.core import (
    EOTask, 
    EOPatch,
    EOWorkflow,
    FeatureType,
    MapFeatureTask,
    RemoveFeatureTask,
    linearly_connect_tasks,
)
from eolearn.io import SentinelHubInputTask
from eolearn.features.utils import spatially_resize_image as resize_images
from sentinelhub import DataCollection
from sentinelhub.exceptions import SHDeprecationWarning

# Import local modules
from data_simulation import sh_config

from data_simulation.simulator.phisat2_constants import (
    S2_RESOLUTION,
    PHISAT2_RESOLUTION,
    ProcessingLevels,
)
from data_simulation.simulator.phisat2_utils import (
    AddPANBandTask,
    AddMetadataTask,
    CalculateRadianceTask,
    CalculateReflectanceTask,
    BandMisalignmentTask,
    PhisatCalculationTask,
    CropTask,
)

from data_simulation.simulator.utils import get_utm_bbox_from_top_left_and_size, ExportEOPatchTask, AddClimateZones, FillEOPatchTask, ResampleZenithAngles

# ----------------------------------------------------------------

import warnings

warnings.filterwarnings("ignore", category=SHDeprecationWarning)


TIFF_CLIMATE_PATH = "/home/ccollado/2_phileo_fm/pretrain/foundation_uniphi/climate_4326.tif"


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
        
        if lat_topleft and lon_topleft and width_pixels and height_pixels:
            self.bbox = self._get_bbox()
        else:
            self.bbox = None
        
        self.use_local_l1c = use_local_l1c
        self.ds_point = ds_point
        
        self.workflow, self.nodes = self._build_workflow()

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
        # 0. CREATE AND FILL EOPATCH
        # ----------------------------------------------------------------
        class CreateEOPatchTask(EOTask):
            def execute(self):
                return EOPatch()
        
        create_eopatch_task = CreateEOPatchTask()
        fill_eopatch_task = FillEOPatchTask(ds_point=self.ds_point)


        # ----------------------------------------------------------------
        # 1. GET AND RESAMPLE SUN ZENITH ANGLES
        # ----------------------------------------------------------------

        sunZenithAngles = [(FeatureType.DATA, name) for name in ["sunZenithAngles"]]
        aux_request_args = {"processing": {"upsampling": "BICUBIC"}}

        input_task_zenith = SentinelHubInputTask(
            data_collection=DataCollection.SENTINEL2_L1C,
            resolution=S2_RESOLUTION,
            bands_feature=None,  # Do not request any bands
            bands=[],            # Ensure no bands are specified
            additional_data=sunZenithAngles,
            aux_request_args=aux_request_args,
            config=sh_config,
            # cache_folder="./temp_data/",
            time_difference=datetime.timedelta(minutes=180),
        )

        resample_zenith_angles = ResampleZenithAngles()


        # ----------------------------------------------------------------
        # 2. METADATA DOWNLOAD
        # ----------------------------------------------------------------
        add_meta_task = AddMetadataTask(config=sh_config)
        
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
        features_to_resize = {
            FeatureType.DATA: ["BANDS-RAD-PAN", "sunZenithAngles", "CLOUD_PROB"],
        }
        NEW_SIZE = tuple(int(dim * S2_RESOLUTION / PHISAT2_RESOLUTION) for dim in (self.height_pixels, self.width_pixels))

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

        # Remove the unnecessary features
        features_to_remove = [
            (FeatureType.DATA, "BANDS"),
            (FeatureType.DATA, "BANDS-RAD"),
            (FeatureType.DATA, "BANDS-RAD-PAN"),
            (FeatureType.DATA, "sunZenithAngles"),
            (FeatureType.DATA, "CLOUD_PROB"),
        ]

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
        features_to_crop = [
            (FeatureType.DATA, "BANDS-RAD-PAN_SHIFTED"),
            (FeatureType.DATA, "sunZenithAngles_RES"),
            (FeatureType.DATA, "CLOUD_PROB_RES"),
        ]

        # Create the CropTask with the updated list
        crop_task = CropTask(features_to_crop=features_to_crop)

        # ----------------------------------------------------------------
        # 7. NOISE CALCULATION
        # ----------------------------------------------------------------

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
        # 9. COMPUTE L1C
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
        # 10. ADD CLIMATE ZONES
        # ----------------------------------------------------------------
        climate_zones_task = AddClimateZones(tiff_climate=TIFF_CLIMATE_PATH)

        # ----------------------------------------------------------------
        # 11. SAVE IMAGE AS TIFF
        # ----------------------------------------------------------------

        export_task = ExportEOPatchTask(folder_path=self.output_path_tifs, filename=self.output_file_name,
                                        lat_topleft=self.lat_topleft, lon_topleft=self.lon_topleft,
                                        use_local_l1c=self.use_local_l1c)
        


        # ----------------------------------------------------------------
        # 12. CREATE WORKFLOW
        # ----------------------------------------------------------------

        nodes = linearly_connect_tasks(
                    create_eopatch_task,
                    fill_eopatch_task,
                    input_task_zenith,
                    resample_zenith_angles,
                    add_meta_task,
                    radiance_task,
                    add_pan_task,
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
        
        workflow = EOWorkflow(nodes)

        
        
        return nodes, workflow

    def run(self):
        if self.use_local_l1c:
            results = self.workflow.execute()
        else:
            results = self.workflow.execute({self.nodes[0]: {"bbox": self.bbox, "time_interval": self.time_interval}})
        return results

