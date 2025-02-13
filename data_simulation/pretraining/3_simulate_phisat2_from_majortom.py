import os
# cwd = os.getcwd()

# import sys
# sys.path.append('/home/ccollado/1_simulate_data')

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

from src import *

def process_row(row_dict, tif_bands, sh_client_id, sh_client_secret, output_path_tifs):
    try:
        # Reconstruct ds_point from the row dictionary
        row = pd.Series(row_dict)
        ds = MajorTOM(pd.DataFrame([row]), os.path.join(local_directory_majortom, 'L1C'), tif_bands=tif_bands, combine_bands=True)

        # Access the first (and only) item in ds
        ds_point = ds[0]

        output_file_name = f"{ds_point['meta']['unique_identifier']}.tif"
        output_file_path = os.path.join(output_path_tifs, output_file_name)

        if not os.path.exists(output_file_path):
            # print(f"Processing {output_file_name}")

            bands = ds_point['bands']
            if bands.shape != (7, 1068, 1068):
                raise ValueError(f"The bands array has the wrong shape. It should be (7, 1068, 1068), but it is {bands.shape}")

            workflow = phisat2simulation(
                output_file_name=output_file_name,
                process_level='L1C',
                output_path_tifs=output_path_tifs,
                sh_client_id=sh_client_id,
                sh_client_secret=sh_client_secret,
                plot_results=False,
                use_local_l1c=True,
                ds_point=ds_point,
                height_pixels=1068,
                width_pixels=1068,
            )

            try:
                workflow.run()
            except Exception as e:
                print(f"An error occurred during workflow run for {output_file_name}: {e}")
    except Exception as e:
        print(f"An error occurred while processing row {row_dict.get('index', 'unknown')}: {e}")



if __name__ == '__main__':

    run_parallel = True

    tif_bands = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'cloud_mask']
    df_simulation = pd.read_csv(df_to_simulate_path)

    ds = MajorTOM(df_simulation, os.path.join(local_directory_majortom, 'L1C'), tif_bands=tif_bands, combine_bands=True)

    output_path_tifs = os.path.join(local_directory_majortom, 'tiff_files')
    os.makedirs(output_path_tifs, exist_ok=True)

    # Prepare the list of row dictionaries
    rows = df_simulation.to_dict('records')

    process_partial = partial(
        process_row,
        tif_bands=tif_bands,
        sh_client_id=sh_config.sh_client_id,
        sh_client_secret=sh_config.sh_client_secret,
        output_path_tifs=output_path_tifs
    )

    if run_parallel:
        with Pool(processes=os.cpu_count()) as pool:
            list(tqdm(pool.imap_unordered(process_partial, rows), total=len(rows)))
    else:
        for row in tqdm(rows, total=len(rows)):
            process_partial(row)

