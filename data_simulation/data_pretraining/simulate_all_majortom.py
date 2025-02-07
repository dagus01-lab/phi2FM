import os
cwd = os.getcwd()

import sys
sys.path.append('/home/ccollado/1_simulate_data')

from osgeo import gdal
gdal.DontUseExceptions()

import pandas as pd
import numpy as np
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

from phisat_2.utils import src_band
from workflow_majortom_phisat2 import phisat2simulation
from sentinelhub import SHConfig
from multiprocessing import Pool
from functools import partial

os.chdir(cwd)
print(os.getcwd())

from src import *



'''	

for i in tqdm(range(len(ds))):
    output_file_name = f"{ds[i]['meta']['product_id']}.tif"

    if not os.path.exists(os.path.join(output_path_tifs, output_file_name)):
        print(f"Processing {output_file_name}")

        bands = ds[i]['bands']
        if (bands.shape[0] != 7) or (bands.shape[1] != 1068) or (bands.shape[2] != 1068):
            raise ValueError('The bands array has the wrong shape. It should be (7, 1068, 1068), but it is {}'.format(bands.shape))

        workflow = phisat2simulation(output_file_name=output_file_name, process_level='L1C', output_path_tifs=output_path_tifs, 
                                    sh_client_id=sh_config.sh_client_id, sh_client_secret=sh_config.sh_client_secret,
                                    plot_results=False,
                                    use_local_l1c = True, ds_point = ds[i],
                                    height_pixels = 1068, width_pixels = 1068,
                                    )

        try:
            workflow.run()
        except Exception as e:
            print(f"An error occurred: {e}")


'''	



def process_row(row_dict, tif_bands, sh_client_id, sh_client_secret, output_path_tifs):
    try:
        # Reconstruct ds_point from the row dictionary
        row = pd.Series(row_dict)
        ds = MajorTOM(pd.DataFrame([row]), '/home/ccollado/phileo_phisat2/MajorTOM/L1C', tif_bands=tif_bands, combine_bands=True)

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
    df_simulation = pd.read_csv('df_existing.csv')
    # df_simulation = df_simulation.iloc[13800:]
    # df_simulation = df_simulation.sample(20)

    ds = MajorTOM(df_simulation, '/home/ccollado/phileo_phisat2/MajorTOM//L1C', tif_bands=tif_bands, combine_bands=True)

    sh_config = SHConfig()
    sh_config.sh_client_id = "8bb6bc93-7333-4e48-ba3a-f527563a1fb0"
    sh_config.sh_client_secret = "0bOrhm6IUmIDdUgOyso45fjIqVhCxQuB"

    output_path_tifs = '/home/ccollado/phileo_phisat2/MajorTOM/tiff_files'
    os.makedirs(output_path_tifs, exist_ok=True)

    # Prepare the list of row dictionaries
    rows = df_simulation.to_dict('records')

    process_partial = partial(
        process_row,c
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

