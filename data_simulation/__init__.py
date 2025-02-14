# ------------------------------------------------
# Sentinel Hub client configuration
# ------------------------------------------------
from sentinelhub import SHConfig

sh_config = SHConfig()
sh_config.sh_client_id = "692b79bf-9a1a-465e-850f-bf656c990ca7"
sh_config.sh_client_secret = "WNgI6nQyB3izQW86aCaMwLjtc2Atrs8Z"

# ------------------------------------------------
# Define a single base folder for everything
# ------------------------------------------------
base_folder = "/home/phimultigpu/phisat2_foundation"
phileo_bench_folder = f"{base_folder}/phi2FM/data_simulation/simulation_aux_files/phileo-bench"
pretrain_folder = f"{base_folder}/phi2FM/data_simulation/simulation_aux_files/pretrain"

# ------------------------------------------------
# PhilEO-Bench paths
# ------------------------------------------------

path_to_data = f"{phileo_bench_folder}/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"
label_names_path = f"{phileo_bench_folder}/metadata/label_names.csv"
old_locations_path = f"{phileo_bench_folder}/metadata/old_locations.csv"
new_locations_path = f"{phileo_bench_folder}/metadata/new_locations.csv"
train_test_files_path = f"{phileo_bench_folder}/train_test_location_all.json"

output_path_tifs_phileobench = f"{base_folder}/output"

month_counts_csv = f"{phileo_bench_folder}/metadata/month_counts_v2.csv"
phileo_s2_tif_path = "/home/phimultigpu/phileo_NFS/phileo_data/downstream/downstream_dataset_tifs"
phileo_phisat2_output_tiff_path = f"{base_folder}/output"

path_executable = f"{base_folder}/phi2FM/data_simulation/simulation_aux_files/phisat2_unix.bin"

# ------------------------------------------------
# Pretraining paths
# ------------------------------------------------
df_to_download_path = f"{pretrain_folder}/df_missing.csv"
df_to_simulate_path = f"{pretrain_folder}/df_missing.csv"
df_simulated_path = f"{pretrain_folder}/df_simulation.csv"
climate_tif_path = f"{pretrain_folder}/climate_4326.tif"
local_directory_majortom = f"{base_folder}/draft"
