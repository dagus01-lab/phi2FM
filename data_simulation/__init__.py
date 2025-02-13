from sentinelhub import SHConfig
sh_config = SHConfig()
sh_config.sh_client_id = "692b79bf-9a1a-465e-850f-bf656c990ca7"
sh_config.sh_client_secret = "WNgI6nQyB3izQW86aCaMwLjtc2Atrs8Z"




# -------------------------
# PhilEO-Bench
# -------------------------

path_to_data = '/home/phimultigpu/phisat2_foundation/phi2FM/data_simulation/simulation_aux_files/phileo-bench/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp'

label_names_path = '/home/phimultigpu/phisat2_foundation/phi2FM/data_simulation/simulation_aux_files/phileo-bench/metadata/label_names.csv'
old_locations_path = '/home/phimultigpu/phisat2_foundation/phi2FM/data_simulation/simulation_aux_files/phileo-bench/metadata/old_locations.csv'
new_locations_path = '/home/phimultigpu/phisat2_foundation/phi2FM/data_simulation/simulation_aux_files/phileo-bench/metadata/new_locations.csv'



train_test_files_path = '/home/phimultigpu/phisat2_foundation/phi2FM/data_simulation/simulation_aux_files/phileo-bench/train_test_location_all.json'

output_path_tifs_phileobench = '/home/phimultigpu/phisat2_foundation/output'


month_counts_csv = '/home/phimultigpu/phisat2_foundation/phi2FM/data_simulation/simulation_aux_files/phileo-bench/metadata/month_counts_v2.csv'
phileo_s2_tif_path = '/home/phimultigpu/phileo_NFS/phileo_data/downstream/downstream_dataset_tifs'
phileo_phisat2_output_tiff_path = '/home/phimultigpu/phisat2_foundation/output'
path_executable = "/home/phimultigpu/phisat2_foundation/phi2FM/data_simulation/simulation_aux_files/phisat2_unix.bin"


# -------------------------
# Pretraining
# -------------------------

# Download data from MajorTOM
df_to_download_path = '/home/phimultigpu/phisat2_foundation/phi2FM/data_simulation/simulation_aux_files/pretrain/df_missing.csv'
df_to_simulate_path = '/home/phimultigpu/phisat2_foundation/phi2FM/data_simulation/simulation_aux_files/pretrain/df_missing.csv'
climate_tif_path = '/home/phimultigpu/phisat2_foundation/phi2FM/data_simulation/simulation_aux_files/pretrain/climate_4326.tif'
local_directory_majortom = '/home/phimultigpu/phisat2_foundation/draft/'



