# %%
from src import *

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_simulation import df_to_download_path, local_directory_majortom

# ### 1. 📅 Filtering based on location, time, and cloud cover
# First we will download a local copy of the dataset metadata, in this case from `Major-TOM/Core-S2L2a`
SOURCE_DATASET = 'Major-TOM/Core-S2L1C' # Identify HF Dataset
DATASET_DIR = Path('./data/Major-TOM/')
DATASET_DIR.mkdir(exist_ok=True, parents=True)
ACCESS_URL = 'https://huggingface.co/datasets/{}/resolve/main/metadata.parquet?download=true'.format(SOURCE_DATASET)
LOCAL_URL = DATASET_DIR / '{}.parquet'.format(ACCESS_URL.split('.parquet')[0].split('/')[-1])

# ### 📩 Downloading a filtered subset of the dataset

df_to_download = pd.read_csv(df_to_download_path)
parallel_process = True

num_splits = 10  # You can adjust the number of splits based on your data size and experiment with this number for optimal performance
dfs = np.array_split(df_to_download, num_splits)

# Directory and source name setup
source_name = 'L1C'

if parallel_process:
  with ThreadPoolExecutor(max_workers=num_splits) as executor:
      # Submit all the tasks to the executor
      futures = [executor.submit(filter_download, df, local_directory_majortom, source_name, by_row=True) for df in dfs]

      # Optionally, you can handle the results or exceptions from the tasks
      for future in as_completed(futures):
          try:
              result = future.result()  # This will raise any exceptions caught during the execution of the task
              # Process result if needed, or just check for successful completion
              print("Task completed successfully")
          except Exception as exc:
              print(f'Task generated an exception: {exc}')

else:
  filter_download(df_to_download, local_dir=local_directory_majortom, source_name='L1C', by_row=True)

# Check file existence
tif_bands = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'cloud_mask']
ds = MajorTOM(pd.read_csv(df_to_download_path), local_directory_majortom, tif_bands = tif_bands, combine_bands=False)
existing_indices, missing_indices, df_existing, df_missing = ds.check_file_existence()
df_existing.to_csv(DATASET_DIR / 'existing_files.csv', index=False)
df_missing.to_csv(DATASET_DIR / 'missing_files.csv', index=False)

