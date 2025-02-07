# %%
from src import *

from pathlib import Path
import urllib.request

# ### 1. 📅 Filtering based on location, time, and cloud cover
# First we will download a local copy of the dataset metadata, in this case from `Major-TOM/Core-S2L2a`
SOURCE_DATASET = 'Major-TOM/Core-S2L1C' # Identify HF Dataset
DATASET_DIR = Path('./data/Major-TOM/')
DATASET_DIR.mkdir(exist_ok=True, parents=True)
ACCESS_URL = 'https://huggingface.co/datasets/{}/resolve/main/metadata.parquet?download=true'.format(SOURCE_DATASET)
LOCAL_URL = DATASET_DIR / '{}.parquet'.format(ACCESS_URL.split('.parquet')[0].split('/')[-1])

if False:
  # download from server to local url
  gdf = metadata_from_url(ACCESS_URL, LOCAL_URL)

  # then use it via our `filter_metadata` function
  filtered_df = filter_metadata(gdf,
                                cloud_cover = (0,5), # cloud cover between 0% and 10%
                              #   region=switzerland, # you can try with different bounding boxes, like in the cell above
                                daterange=('2020-01-01', '2025-01-01'), # temporal range
                                nodata=(0.0,0.0) # only 0% of no data allowed
                                )
  filtered_df = filtered_df[::11]

  filtered_df.to_csv('filtered_df.csv', index=False)

if False:
  filtered_df = pd.read_csv('df_missing.csv')
  dfs = np.array_split(filtered_df, 10)
  index_download = 9
  df_to_download = dfs[index_download]
  
  print(f'Number of images: {len(filtered_df)}, Downloading {len(df_to_download)} images')

# ### 📩 Downloading a filtered subset of the dataset
# Use the `filter_download` function to download all files to the local directory at `local_dir`. Your new dataset will be named using `source_name`.
# More importantly, the `by_row` option allows to download specific rows from the archives. Set it to `True`, if you think you will take only a few files from each parquet file (most parquet files contain samples that are close to each other in space).
# If you expect to take most of the samples from the parquet file, setting `by_row` to `False` will probably be quicker (you then download the data as the entire file, before you rearrange it onto folders with only the files from your dataframe).
from concurrent.futures import ThreadPoolExecutor, as_completed

df_to_download = pd.read_csv('df_missing.csv')
parallel_process = True

num_splits = 10  # You can adjust the number of splits based on your data size and experiment with this number for optimal performance
dfs = np.array_split(df_to_download, num_splits)

# Directory and source name setup
local_directory = '/home/ccollado/phileo_phisat2/MajorTOM/L1C_v2/'
source_name = 'L1C'


if parallel_process:
  with ThreadPoolExecutor(max_workers=num_splits) as executor:
      # Submit all the tasks to the executor
      futures = [executor.submit(filter_download, df, local_directory, source_name, by_row=True) for df in dfs]

      # Optionally, you can handle the results or exceptions from the tasks
      for future in as_completed(futures):
          try:
              result = future.result()  # This will raise any exceptions caught during the execution of the task
              # Process result if needed, or just check for successful completion
              print("Task completed successfully")
          except Exception as exc:
              print(f'Task generated an exception: {exc}')

else:
  filter_download(df_to_download, local_dir=local_directory, source_name='L1C', by_row=True)