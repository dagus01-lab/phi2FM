# 1. Data Simulation


## PhilEO-Bench

Warning, I recommend not running this. It can take over 7 days. It is also very specific, since it takes PhilEO-Bench dataset locations and tries to find the best date without clouds. It is very time consuming, and also SentinelHub credit consuming.

I dont think it's useful to run this.

### Prerrequisites
- `pip install s2cloudless`  (used it to select dates without clouds)
- `pip install eo-learn` (SentinelHub)
- Correct paths in `/home/phimultigpu/phisat2_foundation/phi2FM/data_simulation/__init__.py`


### What to run
- `python simulate_phileo_bench.py`
- Do not even bother running the L1A/L1B simulation. It is way too specific to this use case.



## Pretraining Data

1. `1_build_download_df.py` -- create a csv with the locations to download -- > 13k, sampled uniformly across the world (except some margin of extreme lon/lat, which don't have climate data)
2. `2_download_majortom.py` -- download the data from MajorTOM specified in the csv. It also creates updated csv which what files were able to be downloaded and which ones not
3. `3_simulate_phisat2_from_majortom.py` -- simulates the data from the csv downloaded. The major chunk of this process occurs in `workflow_majortom_phisat2.py`
4. `4_add_climatezones_to_tiff.py` -- simply adds another band with the climate zone class per pixel
5. `5_train_test_split.py` -- divides the tiff into train, val, and test, preparing it for the division of tiff images into np patches (at `pretraining_patches.py`)

## TIFF-To Numpy Patches
Scripts to convert the TIFF files (of pretraining data and PhilEO-Bench data) from TIFF files of bigger size, to smaller patches in numpy format.

Also, optionally use `batch_size_one_arrays.py` to have one data point per file. This helped me in implementing DDP (buteo memmaped arrays could not be opened in DDP).

Finally, `np_to_lmdb.py` converts the individual `.npy` files to `lmdb`. It is specific to the pretraining data, since for example, it converts the climate segmentation label to a single classification value for the entire image.