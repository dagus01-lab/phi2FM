# Standard Library
import os
from glob import glob
import pandas as pd

# External Libraries
import buteo as beo
import numpy as np
import random
import json

from utils.training_utils import MultiArray_1D

import numpy as np
import random

from downstream.utils.data_protocol_utils import (
    check_region_validity,
    to_one_hot_building,
    to_one_hot_lc,
    proportional_subset_indices,
    LP_RUS_with_scale_down,
    sanity_check_labels_exist,
    load_and_crop,
    REGIONS_BUCKETS,
    REGIONS
)

random.seed(97)
np.random.seed(1234) # also affect pandas





def get_testset(folder: str,
                regions: list = None,
                y: str = 'building',
                crop_images: bool = False,
                by_region: bool = False):

    """
    Loads a pre-defined test set data from specified geographic regions.
    :param folder: dataset source folder
    :param regions: geographical regions to sample
    :param y: downstream label from roads, kg, building, lc, coords
    :return: test MultiArrays
    """
    x_test_files = []
    
    if by_region:

        if regions is None:
            regions = REGIONS
        else:
            for r in regions:
                assert r in REGIONS, f"region {r} not found"

        for region in regions:
            # get test samples of region
            x_test_files = x_test_files + sorted(glob(os.path.join(folder, f"{region}*test_s2.npy")))

        y_test_files = [f_name.replace('s2', f'label_{y}') for f_name in x_test_files]
    
    else:
        from data_simulation import df_simulated_path
        df_simulated = pd.read_csv(df_simulated_path)
        df = df_simulated['unique_identifier']

        test_files = os.listdir(folder)
        test_files = [f.replace('_test_s2.npy', '') for f in test_files if f.endswith('test_s2.npy')]

        df_test = df[df.isin(test_files)]
        df_test = df_test + '_test_s2.npy'
        x_test_files = df_test.tolist()
        x_test_files = [os.path.join(folder, f_name) for f_name in x_test_files]

        y_test_files = [f.replace('s2', f'label_{y}') for f in x_test_files]
    
    x_test_files, y_test_files = sanity_check_labels_exist(x_test_files, y_test_files)
    
    x_test = beo.MultiArray([load_and_crop(f, crop_images) for f in x_test_files])
    y_test = beo.MultiArray([load_and_crop(f, crop_images) for f in y_test_files])

    assert len(x_test) == len(y_test), "Lengths of x and y do not match."

    return x_test, y_test




def protocol_split(folder: str,
                   split_percentage: float = 0.1,
                   regions: list = None,
                   y: str = 'building',
                   by_region: bool = False):
    """
    Loads a percentage of the data from specified geographic regions.
    :param folder: dataset source folder
    :param split_percentage: percentage of data to sample from each region
    :param regions: geographical regions to sample
    :param y: downstream label from roads, kg, building, lc, coords
    :return: train, val MultiArrays
    """

    assert 0 < split_percentage <= 1, "split percentage out of range (0 - 1)"

    if by_region:

        if regions is None:
            regions = list(REGIONS_BUCKETS.keys())
        else:
            for r in regions:
                assert r in list(REGIONS_BUCKETS.keys()), f"region {r} not found. Possible regions are {list(REGIONS_BUCKETS.keys())}"
        
        df = pd.read_csv(glob(os.path.join(folder, f"*.csv"))[0])
        df = df.sort_values(by=['samples'])

        x_train_files = []
        shots_per_region = {'total':0}
        # egions =[subregion for r in regions for subregion in REGIONS_BUCKETS[r]]
        for region in regions:
            mask = [False]*len(df)
            for subregion in REGIONS_BUCKETS[region]:
                submask = [subregion in f for f in df.iloc[:, 0]]
                mask = [any(tuple) for tuple in zip(mask, submask)]
            mask = [region in f for f in df.iloc[:, 0]]
            df_temp = df[mask].sample(frac=1).copy().reset_index(drop=True)
            # skip iteration if Region does not belong to current dataset
            if df_temp.shape[0] == 0:
                continue

            df_temp['cumsum'] = df_temp['samples'].cumsum()

            # find row with closest value to the required number of samples
            idx_closest = df_temp.iloc[
                (df_temp['cumsum'] - int(df_temp['samples'].sum() * split_percentage)).abs().argsort()[:1]].index.values[0]
            x_train_files = x_train_files + list(df_temp.iloc[:idx_closest, 0])
            
            shots_per_region[region] = df_temp['cumsum'].values[idx_closest]

        shots_per_region['total'] = sum(shots_per_region.values())

        x_train_files = [os.path.join(folder, f_name) for f_name in x_train_files]
        y_train_files = [f_name.replace('s2', f'label_{y}') for f_name in x_train_files]
        x_val_files = [f_name.replace('train', 'val') for f_name in x_train_files]
        y_val_files = [f_name.replace('train', 'val') for f_name in y_train_files]

    else:
        from data_simulation import df_simulated_path
        df_simulated = pd.read_csv(df_simulated_path)
        df = df_simulated['unique_identifier']

        folder_files = os.listdir(folder)

        def get_split_files(folder, split='train'):
            split_files = [f.replace(f'_{split}_s2.npy', '') for f in folder_files if f.endswith(f'_{split}_s2.npy')]

            df_split = df[df.isin(split_files)]
            df_split = df_split + f'_{split}_s2.npy'
            x_split_files = df_split.tolist()
            x_split_files = [os.path.join(folder, f_name) for f_name in x_split_files]

            N = len(x_split_files)
            K = int(N * split_percentage)

            # Generate equally spaced indices
            indices = np.linspace(0, N - 1, num=K, endpoint=True)
            indices = np.round(indices).astype(int)

            # Ensure indices are within bounds and unique
            indices = np.clip(indices, 0, N - 1)
            indices = np.unique(indices)

            # Select the rows using the indices
            x_split_files = [x_split_files[i] for i in indices]
            x_split_files = [os.path.join(folder, f_name) for f_name in x_split_files]
            y_split_files = [f_name.replace('s2', f'label_{y}') for f_name in x_split_files]
            
            return x_split_files, y_split_files

        x_train_files, y_train_files = get_split_files(folder, split='train')
        
        if (os.path.basename(x_train_files[0])).replace('train', 'val') not in folder_files:
            print("Validation data from different tiff files")
            x_val_files, y_val_files = get_split_files(folder, split='val')
        else:
            print("Validation data from same tiff files as train data")
            x_val_files = [f_name.replace('train', 'val') for f_name in x_train_files]
            y_val_files = [f_name.replace('train', 'val') for f_name in y_train_files]

    # import pdb; pdb.set_trace()

    # checks that s2 and label numpy files are consistent
    x_train_files, y_train_files = sanity_check_labels_exist(x_train_files, y_train_files)
    x_val_files, y_val_files = sanity_check_labels_exist(x_val_files, y_val_files)

    x_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_train_files])
    y_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_train_files])

    x_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_val_files])
    y_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_val_files])

    assert len(x_train) == len(y_train)  and len(x_val) == len(
        y_val), "Lengths of x and y do not match."

    return x_train, y_train, x_val, y_val



from typing import List, Tuple

SEED = 12345
_rng = random.Random(SEED)          # one local RNG instance used everywhere


def _indices_filename(name: str, y: str, n: int) -> str:
    """
    Unique, stable file name for the (name, y, n) configuration.
    """
    return f"indices_phisat2/indices_{name}_{y}_{n}.json"


def protocol_fewshot_memmapped(
    folder: str,
    dst: str,
    n: int = 10,
    val_ratio: float = 0.2,
    regions: List[str] = None,
    y: str = "building",
    data_selection: str = "create",      # <-- new default
    name: str = "128_10m",
    crop_images: bool = False
) -> Tuple[
    "MultiArray_1D", "MultiArray_1D",
    "MultiArray_1D", "MultiArray_1D",
    np.ndarray, np.ndarray
]:
    """
    Few-shot loader that is random only **once** (first run with this
    (name, y, n) configuration).  
    Later runs reuse the saved indices so results are deterministic.

    Parameters
    ----------
    folder : str
        Root directory of the data.
    dst : str
        Not used inside this function but kept for API compatibility.
    n : int
        Total shots per region for the **train** set.
    val_ratio : float
        Validation shots are ceil(n * val_ratio) per region.
    regions : list[str] or None
        Regions to sample. Default = all.
    y : str
        Task label: 'building', 'building_classification',
        'lc', 'lc_classification', ...
    data_selection : {'strict', 'create', 'random'}
        * 'strict'  – must find an existing file; error if not found  
        * 'create'  – load file if it exists, otherwise **create + save**  
        * 'random'  – ignore / overwrite file every run
    name : str
        Spatial resolution identifier that is part of the file key.
    crop_images : bool
        Pass-through to `load_and_crop`.
    """

    # ---------------- region sanity checks ----------------
    if regions is None:
        regions = list(REGIONS_BUCKETS.keys())            # type: ignore
    else:
        for r in regions:
            assert r in REGIONS_BUCKETS, (                # type: ignore
                f"Region {r} not found. "
                f"Possible regions: {list(REGIONS_BUCKETS.keys())}"   # type: ignore
            )
    regions = check_region_validity(folder, regions, y)   # type: ignore
    assert data_selection in {"strict", "create", "random"}

    # ---------------- indices discovery / load ----------------
    index_file = _indices_filename(name, y, n)
    samples_loaded = (
        data_selection != "random" and os.path.exists(index_file)
    )

    if samples_loaded:
        print(f"Loading predefined train/val selection from {index_file}")
        with open(index_file, "r") as f:
            samples_dict = json.load(f)
    else:
        print(f"Creating new train/val selection in {index_file}")
        samples_dict = {}

    # ---------------- accumulators ----------------
    x_train_samples, y_train_samples = [], []
    x_val_samples,   y_val_samples   = [], []


    # ---------------- iterate over regions ----------------
    for region in regions:
        print(region)

        # --- gather all npy paths for the region ---
        x_train_files = []
        for sub in REGIONS_BUCKETS[region]:               # type: ignore
            x_train_files += sorted(
                glob(os.path.join(folder, f"{sub}*train_s2.npy"))
            )

        y_train_files = [f.replace("s2", f"label_{y}") for f in x_train_files]
        x_val_files   = [f.replace("train", "val")        for f in x_train_files]
        y_val_files   = [f.replace("train", "val")        for f in y_train_files]

        # -- file existence sanity checks --
        x_train_files, y_train_files = sanity_check_labels_exist(
            x_train_files, y_train_files
        )
        x_val_files, y_val_files = sanity_check_labels_exist(
            x_val_files, y_val_files
        )

        # -- load into MultiArrays (or ndarray lists) --
        x_train = beo.MultiArray([load_and_crop(f, crop_images) for f in x_train_files])    # type: ignore
        y_train = beo.MultiArray([load_and_crop(f, crop_images) for f in y_train_files])    # type: ignore
        x_val   = beo.MultiArray([load_and_crop(f, crop_images) for f in x_val_files])      # type: ignore
        y_val   = beo.MultiArray([load_and_crop(f, crop_images) for f in y_val_files])      # type: ignore

        # -- number of samples per split for this region --
        n_train_samples = min(n, len(x_train))
        n_val_samples   = min(int(np.ceil(n * val_ratio)), len(x_val))

        # ==================================================
        #  Choose indices
        # ==================================================
        if samples_loaded:
            # consistency checks in strict mode
            assert len(x_train) == samples_dict[region]["length_multi_array_train"]
            assert len(x_val)   == samples_dict[region]["length_multi_array_val"]

            train_indices = samples_dict[region]["train_indices"]
            val_indices   = samples_dict[region]["val_indices"]

        else:
            # ------------------------------------------------------------------
            # 1. Start with a uniform random sample from the local RNG
            train_indices = _rng.sample(range(len(x_train)), n_train_samples)
            val_indices   = _rng.sample(range(len(x_val)),   n_val_samples)

            # ------------------------------------------------------------------
            # 2. Optionally replace them with class-balanced subsets
            #    (exactly the same logic you had, just switching to `_rng`)
            # ------------------------------------------------------------------
            if y in {"building", "building_classification"}:
                if y == "building":
                    hot_train = np.array([to_one_hot_building(yt) for yt in y_train])
                    hot_val   = np.array([to_one_hot_building(yt) for yt in y_val])
                else:  # "building_classification"
                    hot_train = np.array([yt for yt in y_train])
                    hot_val   = np.array([yt for yt in y_val])

                train_indices = proportional_subset_indices(
                    hot_train, n_train_samples, max_n_shot=5000
                )
                val_indices = proportional_subset_indices(
                    hot_val, n_val_samples, max_n_shot=int(5000 * val_ratio)
                )

            elif y in {"lc", "lc_classification"}:
                # keep ~2/3 from LP-RUS, rest random
                chosen_by_lp_rus = 2 / 3

                if y == "lc":
                    hot_train = np.array([to_one_hot_lc(yt) for yt in y_train])
                    hot_val   = np.array([to_one_hot_lc(yt) for yt in y_val])
                else:  # "lc_classification"
                    hot_train = np.array([yt for yt in y_train])
                    hot_val   = np.array([yt for yt in y_val])

                # train
                idx_train_lp = LP_RUS_with_scale_down(
                    hot_train,
                    max_n_shot=5000,
                    n_shot=int(n_train_samples * chosen_by_lp_rus)
                )
                remaining = list(set(range(len(y_train))) - set(idx_train_lp))
                random_extra = _rng.sample(
                    remaining, n_train_samples - len(idx_train_lp)
                )
                train_indices = idx_train_lp + random_extra
                _rng.shuffle(train_indices)

                # val
                idx_val_lp = LP_RUS_with_scale_down(
                    hot_val,
                    max_n_shot=int(5000 * val_ratio),
                    n_shot=int(n_val_samples * chosen_by_lp_rus)
                )
                remaining = list(set(range(len(y_val))) - set(idx_val_lp))
                random_extra = _rng.sample(
                    remaining, n_val_samples - len(idx_val_lp)
                )
                val_indices = idx_val_lp + random_extra
                _rng.shuffle(val_indices)

            # save indices for this region into dict
            samples_dict[region] = {
                "train_indices": train_indices,
                "val_indices":   val_indices,
                "length_multi_array_train": len(x_train),
                "length_multi_array_val":   len(x_val)
            }

        # =================== gather samples ===================
        x_train_samples += [x_train[i] for i in train_indices]
        y_train_samples += [y_train[i] for i in train_indices]
        x_val_samples   += [x_val[i]   for i in val_indices]
        y_val_samples   += [y_val[i]   for i in val_indices]

    # ---------------- persist indices the first time ----------------
    if not samples_loaded and data_selection == "create":
        os.makedirs(os.path.dirname(index_file), exist_ok=True)
        with open(index_file, "w") as f:
            json.dump(samples_dict, f)
        print(f"Saved new sampling schema to {index_file}")

    # ---------------- pos_weight & class weights ----------------
    if y == "lc_classification":
        ys = np.asarray(y_train_samples)           # shape (N, 11)
        pos_counts = ys.sum(axis=0)
        neg_counts = len(ys) - pos_counts
        pos_weight = neg_counts / (pos_counts + 1e-8)
    else:
        pos_weight = None

    if y == "building_classification":
        class_counts = np.sum(y_train_samples, axis=0)
        weights = np.where(class_counts > 0, 1.0 / class_counts, 1.0)
        weights[class_counts > 0] /= np.sum(weights[class_counts > 0])
    else:
        weights = None

    # ---------------- return as before ----------------
    return (
        MultiArray_1D(x_train_samples),  # type: ignore
        MultiArray_1D(y_train_samples),  # type: ignore
        MultiArray_1D(x_val_samples),    # type: ignore
        MultiArray_1D(y_val_samples),    # type: ignore
        pos_weight,
        weights,
    )
