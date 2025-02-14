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

        df_simulated = pd.read_csv('/home/ccollado/1_simulate_data/Major-TOM/df_simulation.csv')
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





def protocol_fewshot_memmapped(folder: str,
                     dst: str,
                     n: int = 10,
                     val_ratio: float = 0.2,
                     regions: list = None,
                     y: str = 'building',
                     data_selection: str = 'strict',
                     name: str = '128_10m',
                     crop_images: bool = False
                     ):

    """
    Loads n-samples data from specified geographic regions.
    :param folder: dataset source folder
    :param dst: save folder
    :param n: number of samples
    :param val_ratio: ratio of validation set
    :param regions: geographical regions to sample
    :param y: downstream label from roads, kg, building, lc, coords
    :param data_selection: choose from 'strict' (take train/val selection from predefined selection), 'create' (use train/val selection if exists, else create it), 'random' (create train/val selection randomly)
    :return: train, val MultiArrays
    """

    if regions is None:
        regions = list(REGIONS_BUCKETS.keys())
        # import pdb ; pdb.set_trace()
    else:
        # import pdb ; pdb.set_trace()
        for r in regions:
            assert r in list(REGIONS_BUCKETS.keys()), f"region {r} not found. Possible regions are {list(REGIONS_BUCKETS.keys())}"
    regions = check_region_validity(folder, regions, y)

    assert data_selection in ['strict','create','random']

    samples_loaded = False
    if data_selection != 'random':
        indices_path = glob(f"indices_phisat2/indices_*_{name}_{y}_{n}.json")
        
        if len(indices_path) == 0:
            if data_selection == 'create':
                samples_dict = {}
                print(f'creating train/val selection for task {y}, nshot={n}')
            else:
                raise ValueError('No file found for nshot sample selection while data_selection="strict". If you want to create fixed indices on the fly or use random train/val samples consider setting data_selction to "create" or "random"')
        
        elif len(indices_path) > 1:
            raise ValueError('Multiple files found for nshot sample selection')
        
        else:
            samples_loaded = True
            print('Loading predefined train/val selection')
            with open(indices_path[0], 'r') as f:
                samples_dict = json.load(f)

    x_train_samples = []
    y_train_samples = []
    x_val_samples = []
    y_val_samples = []

    # pos weights for lc classification
    pos_counts = np.zeros(11)
    neg_counts = np.zeros(11)

    for i, region in enumerate(regions):
        print(i,region)

        # generate multi array for region
        x_train_files = []
        for sub_regions in REGIONS_BUCKETS[region]: 
            x_train_files += sorted(glob(os.path.join(folder, f"{sub_regions}*train_s2.npy")))
        y_train_files = [f_name.replace('s2', f'label_{y}') for f_name in x_train_files]
        x_val_files = [f_name.replace('train', 'val') for f_name in x_train_files]
        y_val_files = [f_name.replace('train', 'val') for f_name in y_train_files]

        # pdb.set_trace()

        # checks that s2 and label numpy files are consistent
        x_train_files, y_train_files = sanity_check_labels_exist(x_train_files, y_train_files)
        x_val_files, y_val_files = sanity_check_labels_exist(x_val_files, y_val_files)

        x_train = beo.MultiArray([load_and_crop(f, crop_images) for f in x_train_files])
        y_train = beo.MultiArray([load_and_crop(f, crop_images) for f in y_train_files])
        x_val = beo.MultiArray([load_and_crop(f, crop_images) for f in x_val_files])
        y_val = beo.MultiArray([load_and_crop(f, crop_images) for f in y_val_files])

        n_train_samples = min(n, len(x_train))
        n_val_samples = min(int(np.ceil(n * val_ratio)), len(x_val))

        if samples_loaded:
            assert len(x_train) == samples_dict[region]['length_multi_array_train']
            assert len(x_val) == samples_dict[region]['length_multi_array_val']

            train_indices = samples_dict[region]['train_indices']
            val_indices = samples_dict[region]['val_indices']

            # assert len(train_indices) == n_train_samples
            # import pdb; pdb.set_trace()
            # assert len(val_indices) == n_val_samples

        else:
            train_indices= random.Random(12345).sample(range(0, len(x_train)), n_train_samples)
            val_indices  = random.Random(12345).sample(range(0, len(y_val)), n_val_samples)

            if y == 'building' or y == 'building_classification':
                if y == 'building':
                    hot_encoded_train = np.array([to_one_hot_building(yt) for yt in y_train])
                    hot_encoded_val = np.array([to_one_hot_building(yt) for yt in y_val])
                elif y == 'building_classification':
                    hot_encoded_train = np.array([yt for yt in y_train])
                    hot_encoded_val = np.array([yt for yt in y_val])

                train_indices = proportional_subset_indices(hot_encoded_train, n_train_samples, max_n_shot=5000)
                val_indices = proportional_subset_indices(hot_encoded_val, n_val_samples, max_n_shot=int(5000*val_ratio))

            # elif y == 'lc' or y == 'lc_classification':
            elif y == 'lc_classification':
                chosen_by_lp_rus = 2/3
                if y == 'lc':
                    hot_encoded_train = np.array([to_one_hot_lc(yt) for yt in y_train])
                    hot_encoded_val = np.array([to_one_hot_lc(yt) for yt in y_val])
                elif y == 'lc_classification':
                    hot_encoded_train = np.array([yt for yt in y_train])
                    hot_encoded_val = np.array([yt for yt in y_val])

                # train
                idx_train = LP_RUS_with_scale_down(hot_encoded_train, max_n_shot=5000, n_shot=int(n_train_samples * chosen_by_lp_rus))
                remaining_indices = list(set(range(len(y_train))) - set(idx_train))
                random_indices = random.sample(remaining_indices, n_train_samples - len(idx_train))
                train_indices = idx_train + random_indices
                random.shuffle(train_indices)
                
                num_samples = len(hot_encoded_train[train_indices])
                pos_counts += np.sum(hot_encoded_train[train_indices], axis=0)
                neg_counts += num_samples - np.sum(hot_encoded_train[train_indices], axis=0)
                
                # val
                idx_val = LP_RUS_with_scale_down(hot_encoded_val, max_n_shot=int(5000*val_ratio), n_shot=int(n_val_samples * chosen_by_lp_rus))
                remaining_indices = list(set(range(len(y_val))) - set(idx_val))
                random_indices = random.sample(remaining_indices, n_val_samples - len(idx_val))
                val_indices = idx_val + random_indices
                random.shuffle(val_indices)



            samples_dict[region] = {'train_indices':train_indices, 'val_indices':val_indices, 'length_multi_array_train':len(x_train), 'length_multi_array_val':len(x_val)}

        x_train_samples += [x_train[i] for i in train_indices]
        y_train_samples += [y_train[i] for i in train_indices]

        x_val_samples += [x_val[i] for i in val_indices]
        y_val_samples += [y_val[i] for i in val_indices]

    # if not samples_loaded and data_selection=='create':
        # out_path = f'indices/indices_{date.today().strftime("%d%m%Y")}_{name}_{y}_{n}.json'
        # print(f'No predefined train/val sampling was used. Saving current sampling schema in {out_path}')
        # with open(out_path, 'w') as f:
        #     json.dump(samples_dict, f)
    
    if y == 'lc_classification':
        pos_weight = neg_counts / (pos_counts + 1e-8)
    else:
        pos_weight = None
    
    if y == 'building_classification':
        class_counts = np.sum(y_train_samples, axis=0)
        weights = np.where(class_counts > 0, 1.0 / class_counts, 1)
        weights[class_counts > 0] /= np.sum(weights[class_counts > 0])
    else:
        weights = None
    
    return MultiArray_1D(x_train_samples), MultiArray_1D(y_train_samples), MultiArray_1D(x_val_samples), MultiArray_1D(y_val_samples), pos_weight, weights

