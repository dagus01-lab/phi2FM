# Standard Library
import os
from glob import glob
import pandas as pd

# External Libraries
import buteo as beo
import numpy as np
import random
import json
from datetime import date
import pdb
from tqdm import tqdm

from utils.training_utils import MultiArray_1D

from collections import defaultdict
import numpy as np
import random
from skmultilearn.problem_transform import LabelPowerset
from sklearn.datasets import make_multilabel_classification
from tabulate import tabulate

def distribute_remainder(r, r_dist, idx):
    p = len(r_dist) - idx + 1
    value = r // p
    curr_rem = r % p

    r_dist[idx:] = np.add(r_dist[idx:], value)
    
    if curr_rem > 0:
        start = len(r_dist) - curr_rem
        r_dist[start:] = np.add(r_dist[start:], 1)



def LP_RUS_with_scale_down(y, max_n_shot, n_shot):
    total_samples = y.shape[0]
    samples_to_delete = total_samples - max_n_shot
    if samples_to_delete <= 0:
        return list(range(total_samples))

    lp = LabelPowerset()
    labelsets = np.array(lp.transform(y))
    label_set_bags = defaultdict(list)
    for idx, label in enumerate(labelsets):
        label_set_bags[label].append(idx)

    # Sort label sets by size descending
    sorted_labels = sorted(label_set_bags.keys(), key=lambda l: len(label_set_bags[l]), reverse=True)

    del_samples = []
    # Iteratively remove samples from the largest sets until we've deleted exactly samples_to_delete
    for label in sorted_labels:
        if len(del_samples) == samples_to_delete:
            break
        # How many can we remove from this label without going below 0
        can_remove = min(len(label_set_bags[label]), samples_to_delete - len(del_samples))
        # Remove "can_remove" samples (no heuristic, just remove from front or randomly)
        for _ in range(can_remove):
            del_samples.append(label_set_bags[label].pop())

    # If we haven't reached the exact deletion count for some reason, try again or adjust logic
    # But if done carefully, it should match exactly.

    all_indices = set(range(total_samples))
    del_set = set(del_samples)
    keep_indices = all_indices - del_set
    keep_indices_max = keep_indices
    
    # Prepare headers for each class column
    # Determine number of classes from y's shape
    n_classes = y.shape[1]
    headers = ["Scenario"] + [f"Class {i+1}" for i in range(n_classes)]
    
    data = []

    # Calculate counts for max_nshot scenario
    pos_counts_max = np.sum(y[list(keep_indices_max)], axis=0)
    data.append([f"max n-shot: {max_n_shot} - count"] + pos_counts_max.tolist())
    
    if n_shot is not None:
        keep_indices = random.sample(keep_indices, n_shot)
        keep_indices_n_shot = keep_indices
        pos_counts_n_shot = np.sum(y[list(keep_indices_n_shot)], axis=0)
        data.append([f"n-shot: {n_shot} - count"] + pos_counts_n_shot.tolist())
    else:
        # If n_shot is not provided, we cannot compute the n_shot row.
        keep_indices_n_shot = None
        pos_counts_n_shot = None


    # Compute percentage rows if counts are available
    # For max_nshot_pct row
    max_nshot_pct = []
    for count in pos_counts_max:
        # Avoid division by zero; assuming max_n_shot > 0
        percentage = (count / max_n_shot) * 100
        max_nshot_pct.append(f"{percentage:.2f}%")
    data.append([f"max n-shot: {max_n_shot} - %"] + max_nshot_pct)

    # For n_shot_pct row, only compute if n_shot scenario exists
    if n_shot is not None and pos_counts_n_shot is not None:
        n_shot_pct = []
        for count in pos_counts_n_shot:
            # Avoid division by zero; assuming n_shot > 0
            percentage = (count / n_shot) * 100
            n_shot_pct.append(f"{percentage:.2f}%")
        data.append([f"n-shot: {n_shot} - %"] + n_shot_pct)

    print(tabulate(data, headers=headers, tablefmt="grid"))
    
    return list(keep_indices)






def compute_balanced_counts_for_max(counts, max_n_shot):
    """
    Distribute max_n_shot across the classes (length = len(counts)) as evenly as 
    possible, but if a class 'capacity' (counts[i]) is smaller than the fair share, 
    take it all and re-distribute the leftover among remaining classes.
    
    Example:
    --------
    counts = [44026, 34671, 179, 3], max_n_shot=5000
    
    We want:
      - class 2 => 179
      - class 3 => 3
      That leaves 4818 for classes 0 & 1
      => 2409 each
    So final = [2409, 2409, 179, 3]
    """
    # We will store the result here
    result = np.zeros_like(counts, dtype=int)
    
    # Remaining budget
    remaining = max_n_shot
    
    # Indices we haven't "finalized" yet
    remaining_indices = list(range(len(counts)))
    
    while remaining_indices and remaining > 0:
        k = len(remaining_indices)  # how many classes remain
        # Fair share for each remaining class
        share = remaining // k  # integer division
        
        # If share is 0, we can't distribute more fairly (rounding down)
        # We'll just assign them 1 by 1 in a later step or exit
        if share == 0:
            # We'll do a final pass (1-by-1) at the bottom
            break
        
        # We'll hold classes that still can take more than `share`
        # (i.e. classes that can fully accept that share)
        still_remaining = []
        
        for i in remaining_indices:
            capacity_i = counts[i]
            # How many we've assigned so far for class i
            already_assigned = result[i]
            # How many are left for class i
            left_for_class_i = capacity_i - already_assigned
            
            if left_for_class_i <= share:
                # We can fill it completely
                to_take = left_for_class_i
            else:
                # We'll only take the 'share'
                to_take = share
            
            # Assign to_take
            result[i] += to_take
            remaining -= to_take
            
            # If we haven't exhausted this class, keep it in the next round
            # Otherwise, it's "fully used"
            if result[i] < capacity_i and remaining > 0:
                still_remaining.append(i)
        
        # Update remaining_indices for the next loop
        remaining_indices = still_remaining
    
    # If there's still leftover but share was 0 in the last iteration,
    # distribute 1-by-1 until we either exhaust `remaining` or fill everything.
    idx_ptr = 0
    while remaining > 0 and remaining_indices:
        i = remaining_indices[idx_ptr]
        capacity_i = counts[i]
        
        if result[i] < capacity_i:
            result[i] += 1
            remaining -= 1
        
        idx_ptr = (idx_ptr + 1) % len(remaining_indices)
        
        # If some class got fully used, remove it
        if result[i] == capacity_i:
            remaining_indices.remove(i)
            # Move idx_ptr back by 1 to remain aligned after removal
            idx_ptr = idx_ptr % max(len(remaining_indices), 1)
    
    return result


def proportional_subset_indices(hot_encoded_labels, n_shot, max_n_shot=5000, verbose=True):
    """
    Returns a list of indices (balanced proportionally) for the given n_shot.
    
    Args:
        hot_encoded_labels (np.ndarray): One-hot-encoded labels (shape: [num_samples, num_classes])
        n_shot (int): The total number of samples you want in the subset
        max_n_shot (int): The reference n_shot you use to compute the desired proportions.
                          By default, it's 5000 (your 'max' scenario).
    Returns:
        final_indices (list): A list of indices into hot_encoded_labels representing the subset.
    """

    # Convert one-hot encoding to class indices
    class_indices = np.argmax(hot_encoded_labels, axis=1)
    
    # Identify unique classes and their counts
    unique_classes, counts = np.unique(class_indices, return_counts=True)
    n_classes = len(unique_classes)

    # ---------------------------------------------------------------------
    # 1) Figure out how to distribute `max_n_shot` as evenly as possible, 
    #    given the actual capacity for each class (counts[i]).
    # ---------------------------------------------------------------------
    ideal_counts_for_max = compute_balanced_counts_for_max(counts, max_n_shot)

    # ---------------------------------------------------------------------
    # 2) Scale these ideal counts to the desired n_shot
    #    scaled_ideal_count[i] = ideal_counts_for_max[i] * (n_shot / max_n_shot)
    #    Then we round and do a leftover distribution pass.
    # ---------------------------------------------------------------------
    factor = n_shot / max_n_shot
    scaled_floats = ideal_counts_for_max * factor
    scaled_ideal_counts = np.floor(scaled_floats).astype(int)  # initial integer down-round
    
    # We'll handle leftover if there's rounding difference
    assigned = scaled_ideal_counts.sum()
    leftover = n_shot - assigned
    
    # A simple approach: distribute leftover 1-by-1 to classes that still have capacity 
    # (that is, scaled_ideal_counts[i] < counts[i]).
    i = 0
    while leftover > 0:
        idx = i % len(scaled_ideal_counts)
        if scaled_ideal_counts[idx] < counts[idx]:
            scaled_ideal_counts[idx] += 1
            leftover -= 1
        i += 1
        if i > 100000:  # just a safeguard
            break

    # ---------------------------------------------------------------------
    # 3) Now we have the final 'desired' count for each class. Let's sample 
    #    them randomly from the available indices.
    # ---------------------------------------------------------------------
    final_indices = []
    for i, c in enumerate(unique_classes):
        # Indices belonging to class c
        c_indices = np.where(class_indices == c)[0]
        # Randomly choose from them
        chosen_indices = np.random.choice(
            c_indices, size=scaled_ideal_counts[i], replace=False
        )
        final_indices.extend(chosen_indices)
    
    random.shuffle(final_indices)

    # ---------------------------------------------------------------------
    # 4) Print a beautified table summarizing results (if verbose).
    # ---------------------------------------------------------------------
    if verbose:
        headers = [
            "Class",
            "Total Available",
            "Ideal for max_n_shot",
            "Scaled Final for n_shot",
            "% of max_n_shot",
            "% of n_shot",
        ]
        
        data = []
        for i, c in enumerate(unique_classes):
            data.append([
                c,
                counts[i],
                ideal_counts_for_max[i],
                scaled_ideal_counts[i],
                f"{(ideal_counts_for_max[i] / max_n_shot * 100):.2f}%",
                f"{(scaled_ideal_counts[i] / n_shot * 100):.2f}%",
            ])
        
        # Use a grid table format for clarity
        print(tabulate(data, headers=headers, tablefmt="grid"))
        print(f"\nSum of chosen samples = {sum(scaled_ideal_counts)} (target={n_shot}).")
        
        # import pdb; pdb.set_trace()

    # Convert to native Python integers
    final_indices = list(map(int, final_indices))

    return final_indices









def balanced_subset_indices(hot_encoded_labels, n_shot):

    # Convert one-hot encoding to class indices
    class_indices = np.argmax(hot_encoded_labels, axis=1)
    
    # Count how many samples in each class
    unique_classes, counts = np.unique(class_indices, return_counts=True)
    # print(f"unique_classes: {unique_classes}, counts: {counts}")
    unique_classes = unique_classes[np.argsort(counts)]
    n_classes = len(unique_classes)
    
    
    final_indices = []
    for i, c in enumerate(unique_classes):
        
        # Ideal per class
        # Ideally, we want n_shot / n_classes samples from each class if perfectly balanced
        ideal_per_class = (n_shot - len(final_indices)) // (n_classes - i)
        # print(f"nshot: {n_shot - len(final_indices)}, n_classes: {n_classes - i}, ideal_per_class: {ideal_per_class}")

        # Get all indices for class c
        c_indices = np.where(class_indices == c)[0]

        # If class doesn't have enough samples to meet chosen_per_class, take them all
        # (this will still keep things as balanced as possible)
        if len(c_indices) < ideal_per_class:
            selected = c_indices
        else:
            # Randomly choose chosen_per_class samples from this class
            selected = np.random.choice(c_indices, size=ideal_per_class, replace=False)
        
        final_indices.extend(selected.tolist())
        
    random.shuffle(final_indices)
    
    return final_indices


def to_one_hot_lc(y, class_labels=np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])):
    y_classification = np.isin(class_labels, y).astype(np.float32)
    return y_classification

def to_one_hot_building(y):
    mean_value = np.mean(y > 0)
    if mean_value < 0 or mean_value > 1:
        raise ValueError('Invalid values in building mask')

    classes = [mean_value == 0, 0 < mean_value <= 0.3, 0.3 < mean_value <= 0.6, 0.6 < mean_value <= 0.9, mean_value > 0.9]    
    y_classification = np.array([float(x) for x in classes], dtype=np.float32)
    return y_classification

random.seed(97)
np.random.seed(1234) # also affect pandas

REGIONS_DOWNSTREAM_DATA = ['denmark-1', 'denmark-2', 'east-africa', 'egypt-1', 'eq-guinea', 'europe', 'ghana-1',
                           'isreal-1', 'isreal-2', 'japan', 'nigeria', 'north-america', 'senegal', 'south-america',
                           'tanzanipa-1', 'tanzania-2', 'tanzania-3', 'tanzania-4', 'tanzania-5', 'uganda-1']

REGIONS_BUCKETS = {'europe': ['europe','denmark-1','denmark-2'],
                   'east-africa':['east-africa','tanzania-1','tanzania-2','tanzania-3','tanzania-4','tanzania-5','uganda-1'],
                   'northwest-africa':['eq-guinea','ghana-1','egypt-1','isreal-1','isreal-2','nigeria','senegal'],
                   'north-america':['north-america'],
                   'south-america':['south-america'],
                   'japan':['japan']}
# REGIONS_BUCKETS = {'japan':['japan']}

REGIONS = REGIONS_DOWNSTREAM_DATA
LABELS = ['label_roads','label_kg','label_building','label_lc', 'label_coords']


def load_and_crop(file_path, crop_image):
    """ Load a numpy file and crop it to 64x64 from the top-left corner """
    data = np.load(file_path, mmap_mode='r')
    if crop_image:
        data = data[:64, :64] if data.ndim == 3 else data[:, :64, :64, :]  # Handle both individual and batch files
    return data

def sanity_check_labels_exist(x_files, y_files):
    """
    checks that s2 and label numpy files are consistent

    :param x_files:
    :param y_files:
    :return:
    """
    existing_x = []
    existing_y = []
    counter_missing = 0

    assert len(x_files) == len(y_files)
    for x_path, y_path in zip(x_files, y_files):

        exists = os.path.exists(y_path)
        if exists:
            existing_x.append(x_path)
            existing_y.append(y_path)
        else:
            counter_missing += 1

    if counter_missing > 0:
        print(f'WARNING: {counter_missing} label(s) not found')
        missing = [y_f for y_f in y_files if y_f not in existing_y]
        print(f'Showing up to 5 missing files: {missing[:5]}')

    return existing_x, existing_y


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
        # with open('/home/ccollado/1_simulate_data/Major-TOM/train_test_split.json', 'r') as f:
        #     x_test_files = json.load(f)['test']
        # x_test_files = [os.path.join(folder, f_name) for f_name in x_test_files]

        df_simulated = pd.read_csv('/home/ccollado/1_simulate_data/Major-TOM/df_simulation.csv')
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

def get_inferenceset(folder: str,
                regions: list = None,
                y: str = 'building',
                crop_images: bool = False):

    """
    Loads a pre-defined test set data from specified geographic regions.
    :param folder: dataset source folder
    :param regions: geographical regions to sample
    :param y: downstream label from roads, kg, building, lc, coords
    :return: test MultiArrays
    """
    x_test_files = []

    if regions is None:
        regions = REGIONS
    else:
        for r in regions:
            assert r in REGIONS, f"region {r} not found"

    for region in regions:
        # get test samples of region
        x_test_files = x_test_files + sorted(glob(os.path.join(folder, f"{region}*_s2.npy")))
    y_test_files = [f_name.replace('s2', f'label_{y}') for f_name in x_test_files]
    x_test_files, y_test_files = sanity_check_labels_exist(x_test_files, y_test_files)

    x_test = beo.MultiArray([load_and_crop(f, crop_images) for f in x_test_files])
    y_test = beo.MultiArray([load_and_crop(f, crop_images) for f in y_test_files])

    assert len(x_test) == len(y_test), "Lengths of x and y do not match."

    return x_test, y_test

def protocol_minifoundation(folder: str, y:str):
    """
    Loads all the data from the data folder.
    """

    x_train = sorted(glob(os.path.join(folder, f"*/*train_s2.npy")))
    y_train = [f_name.replace('s2', f'label_{y}') for f_name in x_train]

    x_val = []
    y_val = []
    for i in range(int(len(x_train)*0.05)):
        j = random.randint(0, len(x_train)-1)
        x_val.append(x_train[j])
        y_val.append(y_train[j])
        del x_train[j]; del y_train[j]

    x_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_train], shuffle=True)
    y_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_train], shuffle=True)
    x_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_val], shuffle=True)
    y_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_val], shuffle=True)

    return x_train, y_train, x_val, y_val

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
        # with open('/home/ccollado/1_simulate_data/Major-TOM/train_test_split.json', 'r') as f:
        #     x_train_files = json.load(f)['train']

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


def check_region_validity(folder, regions, y):
    # import pdb; pdb.set_trace()
    l = []
    for i, region in enumerate(regions):
        x_train_files = []
        for sub_regions in REGIONS_BUCKETS[region]: 
            x_train_files += sorted(glob(os.path.join(folder, f"{sub_regions}*train_s2.npy")))

        # generate multi array for region
        # x_train_files = sorted(glob(os.path.join(folder, f"{region}*train_s2.npy")))
        y_train_files = [f_name.replace('s2', f'label_{y}') for f_name in x_train_files]

        # checks that s2 and label numpy files are consistent
        x_train_files, y_train_files = sanity_check_labels_exist(x_train_files, y_train_files)
        if x_train_files:
            l.append(region)

    # import pdb; pdb.set_trace()
    return l


def protocol_fewshot(folder: str,
                     dst: str,
                     n: int = 10,
                     val_ratio: float = 0.2,
                     regions: list = None,
                     y: str = 'building',
                     resample: bool = False,
                     ):

    """
    Loads n-samples data from specified geographic regions.
    :param folder: dataset source folder
    :param dst: save folder
    :param n: number of samples
    :param val_ratio: ratio of validation set
    :param regions: geographical regions to sample
    :param y: downstream label from roads, kg, building, lc, coords
    :return: train, val MultiArrays
    """
    if os. path. exists(f'{dst}/{n}_shot_{y}/{n}shot_train_s2.npy'):
        train_X_temp = np.load(f'{dst}/{n}_shot_{y}/{n}shot_train_s2.npy', mmap_mode='r')
        train_y_temp = np.load(f'{dst}/{n}_shot_{y}/{n}shot_train_label_{y}.npy', mmap_mode='r')
        val_X_temp = np.load(f'{dst}/{n}_shot_{y}/{n}shot_val_s2.npy', mmap_mode='r')
        val_y_temp = np.load(f'{dst}/{n}_shot_{y}/{n}shot_val_label_{y}.npy', mmap_mode='r')
    else:

        if regions is None:
            regions = list(REGIONS_BUCKETS.keys())
        else:
            for r in regions:
                assert r in list(REGIONS_BUCKETS.keys()), f"region {r} not found. Possible regions are {list(REGIONS_BUCKETS.keys())}"
        regions = check_region_validity(folder, regions, y)

        f_x = glob(os.path.join(folder, f"{regions[0]}*test_s2.npy"))[0]
        ref_x = np.load(f_x, mmap_mode='r')
        f_y = glob(os.path.join(folder, f"{regions[0]}*test_label_{y}.npy"))[0]
        ref_y = np.load(f_y, mmap_mode='r')

        d_size = n*len(regions)
        d_size_val = int(np.ceil(n*val_ratio)*len(regions))

        train_X_temp = np.zeros_like(a=ref_x, shape=(d_size, ref_x.shape[1], ref_x.shape[2], ref_x.shape[3]))
        val_X_temp = np.zeros_like(a=ref_x, shape=(d_size_val, ref_x.shape[1], ref_x.shape[2], ref_x.shape[3]))
        train_y_temp = np.zeros_like(a=ref_y, shape=(d_size, ref_y.shape[1], ref_y.shape[2], ref_y.shape[3]))
        val_y_temp = np.zeros_like(a=ref_y, shape=(d_size_val, ref_y.shape[1], ref_y.shape[2], ref_y.shape[3]))
        del ref_x ; del ref_y

        for i, region in enumerate(regions):
            # generate multi array for region
            x_train_files = []
            for sub_regions in REGIONS_BUCKETS[region]: 
                x_train_files += sorted(glob(os.path.join(folder, f"{sub_regions}*train_s2.npy")))
            y_train_files = [f_name.replace('s2', f'label_{y}') for f_name in x_train_files]
            x_val_files = [f_name.replace('train', 'val') for f_name in x_train_files]
            y_val_files = [f_name.replace('train', 'val') for f_name in y_train_files]

            # checks that s2 and label numpy files are consistent
            x_train_files, y_train_files = sanity_check_labels_exist(x_train_files, y_train_files)
            x_val_files, y_val_files = sanity_check_labels_exist(x_val_files, y_val_files)

            x_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_train_files])
            y_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_train_files])
            x_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_val_files])
            y_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_val_files])

            if n < len(x_train):
                train_indexes = random.sample(range(0, len(x_train)), n)

                for j, idx in enumerate(train_indexes):
                    train_X_temp[(n*i)+j] = x_train[idx]
                    train_y_temp[(n * i) + j] = y_train[idx]

            else:
                # resample if n > than regions number of samples
                for j in range(0, len(x_train)):
                    train_X_temp[(n * i)+j] = x_train[j]
                    train_y_temp[(n * i)+j] = y_train[j]

                if resample:
                    train_indexes = random.choices(range(0, len(x_train)), k=(n - len(x_train)))
                    for j, idx in enumerate(train_indexes):
                        train_X_temp[(n * i)+len(x_train)+j] = x_train[idx]
                        train_y_temp[(n * i)+len(x_train) + j] = y_train[idx]

            if int(np.ceil(n * val_ratio)) < len(x_val):

                val_indexes = random.sample(range(0, len(x_val)), int(np.ceil(n * val_ratio)))

                for j, idx in enumerate(val_indexes):
                    val_X_temp[(int(np.ceil(n * val_ratio)) * i) + j] = x_val[idx]
                    val_y_temp[(int(np.ceil(n * val_ratio)) * i) + j] = y_val[idx]

            else:
                # resample if n > than regions number of samples
                for j in range(0, len(x_val)):
                    val_X_temp[(int(np.ceil(n * val_ratio)))+j] = x_val[j]
                    val_y_temp[(int(np.ceil(n * val_ratio)))+j] = y_val[j]
                if resample:
                    val_indexes = random.choices(range(0, len(x_val)), k=((int(np.ceil(n * val_ratio))) - len(x_val)))
                    for j, idx in enumerate(val_indexes):
                        val_X_temp[(int(np.ceil(n * val_ratio)))+len(x_val)+j] = x_val[idx]
                        val_y_temp[(int(np.ceil(n * val_ratio)))+len(x_val) + j] = y_val[idx]

            del x_train; del y_train; del x_val; del y_val

        os.makedirs(f'{dst}/{n}_shot_{y}', exist_ok=True)
        np.save(f'{dst}/{n}_shot_{y}/{n}shot_train_s2.npy', train_X_temp)
        np.save(f'{dst}/{n}_shot_{y}/{n}shot_train_label_{y}.npy', train_y_temp)
        np.save(f'{dst}/{n}_shot_{y}/{n}shot_val_s2.npy', val_X_temp)
        np.save(f'{dst}/{n}_shot_{y}/{n}shot_val_label_{y}.npy', val_y_temp)
    return train_X_temp, train_y_temp, val_X_temp, val_y_temp


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

                # train_indices = balanced_subset_indices(hot_encoded_train, n_train_samples)
                # val_indices = balanced_subset_indices(hot_encoded_val, n_val_samples)
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


def sanity_check_labels_exist_fm(x_files, y_files):
    """
    Checks that s2 files and label files (for multiple tasks) are consistent.

    :param x_files: list of x file paths
    :param y_files: dict of {task: list of y file paths}
    :return: (existing_x, existing_y)
        existing_x: list of x files that have all corresponding y files
        existing_y: dict of {task: list of y files} corresponding to existing_x
    """

    # Ensure each task in y_files has the same length as x_files
    for task, y_list in y_files.items():
        assert len(y_list) == len(x_files), f"Number of label files for task '{task}' does not match number of x_files."

    existing_x = []
    existing_y = {task: [] for task in y_files.keys()}
    counter_missing = 0

    # Iterate through each x_file and corresponding y_files
    for i, x_path in enumerate(x_files):
        y_paths_for_x = {task: y_files[task][i] for task in y_files.keys()}
        
        # Check existence of all label files for this x_file
        all_exist = True
        for task, y_path in y_paths_for_x.items():
            if not os.path.exists(y_path):
                all_exist = False
                counter_missing += 1
                break

        # If all files exist, append them to the existing lists
        if all_exist:
            existing_x.append(x_path)
            for task, y_path in y_paths_for_x.items():
                existing_y[task].append(y_path)

    # Print warnings if any missing labels were detected
    if counter_missing > 0:
        print(f'WARNING: {counter_missing} label(s) not found.')
        # Collect missing files (up to 5)
        missing = []
        for task, y_list in y_files.items():
            for y_path in y_list:
                # If the y_path was never added to existing_y, it's missing
                if y_path not in existing_y[task]:
                    missing.append(y_path)
        print(f'Showing up to 5 missing files: {missing[:5]}')

    return existing_x, existing_y


def protocol_split_fm(folder: str,
                      y_supervised: list = ['coords', 'climate'],
                      split_percentage: float = 0.1,
                      rank: int = 0,
                      use_ddp: bool = False,
                      ):
    """
    Loads a percentage of the data from specified geographic regions.
    :param folder: dataset source folder
    :param split_percentage: percentage of data to sample from each region
    :param regions: geographical regions to sample
    :param y: downstream label from roads, kg, building, lc, coords
    :return: train, val MultiArrays
    """

    assert 0 < split_percentage <= 1, "split percentage out of range (0 - 1)"

    # with open('/home/ccollado/1_simulate_data/Major-TOM/train_test_split.json', 'r') as f:
    #     x_train_files = json.load(f)['train']

    df_simulated = pd.read_csv('other/df_simulation.csv')
    df = df_simulated['unique_identifier']

    folder_files = os.listdir(folder)

    def get_split_files_ddp(folder, split='train'):
        
        # Optimized filtering of files
        split_tag = f'_{split}'
        split_files_full_name = [f for f in folder_files if 's2' in f and split_tag in f]
        base_names = [f.split(split_tag)[0] for f in split_files_full_name]
        
        # Use set for faster membership tests
        base_set = set(df)
        matched_files_full_name = [fname for fname, bname in zip(split_files_full_name, base_names) if bname in base_set]
        
        # Generate file paths once
        x_split_files = [os.path.join(folder, fname) for fname in matched_files_full_name]
        
        N = len(x_split_files)
        K = int(N * split_percentage)
        if rank == 0:
            print(f"{split} split: taking {K} files out of {N} in total")

        # Random sampling indices
        indices = np.random.choice(N, size=K, replace=False)
        
        # Select the rows using the indices
        selected_files = [x_split_files[i] for i in sorted(indices)]
        
        y_split_files = {}
        for task in y_supervised:
            y_split_files[task] = [fname.replace('s2', f'label_{task}') for fname in selected_files]
        
        return selected_files, y_split_files

    x_train_files, y_train_files = get_split_files_ddp(folder, split='train')
    
    # VALIDATION DATA
    if (os.path.basename(x_train_files[0])).replace('train', 'val') not in folder_files:
        if rank == 0:
            print("Validation data from different tiff files")
        x_val_files, y_val_files = get_split_files_ddp(folder, split='val')
    else:
        if rank == 0:
            print("Validation data from same tiff files as train data")
        x_val_files = [f_name.replace('train', 'val') for f_name in x_train_files]
        y_val_files = {}
        for task in y_supervised:
            y_val_files[task] = [f_name.replace('train', 'val') for f_name in y_train_files[task]]

    # import pdb; pdb.set_trace()

    # checks that s2 and label numpy files are consistent
    x_train_files, y_train_files = sanity_check_labels_exist_fm(x_train_files, y_train_files)
    x_val_files, y_val_files = sanity_check_labels_exist_fm(x_val_files, y_val_files)

    if True:
        return x_train_files, y_train_files, x_val_files, y_val_files

    x_train = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_train_files])
    x_val = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_val_files])

    y_train = {}
    y_val = {}
    for task in y_supervised:
        y_train[task] = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_train_files[task]])
        y_val[task] = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_val_files[task]])
    
    for task in y_supervised:
        assert len(x_train) == len(y_train[task]), f"Lengths of x and y for task {task} do not match in training set. len(x_train)={len(x_train)}, len(y_train)={len(y_train[task])}"
        assert len(x_val) == len(y_val[task]), f"Lengths of x and y for task {task} do not match in validation set. len(x_val)={len(x_val)}, len(y_val)={len(y_val[task])}"

    return x_train, y_train, x_val, y_val



def get_testset_fm(folder: str,
                   y_supervised: list = ['coords', 'climate'],
                   use_ddp=False):

    """
    Loads a pre-defined test set data from specified geographic regions.
    :param folder: dataset source folder
    :param y: downstream label from roads, kg, building, lc, coords
    :return: test MultiArrays
    """
    test_files = os.listdir(folder)

    split_tag = '_test'
    split_files_full_name = [f for f in test_files if 's2' in f and split_tag in f]
    base_names = [f.split(split_tag)[0] for f in split_files_full_name]
    
    # Use set for faster membership tests
    df_simulated = pd.read_csv('other/df_simulation.csv')
    df = df_simulated['unique_identifier']
    base_set = set(df)
    matched_files_full_name = [fname for fname, bname in zip(split_files_full_name, base_names) if bname in base_set]
    
    # Generate file paths once
    x_test_files = [os.path.join(folder, f_name) for f_name in matched_files_full_name]

    y_test_files = {}
    for task in y_supervised:
        y_test_files[task] = [f_name.replace('s2', f'label_{task}') for f_name in x_test_files]
    
    x_test_files, y_test_files = sanity_check_labels_exist_fm(x_test_files, y_test_files)

    if True:
        return x_test_files, y_test_files

    x_test = beo.MultiArray([np.load(f, mmap_mode='r') for f in x_test_files])
    y_test = {}
    for task in y_supervised:
        y_test[task] = beo.MultiArray([np.load(f, mmap_mode='r') for f in y_test_files[task]])

    for task in y_supervised:
        assert len(x_test) == len(y_test[task]), f"Lengths of x and y for task {task} do not match in test set. len(x_test)={len(x_test)}, len(y_test)={len(y_test[task])}"

    return x_test, y_test



if __name__ == '__main__':
    label =['roads', 'building', 'lc']
    n_shots = [1, 2, 5, 10, 50, 100, 150, 200, 500, 750, 1000]
    for l in label:
        for n in n_shots:
            x_train, y_train, x_val, y_val = protocol_fewshot('/phileo_data/downstream/downstream_dataset_patches_np/',
                                                              dst='/phileo_data/downstream/downstream_datasets_nshot/',
                                                              n=n,
                                                              y=l)