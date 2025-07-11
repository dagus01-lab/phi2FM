# Standard Library
import os
from glob import glob

# External Libraries
import numpy as np
import random

from collections import defaultdict
import numpy as np
import random
from skmultilearn.problem_transform import LabelPowerset
from tabulate import tabulate


REGIONS_DOWNSTREAM_DATA = ['denmark-1', 'denmark-2', 'east-africa', 'egypt-1', 'eq-guinea', 'europe', 'ghana-1',
                           'isreal-1', 'isreal-2', 'japan', 'nigeria', 'north-america', 'senegal', 'south-america',
                           'tanzanipa-1', 'tanzania-2', 'tanzania-3', 'tanzania-4', 'tanzania-5', 'uganda-1']

REGIONS_BUCKETS = {'europe': ['europe','denmark-1','denmark-2'],
                   'east-africa':['east-africa','tanzania-1','tanzania-2','tanzania-3','tanzania-4','tanzania-5','uganda-1'],
                   'northwest-africa':['eq-guinea','ghana-1','egypt-1','isreal-1','isreal-2','nigeria','senegal'],
                   'north-america':['north-america'],
                   'south-america':['south-america'],
                   'japan':['japan']}

REGIONS = REGIONS_DOWNSTREAM_DATA
LABELS = ['label_roads','label_kg','label_building','label_lc', 'label_coords']


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


def load_and_crop(file_path, crop_image):
    """ Load a numpy file and crop it to 64x64 from the top-left corner """
    data = np.load(file_path, mmap_mode='r')
    if crop_image and data.ndim >= 3:
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


