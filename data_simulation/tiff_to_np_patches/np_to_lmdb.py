import os
import re
import lmdb
import pickle
import numpy as np
from tqdm import tqdm
import multiprocessing

# Adjust this path to where your .npy files are stored
DATA_DIR = "/home/ccollado/phileo_phisat2/MajorTOM/np_patches_256_crop_1536_ind"

# Output folder for separate LMDB files
LMDB_OUTPUT_DIR = "/home/ccollado/phileo_phisat2/MajorTOM/lmdb_patches_256_1536"
os.makedirs(LMDB_OUTPUT_DIR, exist_ok=True)

# Regex to identify the three file types
pattern_climate = re.compile(r'^(.*?)(label_climate)_(\d+)\.npy$')
pattern_coords  = re.compile(r'^(.*?)(label_coords)_(\d+)\.npy$')
pattern_s2      = re.compile(r'^(.*?)(s2)_(\d+)\.npy$')

def store_file_in_map(fname, kind, prefix, index, file_map):
    """Helper to store a file in the file_map structure."""
    key = (prefix, index)
    if key not in file_map:
        file_map[key] = {"climate": None, "coords": None, "s2": None}
    file_map[key][kind] = os.path.join(DATA_DIR, fname)

def build_file_map():
    """Scan DATA_DIR for .npy files, group them into (prefix, index)."""
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".npy")]
    file_map = {}

    for fname in all_files:
        m_climate = pattern_climate.match(fname)
        m_coords  = pattern_coords.match(fname)
        m_s2      = pattern_s2.match(fname)

        if m_climate:
            prefix, kind, idx_str = m_climate.groups()  
            store_file_in_map(fname, "climate", prefix, idx_str, file_map)
        elif m_coords:
            prefix, kind, idx_str = m_coords.groups()
            store_file_in_map(fname, "coords", prefix, idx_str, file_map)
        elif m_s2:
            prefix, kind, idx_str = m_s2.groups()
            store_file_in_map(fname, "s2", prefix, idx_str, file_map)
        else:
            pass

    return file_map

def majority_class_label(climate_np):
    """Given a (256,256,1) climate mask, return the majority class as a single int."""
    climate_flat = climate_np.reshape(-1)
    freq = np.bincount(climate_flat)
    return np.argmax(freq).astype(np.uint8)

def write_lmdb(records, lmdb_path, map_size=1 << 38, commit_interval=1000):
    env = lmdb.open(lmdb_path, map_size=map_size)
    txn = env.begin(write=True)
    idx = 0

    for prefix, index_str, paths in tqdm(records, desc=f"Building {os.path.basename(lmdb_path)}"):
        if not all(paths[k] for k in ["climate", "coords", "s2"]):
            continue

        try:
            image_np    = np.load(paths["s2"])         # (256,256,8) uint16
            climate_np  = np.load(paths["climate"])    # (256,256,1) uint8
            coords_np   = np.load(paths["coords"])     # (4,) float64

            climate_label = majority_class_label(climate_np)
            image_np_label = image_np.transpose(2, 0, 1)  # (8,256,256) uint16

            key_prefix = f"{idx:08d}"
            txn.put(f"{key_prefix}_image".encode("ascii"), image_np_label.tobytes())
            txn.put(f"{key_prefix}_coords".encode("ascii"), coords_np.tobytes())
            txn.put(f"{key_prefix}_climate".encode("ascii"), np.array([climate_label], dtype=np.uint8).tobytes())

            if (idx + 1) % commit_interval == 0:  # commit_interval reached
                txn.commit()           # commit current transaction
                txn = env.begin(write=True)  # start a new transaction

            idx += 1
        except Exception as e:
            print(f"Error processing {prefix}_{index_str}: {e}")

    # Commit any remaining transactions after loop
    txn.commit()
    env.close()

def main():
    print("Building file map...")
    fm = build_file_map()

    train_records = []
    val_records   = []
    test_records  = []

    for (prefix, index_str), paths in fm.items():
        if "_train" in prefix:
            train_records.append((prefix, index_str, paths))
        elif "_val" in prefix:
            val_records.append((prefix, index_str, paths))
        elif "_test" in prefix:
            test_records.append((prefix, index_str, paths))

    print(f"Train samples: {len(train_records)}")
    print(f"Val samples:   {len(val_records)}")
    print(f"Test samples:  {len(test_records)}")

    train_lmdb = os.path.join(LMDB_OUTPUT_DIR, "train.lmdb")
    val_lmdb   = os.path.join(LMDB_OUTPUT_DIR, "val.lmdb")
    test_lmdb  = os.path.join(LMDB_OUTPUT_DIR, "test.lmdb")

    # write_lmdb(train_records, train_lmdb, 1 << 39)  # Train set
    write_lmdb(val_records, val_lmdb, 1 << 38)      # Validation set
    write_lmdb(test_records, test_lmdb, 1 << 38)    # Test set

    print("All LMDBs created successfully.")

if __name__ == "__main__":
    main()
