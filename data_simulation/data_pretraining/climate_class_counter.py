import lmdb
import numpy as np
from tqdm import tqdm

lmdb_path = '/home/ccollado/phileo_phisat2/MajorTOM/lmdb_patches_256_1536/train.lmdb'

# Open the LMDB environment in read-only mode
with lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False) as env:
    with env.begin(write=False) as txn:
        # Calculate the number of records (3 keys per record)
        num_records = txn.stat()["entries"] // 3
        
        # Initialize an array for counting classes 0 to 30
        counts = np.zeros(31, dtype=int)
        
        for idx in tqdm(range(num_records), desc="Counting climate labels"):
            key = f"{idx:08d}_climate".encode("ascii")
            climate_data = txn.get(key)
            if climate_data is None:
                continue

            # Deserialize the climate data; it contains one value as a uint8
            climate_label = np.frombuffer(climate_data, dtype=np.uint8)[0]
            counts[climate_label] += 1

print("Climate label counts:", counts)
