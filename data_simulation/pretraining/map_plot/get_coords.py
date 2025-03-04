import os
import lmdb
import numpy as np
import csv
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def read_chunk(lmdb_path, start_idx, end_idx):
    """
    Reads records from index `start_idx` to `end_idx` (exclusive),
    returning a list of rows (each row: [i, coord_0, coord_1, ...]).
    """
    rows = []
    # Open a new LMDB environment in each process
    with lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False) as env:
        with env.begin(write=False) as txn:
            for i in range(start_idx, end_idx):
                coords_data = txn.get(f"{i:08d}_coords".encode("ascii"))
                if coords_data is None:
                    # skip if missing
                    continue
                y_coords = np.frombuffer(coords_data, dtype=np.float64)
                coords_list = y_coords.tolist()
                row = [i] + coords_list
                rows.append(row)
    return rows

def export_coords_to_csv_parallel(lmdb_path, csv_output_path, num_workers=None, chunk_size=10000):
    """
    Parallelized export of coordinates to CSV using multiple processes.
    
    Args:
        lmdb_path (str): Path to the LMDB environment.
        csv_output_path (str): Output CSV file path.
        num_workers (int, optional): Number of processes to use in the pool.
            If None, defaults to the number of CPUs on the system.
        chunk_size (int): How many records each process will handle at a time.
    """
    if num_workers is None:
        # Default to number of CPU cores
        import multiprocessing
        num_workers = multiprocessing.cpu_count()

    # First, figure out how many records we have
    with lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False) as env:
        with env.begin(write=False) as txn:
            length = txn.stat()["entries"] // 3
    
    print(f"Total records: {length}")
    print(f"Using {num_workers} processes, chunk_size={chunk_size}")

    # Prepare chunk boundaries
    chunk_ranges = []
    for start_idx in range(0, length, chunk_size):
        end_idx = min(start_idx + chunk_size, length)
        chunk_ranges.append((start_idx, end_idx))

    # We'll write to CSV after collecting data, to keep order consistent
    with open(csv_output_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # Write a header if desired (adjust to your actual coordinate schema)
        writer.writerow(["index", "coord_0", "coord_1", "..."])

        # Launch parallel jobs
        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Keep track of which future corresponds to which chunk index
            future_to_chunk_idx = {}
            for chunk_idx, (start_idx, end_idx) in enumerate(chunk_ranges):
                f = executor.submit(read_chunk, lmdb_path, start_idx, end_idx)
                future_to_chunk_idx[f] = chunk_idx

            # We'll store rows in a list so we can write them in ascending chunk order
            chunk_output_storage = [None] * len(chunk_ranges)

            # Use tqdm to show progress over the total number of chunks
            for future in tqdm(as_completed(future_to_chunk_idx), total=len(future_to_chunk_idx), desc="Processing Chunks"):
                chunk_idx = future_to_chunk_idx[future]
                rows = future.result()
                chunk_output_storage[chunk_idx] = rows

            # Now write them in order (chunk 0, chunk 1, etc.)
            for chunk_rows in chunk_output_storage:
                writer.writerows(chunk_rows)

    print(f"Finished writing CSV to {csv_output_path}")

if __name__ == "__main__":
    lmdb_path_train = "/home/phimultigpu/phisat2_foundation/lmdb_patches_256_1536/train.lmdb"
    csv_output_path = "train_coords.csv"
    
    # Adjust num_workers and chunk_size if desired
    export_coords_to_csv_parallel(
        lmdb_path=lmdb_path_train,
        csv_output_path=csv_output_path,
        num_workers=None,       # or set to an integer, e.g. 8
        chunk_size=10000        # tune for your memory constraints / data size
    )
