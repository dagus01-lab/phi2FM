import lmdb
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

lmdb_path = "/home/phimultigpu/phisat2_foundation/lmdb_patches_256_1536/train.lmdb"

# Global parameters
NUM_CHANNELS = 8
HEIGHT = 256
WIDTH = 256
NUM_PIXELS = HEIGHT * WIDTH

def compute_partial_stats(args):
    """
    Worker function to compute partial statistics of a range of LMDB entries,
    displaying an individual progress bar for each chunk.
    """
    start_idx, end_idx, db_path, chunk_id = args
    total = end_idx - start_idx

    # Local accumulators
    min_vals = np.full((NUM_CHANNELS,), np.inf, dtype=np.float64)
    max_vals = np.full((NUM_CHANNELS,), -np.inf, dtype=np.float64)
    sum_vals = np.zeros((NUM_CHANNELS,), dtype=np.float64)
    sum_sq_vals = np.zeros((NUM_CHANNELS,), dtype=np.float64)

    # Use position=chunk_id to (attempt to) keep each bar on its own line
    pbar = tqdm(total=total, desc=f"Chunk {chunk_id:02d}", position=chunk_id, leave=True)

    # Open LMDB in the worker
    env = lmdb.open(db_path, readonly=True, lock=False, readahead=False)
    with env.begin(write=False) as txn:
        for idx in range(start_idx, end_idx):
            pbar.update(1)
            image_key = f"{idx:08d}_image".encode("ascii")
            image_data = txn.get(image_key)
            if image_data is None:
                continue
            
            # Convert bytes to ndarray of shape (8, 256, 256)
            x = np.frombuffer(image_data, dtype=np.uint16).reshape(NUM_CHANNELS, HEIGHT, WIDTH)
            x = np.clip(x, 0, 10000)
            x = np.sqrt(x)

            # Update local min/max
            channel_mins = x.min(axis=(1, 2))
            channel_maxs = x.max(axis=(1, 2))
            min_vals = np.minimum(min_vals, channel_mins)
            max_vals = np.maximum(max_vals, channel_maxs)

            # Update sums and sums of squares
            channel_sums = x.sum(axis=(1, 2), dtype=np.float64)
            channel_sums_sq = np.square(x, dtype=np.float64).sum(axis=(1, 2))

            sum_vals += channel_sums
            sum_sq_vals += channel_sums_sq

    pbar.close()
    env.close()

    return min_vals, max_vals, sum_vals, sum_sq_vals

def main():
    # Get total number of records
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    with env.begin(write=False) as txn:
        total_entries = txn.stat()["entries"]
        total_records = total_entries // 3  # each sample has 3 keys
    env.close()

    print(f"Total records: {total_records}")

    # Decide how many processes to use
    num_processes = min(24, cpu_count())
    chunk_size = (total_records + num_processes - 1) // num_processes

    # Create tasks (start_idx, end_idx, lmdb_path, chunk_id)
    tasks = []
    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_records)
        if start_idx >= end_idx:
            break
        tasks.append((start_idx, end_idx, lmdb_path, i))

    print(f"Spawning {len(tasks)} processes ...")

    # Launch parallel processes
    # We do not wrap this in an extra tqdm because each worker shows its own progress bar
    with Pool(processes=len(tasks)) as p:
        results = p.map(compute_partial_stats, tasks)

    # Combine partial results
    global_min = np.full((NUM_CHANNELS,), np.inf, dtype=np.float64)
    global_max = np.full((NUM_CHANNELS,), -np.inf, dtype=np.float64)
    global_sum = np.zeros((NUM_CHANNELS,), dtype=np.float64)
    global_sum_sq = np.zeros((NUM_CHANNELS,), dtype=np.float64)

    for (min_vals, max_vals, sum_vals, sum_sq_vals) in results:
        global_min = np.minimum(global_min, min_vals)
        global_max = np.maximum(global_max, max_vals)
        global_sum += sum_vals
        global_sum_sq += sum_sq_vals

    # Final mean/std
    total_pixels = total_records * NUM_PIXELS
    means = global_sum / total_pixels
    variances = (global_sum_sq / total_pixels) - (means ** 2)
    stds = np.sqrt(np.maximum(variances, 0.0))

    # Print final stats
    print("\n=== Final per-channel stats ===")
    print("Min:", global_min)
    print("Max:", global_max)
    print("Mean:", means)
    print("Std: ", stds)

if __name__ == "__main__":
    main()
