import os
import glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Define input and output directories
input_dir = "/home/ccollado/phileo_phisat2/MajorTOM/np_patches_256_crop_1536"
output_dir = "/home/ccollado/phileo_phisat2/MajorTOM/np_patches_256_crop_1536_ind"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

def process_file(file_path):
    """
    Load a .npy file, split it along the first dimension,
    and save each slice to the output directory.
    """
    arr = np.load(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    num_slices = arr.shape[0]
    
    for i in range(num_slices):
        slice_arr = arr[i]  # Slice along the first dimension
        out_name = f"{base_name}_{i}.npy"
        out_path = os.path.join(output_dir, out_name)
        np.save(out_path, slice_arr)
    
    return f"Processed {base_name} into {num_slices} slices."

def main():
    # Gather all .npy files from the input directory
    file_list = glob.glob(os.path.join(input_dir, "*.npy"))
    
    # Use a ProcessPoolExecutor to process files in parallel
    with ProcessPoolExecutor() as executor:
        # Submit tasks for all files
        futures = {executor.submit(process_file, file_path): file_path for file_path in file_list}
        
        # Wrap the as_completed iterator with tqdm for progress indication
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Files"):
            try:
                result = future.result()
                # Optionally print the result for logging
                # print(result)
            except Exception as exc:
                print(f"File {futures[future]} generated an exception: {exc}")

if __name__ == "__main__":
    main()
