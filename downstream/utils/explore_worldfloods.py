import zarr
import numpy as np

def compute_band_stats(zarr_path, group='trainval', img_key='img'):
    """
    Computes mean and stddev for each band in a Zarr dataset.
    Assumes images are stored under group/img_key with shape [bands, height, width] or [height, width, bands].
    """
    # Open Zarr group
    z = zarr.open(zarr_path, mode='r')
    if group not in z:
        raise ValueError(f"Group '{group}' not found in Zarr archive.")
    g = z[group]

    # Collect all sample keys
    sample_keys = [k for k in g.array_keys() if img_key in k or k == img_key]
    if not sample_keys:
        sample_keys = [k for k in g.group_keys()]
    if not sample_keys:
        raise ValueError(f"No image arrays found under group '{group}'.")

    # Accumulate stats
    band_sum = None
    band_sum_sq = None
    n_pixels = 0

    for sid in g.group_keys():
        arr = g[sid][img_key][:]
        label = g[sid]['label'][:]
        print(f"Sample {sid}: number of zeros: {(label == 0).sum()}, number of nans: {np.isnan(arr).sum()}")
        if np.isnan(arr).any():
            print(f"Sample {sid}: min={np.nanmin(arr)}, max={np.nanmax(arr)}, nan_count={np.isnan(arr).sum()}")
            continue
        # Ensure shape is [bands, height, width]
        if arr.ndim == 3:
            if arr.shape[0] <= arr.shape[-1]:  # [bands, h, w]
                arr = arr
            else:  # [h, w, bands]
                arr = np.transpose(arr, (2, 0, 1))
        else:
            continue  # skip non-3D arrays

        bands, h, w = arr.shape
        arr_reshaped = arr.reshape(bands, -1)  # [bands, pixels]
        if band_sum is None:
            band_sum = np.zeros(bands, dtype=np.float64)
            band_sum_sq = np.zeros(bands, dtype=np.float64)
        band_sum += arr_reshaped.sum(axis=1)
        band_sum_sq += (arr_reshaped ** 2).sum(axis=1)
        n_pixels += arr_reshaped.shape[1]

    mean = band_sum / n_pixels
    std = np.sqrt(band_sum_sq / n_pixels - mean ** 2)

    print("Band statistics:")
    print(f"Means per band: {mean}")
    print(f"Stds per band: {std}")
    #for i, (m, s) in enumerate(zip(mean, std)):
    #    print(f"Band {i}: mean={m:.5f}, std={s:.5f}")

    return mean, std

if __name__ == "__main__":
    import sys
    zarr_path = sys.argv[1] if len(sys.argv) > 1 else "/Data/anomaly_detection/marine_area_dataset.zarr"
    compute_band_stats(zarr_path)