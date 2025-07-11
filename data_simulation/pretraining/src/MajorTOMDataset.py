import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import rasterio as rio
from PIL import Image
import torchvision.transforms as transforms
from rasterio.enums import Resampling
import numpy as np
from tqdm import tqdm

class MajorTOM(Dataset):
    """MajorTOM Dataset (https://huggingface.co/Major-TOM)

    Args:
        df ((geo)pandas.DataFrame): Metadata dataframe
        local_dir (string): Root directory of the local dataset version
        tif_bands (list): A list of tif file names to be read
        png_bands (list): A list of png file names to be read
        combine_bands (bool): If True, combines specified bands into a single array
        resample (bool): If True, resamples lower-resolution bands to match the highest resolution

    """

    def __init__(self,
                 df,
                 local_dir=None,
                 tif_bands=['B04', 'B03', 'B02'],
                 png_bands=['thumbnail'],
                 tif_transforms=None,
                 png_transforms=None,
                 combine_bands=False,
                 resample=True
                 ):
        super().__init__()
        self.df = df
        self.local_dir = Path(local_dir) if isinstance(local_dir, str) else local_dir
        self.tif_bands = tif_bands if not isinstance(tif_bands, str) else [tif_bands]
        self.png_bands = png_bands if not isinstance(png_bands, str) else [png_bands]
        self.tif_transforms = tif_transforms
        self.png_transforms = png_transforms
        self.combine_bands = combine_bands
        self.resample = resample
        self.bands_order = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
        self.bands_to_combine = [band for band in self.bands_order if band in self.tif_bands]
        self.band_resolutions = {
            'B01': 60,
            'B02': 10,
            'B03': 10,
            'B04': 10,
            'B05': 20,
            'B06': 20,
            'B07': 20,
            'B08': 10,
            'B8A': 20,
            'B09': 60,
            'B10': 60,
            'B11': 20,
            'B12': 20
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        meta = self.df.iloc[idx]

        product_id = meta.product_id
        grid_cell = meta.grid_cell
        row = grid_cell.split('_')[0]

        path = self.local_dir / Path("{}/{}/{}".format(row, grid_cell, product_id))
        out_dict = {'meta': meta}

        if self.combine_bands:
            band_arrays = []
            # First, get the target shape by reading one of the high-resolution bands
            high_res_bands = [band for band in self.bands_to_combine if self.band_resolutions[band] == 10]
            if not high_res_bands:
                raise ValueError("No high-resolution bands found to determine target shape.")
            high_res_band = high_res_bands[0]

            with rio.open(path / '{}.tif'.format(high_res_band)) as f:
                # Read data as (H, W)
                out = f.read(1)
                target_shape = out.shape  # (H, W)
                band_arrays.append(out)

            # Now read the rest of the bands
            for band in self.bands_to_combine:
                if band == high_res_band:
                    continue  # Already read
                with rio.open(path / '{}.tif'.format(band)) as f:
                    if self.resample and self.band_resolutions[band] != 10:
                        # Resample to target shape
                        out = f.read(
                            1,
                            out_shape=target_shape,
                            resampling=Resampling.bilinear)
                    else:
                        out = f.read(1)
                    band_arrays.append(out)

            # Stack bands along axis 0 to get array of shape (B, H, W)
            bands_array = np.stack(band_arrays, axis=0)
            # Convert to torch tensor
            bands_tensor = torch.from_numpy(bands_array.astype(np.float32))
            # Apply transforms if any
            if self.tif_transforms is not None:
                bands_tensor = self.tif_transforms(bands_tensor)
            out_dict['bands'] = bands_tensor

            # Process remaining tif bands (e.g., 'cloud_mask')
            remaining_tif_bands = [band for band in self.tif_bands if band not in self.bands_to_combine]
            for band in remaining_tif_bands:
                with rio.open(path / '{}.tif'.format(band)) as f:
                    out = f.read(1)
                out_tensor = torch.from_numpy(out.astype(np.float32))
                if self.tif_transforms is not None:
                    out_tensor = self.tif_transforms(out_tensor)
                out_dict[band] = out_tensor
        else:
            for band in self.tif_bands:
                with rio.open(path / '{}.tif'.format(band)) as f:
                    out = f.read(1)
                out_tensor = torch.from_numpy(out.astype(np.float32))
                if self.tif_transforms is not None:
                    out_tensor = self.tif_transforms(out_tensor)
                out_dict[band] = out_tensor

        for band in self.png_bands:
            out = Image.open(path / '{}.png'.format(band))
            if self.png_transforms is not None:
                out = self.png_transforms(out)
            out_dict[band] = out

        return out_dict

    def exists(self, idx):
        """
        Check if all required files for the given index exist.

        Args:
            idx (int): Index of the data item.

        Returns:
            bool: True if all required files exist, False otherwise.
        """
        try:
            meta = self.df.iloc[idx]
            product_id = meta.product_id
            grid_cell = meta.grid_cell
            row = grid_cell.split('_')[0]
            path = self.local_dir / Path(f"{row}/{grid_cell}/{product_id}")

            # Check TIFF bands
            for band in self.tif_bands:
                tif_path = path / f"{band}.tif"
                if not tif_path.exists():
                    print(f"Missing TIFF file: {tif_path}")
                    return False

            # Check PNG bands
            for band in self.png_bands:
                png_path = path / f"{band}.png"
                if not png_path.exists():
                    print(f"Missing PNG file: {png_path}")
                    return False

            return True
        except IndexError:
            print(f"Index {idx} is out of range.")
            return False
        except Exception as e:
            print(f"An error occurred while checking existence: {e}")
            return False

    def check_file_existence(self):
        """
        Check the existence of required files for all indices in the dataset.

        Returns:
            existing_indices (list): Indices where all required files exist.
            missing_indices (dict): Dictionary mapping indices to lists of missing files.
            df_existing (pd.DataFrame): Subset of the dataframe with existing files.
            df_missing (pd.DataFrame): Subset of the dataframe with missing files, including a column listing missing files.
        """
        existing_indices = []
        missing_indices = {}
        missing_files_list = []  # To store missing files per index

        for idx in tqdm(range(len(self)), desc="Checking file existence"):
            meta = self.df.iloc[idx]
            product_id = meta.product_id
            grid_cell = meta.grid_cell
            row = grid_cell.split('_')[0]
            path = self.local_dir / Path(f"{row}/{grid_cell}/{product_id}")

            missing_files = []

            # Check TIFF bands
            for band in self.tif_bands:
                tif_path = path / f"{band}.tif"
                if not tif_path.exists():
                    missing_files.append(str(tif_path))

            # Check PNG bands
            for band in self.png_bands:
                png_path = path / f"{band}.png"
                if not png_path.exists():
                    missing_files.append(str(png_path))

            if not missing_files:
                existing_indices.append(idx)
            else:
                missing_indices[idx] = missing_files
                missing_files_list.append((idx, missing_files))

        # Create subsets of the dataframe
        if existing_indices:
            df_existing = self.df.iloc[existing_indices].reset_index(drop=True)
        else:
            df_existing = pd.DataFrame(columns=self.df.columns)

        if missing_indices:
            # Create a list of indices with missing files
            missing_idxs = list(missing_indices.keys())
            df_missing = self.df.iloc[missing_idxs].copy()
            # Add a new column listing the missing files
            df_missing['missing_files'] = df_missing.index.map(missing_indices)
            df_missing = df_missing.reset_index(drop=True)
        else:
            df_missing = pd.DataFrame(columns=self.df.columns.tolist() + ['missing_files'])

        return existing_indices, missing_indices, df_existing, df_missing
