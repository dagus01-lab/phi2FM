#!/usr/bin/env python3
"""
Module: standard_methane_dataloader.py

Description
-----------
A *random‑access* PyTorch ``Dataset`` for methane patch data stored in one or
multiple Zarr stores. Because it implements ``__len__`` and ``__getitem__`` you
can:

* query ``len(dataset)`` for the number of available patches;
* index or slice it directly (``x, y = dataset[i]``);
* feed it into a conventional ``torch.utils.data.DataLoader`` to enable
  shuffling, weighted sampling, etc.

The code is stateless between processes so it works with
``num_workers > 0``.  Each worker lazily opens its own file handles the first
time it needs them, then keeps them around for the lifetime of the worker to
avoid the Zarr open‑file overhead.

Usage Example
-------------
>>> from standard_methane_dataloader import (to_channel_first,
...                                          MethaneZarrDataset,
...                                          load_dataset_from_dir)
>>> ds = load_dataset_from_dir("/Data/evgenios/methane/methane_patches",
...                            transform=to_channel_first)
>>> print(len(ds))                    # how many samples?
9216
>>> x, y = ds[42]                    # NumPy arrays (channels, H, W) and (H, W, 1)
>>> loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)

"""
from __future__ import annotations

import bisect
import glob
import logging
import os
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, DataLoader

__all__ = [
    "MethaneZarrDataset",
    "to_channel_first",
    "get_all_zarr_paths",
    "load_dataset_from_dir",
]

# -----------------------------------------------------------------------------
# Logging ---------------------------------------------------------------------
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Utility transforms ----------------------------------------------------------
# -----------------------------------------------------------------------------

def to_channel_first(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert *x* from (H, W, C) to (C, H, W).  ``y`` is returned unchanged.

    Raises
    ------
    ValueError
        If *x* does not have three dimensions.
    """
    if x.ndim != 3:
        raise ValueError(
            f"Expected x with 3 dimensions (H, W, C); got shape {x.shape} instead."
        )
    return np.transpose(x, (2, 0, 1)), y


# -----------------------------------------------------------------------------
# Helper for discovering Zarr stores -----------------------------------------
# -----------------------------------------------------------------------------

def get_all_zarr_paths(directory: str | Path) -> List[Path]:
    """Return a list of ``Path`` objects for every *.zarr* store in *directory*."""
    pattern = Path(directory).expanduser().joinpath("*.zarr")
    paths = [Path(p) for p in glob.glob(str(pattern))]
    logger.info("Found %d Zarr stores in '%s'.", len(paths), directory)
    return paths


# -----------------------------------------------------------------------------
# Core random‑access Dataset ---------------------------------------------------
# -----------------------------------------------------------------------------

class MethaneZarrDataset(Dataset):
    """Random‑access dataset across many Zarr stores.

    Parameters
    ----------
    zarr_paths : list[str | Path]
        One or more paths to *.zarr* directories containing the *data* and *mask*
        variables with shapes *(patch, H, W, channels)* and *(patch, H, W)*
        respectively.
    transform : callable, optional
        Function applied *after* reading a sample. Must take and return the pair
        *(x, y)*.

    Notes
    -----
    * Length is the **sum** of patch counts across all stores.
    * File handles are cached **per process** (one cache per DataLoader worker
      or the main process when ``num_workers=0``).
    """

    def __init__(self, zarr_paths: List[str | Path], transform: Callable | None = None):
        super().__init__()

        # Normalise to Path objects and sort for reproducibility
        self.zarr_paths: List[Path] = sorted(Path(p) for p in zarr_paths)
        if not self.zarr_paths:
            raise ValueError("No .zarr stores provided!")
        self.transform = transform

        # Build cumulative index mapping                          ──────────────
        self._lengths: List[int] = []      # patches per store
        self._starts: List[int] = [0]      # cumulative start indices
        total = 0
        for p in self.zarr_paths:
            try:
                ds = xr.open_zarr(p, consolidated=False)
                n = int(ds["data"].shape[0])
                ds.close()
            except Exception:
                logger.exception("Failed to inspect Zarr store '%s'.", p)
                raise
            self._lengths.append(n)
            total += n
            self._starts.append(total)  # after last index
        self._total = total
        logger.info(
            "Initialised MethaneZarrDataset with %d patches from %d stores.",
            self._total,
            len(self.zarr_paths),
        )

        # One‑process‑only cache for open datasets -----------------------------
        self._cache: dict[Path, xr.Dataset] = {}

    # ---------------------------------------------------------------------
    # PyTorch API ---------------------------------------------------------
    # ---------------------------------------------------------------------
    def __len__(self) -> int:  # type: ignore[override]
        return self._total

    def __getitem__(self, idx: int):  # type: ignore[override]
        if idx < 0:
            idx = self._total + idx  # support negative indexing
        if not 0 <= idx < self._total:
            raise IndexError(f"Index {idx} out of range for dataset of length {self._total}.")

        # Which store does *idx* belong to?
        store_pos = bisect.bisect_right(self._starts, idx) - 1  # -1 because starts includes 0 and end points
        local_idx = idx - self._starts[store_pos]
        store_path = self.zarr_paths[store_pos]

        ds = self._open_dataset(store_path)
        try:
            x = ds["data"][local_idx].values              # (H, W, C)
            y = ds["mask"][local_idx].values              # (H, W)
        except Exception:
            logger.exception(
                "Unable to read sample %d (local %d) from '%s'.", idx, local_idx, store_path
            )
            raise

        # Ensure mask has channel dimension
        if y.ndim == 2:
            y = np.expand_dims(y, axis=-1)                # (H, W, 1)

        if self.transform is not None:
            x, y = self.transform(x, y)

        x = np.array(x, copy=True)
        y = np.array(y, copy=True)
        return x, y

        # # Return *writable* copies so that downstream code isn't surprised
        # return np.copy(x), np.copy(y)

    # ---------------------------------------------------------------------
    # Private helpers -----------------------------------------------------
    # ---------------------------------------------------------------------
    def _open_dataset(self, path: Path) -> xr.Dataset:
        """Return a *cached* ``xarray.Dataset`` for *path* (lazy open)."""
        ds = self._cache.get(path)
        if ds is None:
            ds = xr.open_zarr(path, consolidated=False)
            self._cache[path] = ds
        return ds

    def __getstate__(self):
        """Exclude the open‑file cache when pickling (DataLoader workers)."""
        state = self.__dict__.copy()
        state["_cache"] = {}  # new process → new cache
        return state

    def __del__(self):
        for ds in self._cache.values():
            try:
                ds.close()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Convenience wrapper ---------------------------------------------------------
# -----------------------------------------------------------------------------

def load_dataset_from_dir(directory: str | Path, transform: Callable | None = to_channel_first) -> MethaneZarrDataset:
    """Return a :class:`MethaneZarrDataset` covering *all* stores in *directory*."""
    paths = get_all_zarr_paths(directory)
    return MethaneZarrDataset(paths, transform=transform)


# -----------------------------------------------------------------------------
# Demo ------------------------------------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Change this to your directory with *.zarr stores
    ZARR_DIR = "/Data/evgenios/methane/methane_patches"

    dataset = load_dataset_from_dir(ZARR_DIR)
    # logger.info("Dataset length: %d", len(dataset))

    # Show a couple of random indices
    import random

    # for i in random.sample(range(len(dataset)), 3):
    #     x, y = dataset[i]
    #     logger.info("Sample %d → x %s, y %s", i, x.shape, y.shape)

    # Wrap in a DataLoader ------------------------------------------------
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    # for batch_x, batch_y in loader:
    #     logger.info("Batch shapes: x %s, y %s", batch_x.shape, batch_y.shape)
    #     break
    import pdb; pdb.set_trace()