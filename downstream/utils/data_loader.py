import zarr
import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, List, Tuple, Dict, Union
import random
import torch.nn.functional as F
from copy import deepcopy
import os
from sklearn.cluster import KMeans


SEED=42
random.seed(SEED)
def callback_post_augmentation_val(x, y):
    return x, y

def callback_pre_augmentation_val(x, y):
    return x, y
    
def callback_post_augmentation_test(x, y):
    return x, y

def callback_pre_augmentation_test(x, y):
    return x, y
    
def callback_pre_augmentation_training(x, y):
    x = pad_bands(x)
    y = y.astype(np.float32, copy=False)
    return x, y
    
def callback_post_augmentation_training(x, y):
    x = beo.channel_last_to_first(x)
    if y.ndim > 2:
        y = beo.channel_last_to_first(y)
    x = minmax_normalize_image(x) #normalize_image_burned_area(x)
    return torch.from_numpy(x), torch.from_numpy(y)

def get_region_by_coords(coords, crs):
    pass

class AugmentationRotationXY:
    def __init__(self, p=0.5, inplace=False):
        self.p = p
        self.inplace = inplace

    def __call__(self, img, label):
        if random.random() > self.p:
            return img, label

        k = random.choice([1, 2, 3])  # 90, 180, 270 degrees

        img_rot = np.rot90(img, k=k, axes=(0, 1)) if img.ndim == 3 else np.rot90(img, k=k)
        label_rot = np.rot90(label, k=k, axes=(0, 1)) if label.ndim == 3 else label

        return img_rot.copy(), label_rot.copy()

class AugmentationMirrorXY:
    def __init__(self, p=0.5, inplace=False):
        self.p = p
        self.inplace = inplace

    def __call__(self, img, label):
        if random.random() > self.p:
            return img, label

        # Randomly decide to flip horizontally, vertically, or both
        flip_h = random.choice([True, False])
        flip_v = random.choice([True, False])

        if flip_h:
            img = np.flip(img, axis=1)  # width
            if label.ndim == 3:
                label = np.flip(label, axis=2)
        if flip_v:
            img = np.flip(img, axis=0)  # height
            if label.ndim == 3:
                label = np.flip(label, axis=1)

        return img.copy(), label.copy()
class AugmentationNoiseNormal:
    def __init__(self, p=0.5, std=0.05, inplace=False):
        self.p = p
        self.std = std
        self.inplace = inplace

    def __call__(self, img, label):
        if random.random() > self.p:
            return img, label

        noise = np.random.normal(loc=0.0, scale=self.std, size=img.shape)
        noisy_img = img + noise

        return noisy_img.astype(img.dtype), label


class PhiSatDataset(Dataset):
    def __init__(
        self,
        zarr_path: str,
        dataset_set: str,
        callback_pre_augmentation: Optional[Callable] = None,
        callback_post_augmentation: Optional[Callable] = None,
        augmentations: Optional[List[Callable]] = None,
        task_filter: Optional[str] = None,
        metadata_keys: Optional[List[str]] = None,
        verbose: bool = False,
        crop_images: bool = False,
        num_classes: int = 4,
        n_shot: int = 0,
        patch_size: Optional[Tuple[int, int]] = None, 
        weights_dir: str = None,
        n_regions: int = 6, 
        n_shot_strategy: str = 'stratified'
    ):
        # Open Zarr and set basic attributes
        self.root = zarr.open(zarr_path, mode='r')
        self.dataset_set = dataset_set.lower()
        self.task_filter = task_filter.lower() if task_filter else None
        self.metadata_keys = metadata_keys or []
        self.verbose = verbose
        self.crop_images = crop_images
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.callback_pre_augmentation = callback_pre_augmentation
        self.callback_post_augmentation = callback_post_augmentation
        self.augmentations = augmentations
        self.n_shot = n_shot
        self.weights_dir = weights_dir
        self.n_regions = n_regions
        self.n_shot_strategy = n_shot_strategy

        # Access the dataset group
        if self.dataset_set not in self.root:
            raise ValueError(f"Dataset set '{dataset_set}' not found in Zarr store")
        self.dataset_group = self.root[self.dataset_set]
        self.sample_ids = sorted(self.dataset_group.keys())

        # Filter by task if necessary
        if self.task_filter:
            self.sample_ids = [
                sid for sid in self.sample_ids
                if self.dataset_group[sid].attrs.get('task', '') == self.task_filter
            ]
            if not self.sample_ids:
                raise ValueError("No matching samples after task filtering.")

        # Build patches list
        self.patches = self._generate_patches(self.sample_ids)

        # Compute class and pos weights at init
        self.class_weights, self.pos_weights = self._load_or_compute_weights()

        # If n_shot requested, subsample patches stratified
        if self.n_shot > 0:
            self._cluster_and_nshot(n_regions=self.n_regions, strategy=self.n_shot_strategy, seed=SEED) #self.get_n_shots(strategy='stratified', seed=SEED)

    def _generate_patches(self, sample_ids):
        """
        Generates patch coordinates from the given sample IDs.
        If patching is enabled (via `patch_size`), returns a list of tuples (sample_id, y, x)
        corresponding to top-left coordinates of patches.
        If not, returns the original list of sample IDs.
        
        Args:
            sample_ids (List[str]): List of sample identifiers.
    
        Returns:
            List[Union[str, Tuple[str, int, int]]]: List of patches or sample IDs.
        """
        if self.patch_size:
            patches = []
            ph, pw = self.patch_size
            for sid in sample_ids:
                img = self.dataset_group[sid]['img']
                h, w = img.shape[1:] if img.ndim == 3 else img.shape[2:]
                for y in range(0, h, ph):
                    for x in range(0, w, pw):
                        if y + ph <= h and x + pw <= w:
                            patches.append((sid, y, x))
            return patches
        else:
            return list(sample_ids)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx: int) -> Dict:
        sid, y, x = self._unpack_patch(self.patches[idx])
        sample_group = self.dataset_group[sid]
        img = sample_group['img'][:]
        label = self._load_label_array(sample_group['label'])

        img, label = self._preprocess(img, label, y, x)
        sample = {'img': img, 'label': label, 'task': sample_group.attrs.get('task', ''), 'sample_id': sid}
        for key in self.metadata_keys:
            if key in sample_group.attrs:
                sample[key] = sample_group.attrs[key]
        if self.patch_size:
            sample['patch_coord'] = (y, x)
        return sample

    def _unpack_patch(self, patch):
        if isinstance(patch, tuple):
            return patch  # (sid, y, x)
        else:
            return patch, 0, 0

    def _load_label_array(self, ds) -> np.ndarray:
        # Handle scalar and array labels
        if ds.shape == ():
            return np.array(ds[()])
        return ds[:]


    def _cluster_and_nshot(self, n_shots: int = 0, n_regions: int = 5, strategy: str = 'stratified', seed: int = 0):
        """
        Clusters samples by geolocation, applies n-shot sampling within each region,
        and saves or loads the resulting sample IDs and clustering info for reproducibility.
        """
        if not self.weights_dir:
            raise ValueError("weights_dir must be set to enable reproducibility")
        os.makedirs(self.weights_dir, exist_ok=True)

        # file names inside weights_dir
        sample_file  = os.path.join(
            self.weights_dir, f"{self.weights_dir}_{n_shots}_samples.json"
        )
        cluster_file = os.path.join(
            self.weights_dir, f"{self.weights_dir}_clusters.json"
        )
        
        sid_clusters = {i: [] for i in range(n_regions)}

        # 1) if we already have the samples file, just load & return
        if os.path.exists(sample_file):
            with open(sample_file, 'r') as f:
                self.sample_ids = json.load(f)
            self.patches = self._generate_patches(self.sample_ids)
            if self.verbose:
                print(f"[INFO] Loaded {len(self.sample_ids)} from {sample_file}")
            return
        
        loaded = False
        empty_regions = False
        if os.path.exists(cluster_file):
            try:
                with open(cluster_file) as f:
                    info = json.load(f)
                if info.get("n_regions", None) == 0:
                    if self.verbose:
                        print(f"[INFO] Cluster file indicates zero regions; skipping clustering.")
                    # fallback: just do a single global n-shot
                    loaded = True
                    empty_regions = True
                    if self.verbose:
                        print(f"[INFO] No cluster found at {cluster_file}")

                else:    
                    for sid, meta in info['samples'].items():
                        cid = int(meta['cluster'])
                        if sid in sid_to_coord:
                            sid_clusters[cid].append(sid)
                    loaded = True
                    if self.verbose:
                        print(f"[INFO] Loaded clusters from {cluster_file}")
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Invalid cluster file, recomputing: {e}")

        # 2) gather geolocations
        coords, sid_to_coord = [], {}
        #in case the data of the regions was loaded and is empty, avoid repeating the calculation needlessly
        if loaded and not empty_regions:
            for sid in self.sample_ids:
                ll = self.dataset_group[sid].attrs.get('geolocation', {}).get('LL', None)
                # skip if missing or contains NaN
                if ll is None or any(np.isnan(ll)):
                    if self.verbose:
                        print(f"[WARN] skipping sample {sid} — invalid LL: {ll}")
                    continue
                coords.append(ll)
                sid_to_coord[sid] = ll
        if len(coords) == 0:
            if self.verbose:
                print("[WARN] No valid geolocations found; using get_n_shots on whole dataset")
            self.get_n_shots(strategy=strategy, n=n_shots, seed=seed)
            final_ids = self.patches
        else: 
            if len(coords) < n_regions:
                print(f"{len(coords)} geolocated samples < {n_regions} regions")
                #raise ValueError(f"{len(coords)} geolocated samples < {n_regions} regions")

            # 3) load or compute clusters

            if not loaded:
                coords_np = np.array(coords)
                km = KMeans(n_clusters=n_regions, random_state=seed).fit(coords_np)
                for sid, coord in sid_to_coord.items():
                    cid = int(km.predict([coord])[0])
                    sid_clusters[cid].append(sid)

                serial = {
                    'n_regions': n_regions,
                    'cluster_centers': km.cluster_centers_.tolist(),
                    'samples': {
                        sid: {
                            'cluster': int(km.predict([coord])[0]),
                            'coord': [float(c) for c in coord]
                        } for sid, coord in sid_to_coord.items()
                    }
                }
                with open(cluster_file, 'w') as f:
                    json.dump(serial, f, indent=2)
                if self.verbose:
                    print(f"[INFO] Saved clusters to {cluster_file}")

            # 4) for each region, draw exactly n_shots
            final_ids = []
            # compute region sizes
            region_sizes = {i: len(sids) for i, sids in sid_clusters.items()}
            total = sum(region_sizes.values())
            if total == 0:
                # no clusters → fallback
                return self.get_n_shots(strategy=strategy, n=n_shots, seed=seed)

            # compute how many shots per region
            shots_per_region = {}
            cum = 0
            for i, size in region_sizes.items():
                if i < len(region_sizes) - 1:
                    k = int(round(n_shots * size / total))
                    shots_per_region[i] = k
                    cum += k
                else:
                    # last region gets remainder to ensure sum == n_shots
                    shots_per_region[i] = n_shots - cum

            # now sample per region
            for region_idx, region_ids in sid_clusters.items():
                k = shots_per_region.get(region_idx, 0)
                if k <= 0 or len(region_ids) == 0:
                    continue
                # make a temp view of just this region
                tmp = deepcopy(self)
                tmp.sample_ids = region_ids
                tmp.patches    = tmp._generate_patches(region_ids)
                tmp.get_n_shots(strategy=strategy, n=k, seed=seed)
                # collect the selected patches
                final_ids.extend(tmp.patches)

        # 5) save the final list
        self.sample_ids = final_ids
        self.patches   = self._generate_patches(final_ids)
        with open(sample_file, 'w') as f:
            json.dump(self.sample_ids, f, indent=2)
        if self.verbose:
            print(f"[INFO] Saved {len(self.sample_ids)} samples to {sample_file}")

    def _preprocess(self, img: np.ndarray, label: np.ndarray, y: int, x: int):
        # Transpose channels if needed
        img = self._transpose_if_needed(img)
        label = self._transpose_if_needed(label)

        # Squeeze single-channel masks to one-hot
        if label.ndim == 3 and label.shape[-1] == 1:
            lab = torch.from_numpy(label.squeeze(-1)).long()
            label = F.one_hot(lab, num_classes=self.num_classes).numpy()

        # Crop patch
        if self.patch_size:
            ph, pw = self.patch_size
            img = img[y:y+ph, x:x+pw, ...]
            if label.ndim==3:
                label = label[y:y+ph, x:x+pw, ...]
            elif label.ndim==2:
                label = label[y:y+ph, x:x+pw]

        # Optional image cropping
        if self.crop_images:
            img = img[:64, :64, ...]
            if label.ndim==3:
                label = label[:64, :64, ...]

        # Augmentations
        if self.callback_pre_augmentation:
            img, label = self.callback_pre_augmentation(img, label)
        if self.augmentations:
            for aug in self.augmentations:
                img, label = aug(img, label)
        if self.callback_post_augmentation:
            img, label = self.callback_post_augmentation(img, label)

        return img, label

    def _load_or_compute_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads class_weights and pos_weights from disk if available,
        otherwise computes and saves them.
        """
        if self.weights_dir:
            os.makedirs(self.weights_dir, exist_ok=True)
            cw_path = os.path.join(self.weights_dir, f"{self.dataset_set}_class_weights.npy")
            pw_path = os.path.join(self.weights_dir, f"{self.dataset_set}_pos_weights.npy")
            if os.path.exists(cw_path) and os.path.exists(pw_path):
                cw = torch.from_numpy(np.load(cw_path))
                pw = torch.from_numpy(np.load(pw_path))
                return cw, pw
                # Compute fresh
        cw, pw = self.compute_class_and_pos_weights()
        if self.weights_dir:
            np.save(cw_path, cw.numpy())
            np.save(pw_path, pw.numpy())
        return cw, pw
                
    def _transpose_if_needed(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3:
            c0, c1, c2 = arr.shape
            if c2 > c0:
                return arr.transpose(1, 2, 0)
            if c1 < c2:
                return arr.transpose(0, 2, 1)
        return arr

    def _infer_class_from_label(self, lbl: Union[np.ndarray, int, float]) -> Optional[int]:
        """
        Infers the dominant class from a given label array.
    
        The method supports different label formats:
        - Scalar label: returns the integer value directly.
        - One-hot vector: returns the index of the maximum value.
        - 2D mask: returns the most frequent class in the mask.
        - 3D one-hot mask: infers class per-pixel and returns the most frequent one.
    
        Args:
            lbl (Union[np.ndarray, int, float]): Label in various possible formats.
    
        Returns:
            Optional[int]: The inferred class index or None if inference fails.
        """
        # 1) Scalar or 0‑dim array
        if np.isscalar(lbl) or (isinstance(lbl, np.ndarray) and lbl.ndim == 0):
            return int(lbl)

        # 2) If it has a singleton channel (e.g. H×W×1 or 1×H×W), squeeze it out
        if isinstance(lbl, np.ndarray) and lbl.ndim == 3 and (lbl.shape[0] == 1 or lbl.shape[-1] == 1):
            lbl = np.squeeze(lbl, axis=0 if lbl.shape[0] == 1 else -1)

        # 3) One‑hot (C×H×W)
        if isinstance(lbl, np.ndarray) and lbl.ndim == 3 and lbl.shape[0] == self.num_classes:
            mask = np.argmax(lbl, axis=0)
            vals, counts = np.unique(mask, return_counts=True)
            return int(vals[np.argmax(counts)])

        # 4) One‑hot (H×W×C)
        if isinstance(lbl, np.ndarray) and lbl.ndim == 3 and lbl.shape[-1] == self.num_classes:
            mask = np.argmax(lbl, axis=2)
            vals, counts = np.unique(mask, return_counts=True)
            return int(vals[np.argmax(counts)])

        # 5) Plain 2D mask
        if isinstance(lbl, np.ndarray) and lbl.ndim == 2:
            vals, counts = np.unique(lbl, return_counts=True)
            return int(vals[np.argmax(counts)])

        # Otherwise we don’t know how to interpret it
        return None
        
    def compute_class_and_pos_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes class weights and positive weights for handling class imbalance.
    
        - Class weights are inverse-frequency based, for use in cross-entropy loss.
        - Positive weights are used for balancing binary loss per class.
    
        The method counts sample-level and pixel-level occurrences across the dataset.
    
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: class_weights and pos_weights tensors.
        """
        sample_pos = np.zeros(self.num_classes, int)
        pixel_counts = np.zeros(self.num_classes, int)
        total_pixels = 0

        for sid in self.sample_ids:
            ds = self.dataset_group[sid]['label']
            lbl = self._load_label_array(ds)
            cls = self._infer_class_from_label(lbl)
            # skip invalid class indices
            if cls is None or cls < 0 or cls >= self.num_classes:
                continue
            sample_pos[cls] += 1
            # pixel-wise
            if lbl.ndim >= 2:
                flat = lbl.ravel()
                total_pixels += flat.size
                for c in range(self.num_classes):
                    pixel_counts[c] += int((flat == c).sum())

        # sample-based weights
        freq = sample_pos.astype(float)
        freq[freq == 0] = 1.0
        class_weights = torch.from_numpy((freq.sum() / (freq * self.num_classes)).astype(np.float32))
        neg = sample_pos.sum() - sample_pos.astype(float)
        pos = sample_pos.astype(float)
        pos[pos == 0] = 1.0
        pos_weights = torch.from_numpy((neg / pos).astype(np.float32))
        return class_weights, pos_weights

    def split_by_percentages(self, split: List[float]) -> List['PhiSatDataset']:
        """
        Splits the dataset into subsets according to specified percentages, 
        while maintaining class distribution as best as possible.
    
        Args:
            split (List[float]): A list of floats that sum to 1.0, indicating split ratios.
    
        Returns:
            List[PhiSatDataset]: A list of new PhiSatDataset instances representing each subset.
        """
        if not np.isclose(sum(split), 1.0):
            raise ValueError("Split percentages must sum to 1.0")

        class_to_ids: Dict[int, List] = {c: [] for c in range(self.num_classes)}
        for sid in self.sample_ids:
            lbl = self._load_label_array(self.dataset_group[sid]['label'])
            cls = self._infer_class_from_label(lbl)
            if cls is not None:
                class_to_ids[cls].append(sid)

        subsets: List[List[str]] = [[] for _ in split]
        for cls, ids in class_to_ids.items():
            rng = random.Random(0)
            rng.shuffle(ids)
            n = len(ids)
            sizes = [int(r * n) for r in split]
            sizes[-1] = n - sum(sizes[:-1])
            start = 0
            for i, size in enumerate(sizes):
                subsets[i].extend(ids[start:start+size])
                start += size

        result = []
        for ids in subsets:
            ds = deepcopy(self)
            ds.sample_ids = ids
            ds.patches = ds._generate_patches(ids)
            result.append(ds)
        return result

    def get_n_shots(self, strategy: str = 'stratified', n: int= None, seed: Optional[int] = None) -> None:
        """
        Selects a fixed number of patches (n-shot) using a given strategy.
    
        Currently supports 'stratified' sampling to ensure class balance.
        Updates `self.patches` in-place with the selected subset.
    
        This method supports both classification and segmentation tasks. It uses 
        `_infer_class_from_label()` to identify the dominant class from labels that 
        can be:
            - scalar (for classification),
            - 2D segmentation masks     
        For segmentation tasks, the most frequent class in the patch is used for sampling decisions.
    
        Args:
            strategy (str): Sampling strategy ('stratified' supported).
            n (int, optional): Number of samples to select. Overrides self.n_shot if given.
            seed (Optional[int]): Random seed for reproducibility.
        """
        if not n is None: 
            self.n_shot = n
        
        if self.n_shot <= 0:
            return
        if len(self.patches) < self.n_shot:
            self.n_shot = len(self.patches)
            print("Not enough patches for n_shot sampling")
        
        rng = random.Random(seed)
        class_to_patches: Dict[int, List] = {c: [] for c in range(self.num_classes)}
        for patch in self.patches:
            sid, y, x = self._unpack_patch(patch)
            lbl = self._load_label_array(self.dataset_group[sid]['label'])
            if self.patch_size:
                ph, pw = self.patch_size
                lbl = lbl[y:y+ph, x:x+pw, ...] if lbl.ndim >= 2 else lbl
            cls = self._infer_class_from_label(lbl)
            if cls is not None:
                class_to_patches[cls].append(patch)

        total = sum(len(v) for v in class_to_patches.values())
        selected: List[Union[str, Tuple[str,int,int]]] = []
        for cls, plist in class_to_patches.items():
            k = int(round((len(plist)/total) * self.n_shot))
            k = min(k, len(plist))
            selected.extend(rng.sample(plist, k))
        if len(selected) < self.n_shot:
            leftovers = [p for p in self.patches if p not in selected]
            selected.extend(rng.sample(leftovers, self.n_shot - len(selected)))
        self.patches = selected[:self.n_shot]
        
def collate_fn(batch: List[Dict]) -> (torch.Tensor, torch.Tensor):
    """
    Simplified collate that stacks 'img' and 'label' from each sample in the batch.
    Returns:
        images: Tensor of shape (B, C, H, W) or similar, obtained by stacking sample['img']
        labels: Tensor of shape (B, ...) by stacking sample['label']
    """
    # Collect all images and labels
    imgs = []
    lbls = []
    for sample in batch:
        img = sample['img']
        lbl = sample['label']

        # Convert NumPy→Tensor if needed
        if isinstance(img, torch.Tensor):
            img_t = img
        else:
            img_t = torch.from_numpy(img)

        if isinstance(lbl, torch.Tensor):
            lbl_t = lbl
        else:
            lbl_t = torch.from_numpy(lbl)

        imgs.append(img_t)
        lbls.append(lbl_t)

    # Stack them along dimension 0
    images_batch = torch.stack(imgs, dim=0)
    labels_batch = torch.stack(lbls, dim=0)

    return images_batch, labels_batch

def adv_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function to handle different task types and metadata.
    
    Args:
        batch: List of sample dictionaries from PhiSatDataset
        
    Returns:
        Dictionary with batched tensors and metadata
    """
    # Group samples by task to handle different label shapes
    task_groups = {}
    for sample in batch:
        task = sample['task']
        if task not in task_groups:
            task_groups[task] = []
        task_groups[task].append(sample)
    
    result = {}
    
    # Process each task group separately
    for task, samples in task_groups.items():
        # Get all keys from the first sample
        keys = samples[0].keys()
        
        for key in keys:
            # Skip task and sample_id for batching
            if key in ['task', 'sample_id']:
                continue
                
            # Handle tensors
            if isinstance(samples[0][key], torch.Tensor):
                # Stack tensors with same shapes
                try:
                    result[f"{task}_{key}"] = torch.stack([s[key] for s in samples])
                except RuntimeError:
                    # If tensors have different shapes, return as list
                    result[f"{task}_{key}"] = [s[key] for s in samples]
            else:
                # For non-tensor data, collect as list
                result[f"{task}_{key}"] = [s[key] for s in samples]
        
        # Store sample IDs
        result[f"{task}_sample_ids"] = [s['sample_id'] for s in samples]
        
    # Store task information
    result['tasks'] = list(task_groups.keys())
    result['task_counts'] = {task: len(samples) for task, samples in task_groups.items()}
    
    return result


def get_zarr_dataloader(
    zarr_path: str,
    dataset_set: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    task_filter: Optional[str] = None,
    metadata_keys: Optional[List[str]] = None,
    verbose: bool = False,
    num_classes: int = 2,
    split: list = None, 
    callback_pre_augmentation: list = None,
    callback_post_augmentation: list = None,
    augmentations: list = None, 
    crop_images: bool= False, 
    generator: torch.Generator = None, 
    pin_memory: bool = True, 
    drop_last: bool = False, 
    n_shot: Union[int, List[int]]= 0, 
    weights_dir: str = None, 
    n_regions: int = 6
    
) -> DataLoader:
    """
    Create a DataLoader for Zarr dataset.
    
    Args:
        zarr_path: Path to Zarr dataset
        dataset_set: One of {"trainval", "test"}
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle samples
        num_workers: Number of data loading workers
        task_filter: Filter to specific task (classification, segmentation, etc.)
        metadata_keys: Keys of metadata to include
        verbose: Whether to print debug information
        split: list containing the percentage of values from the original dataset to form other collections
        callback_pre_augmentation: list containing, for every subset, the callbacks performed before augmentation
        callback_post_augmentation: list containing, for every subset, the callbacks performed after augmentation
        crop_images: boolean flag, indicating whether each sample will be cropped
        augmentations: list containing, for every subset, the augmentations performed on the data
        
    Returns:
        PyTorch DataLoader for the dataset
    """
    if isinstance(split, list) and isinstance(callback_pre_augmentation, list) and isinstance(callback_post_augmentation, list) and isinstance(augmentations, list):
        assert len(split) == len(callback_pre_augmentation) == len(callback_post_augmentation) == len(augmentations) == len(n_shot), \
            "Mismatch in lengths of split subsets and callbacks"
    batch_size = 16
    dataset = PhiSatDataset(
        zarr_path=zarr_path,
        dataset_set=dataset_set,
        #transform=transform,
        #target_transform=target_transform,
        task_filter=task_filter,
        metadata_keys=metadata_keys,
        verbose=verbose, 
        crop_images=crop_images, 
        num_classes=num_classes,
        #patch_size=(256,256),
        weights_dir=weights_dir, 
        n_regions=n_regions, 
        n_shot_strategy='stratified'
    )
    img, label = dataset[0]['img'], dataset[0]['label']
    print(f"Dataset {dataset_set} shapes: img={img.shape}, label={label.shape}")
    weights, pos_weights = dataset.class_weights, dataset.pos_weights
    print(f"weights: {weights}, pos_weights:{pos_weights}")
    if not split is None:
        sub_datasets = dataset.split_by_percentages(split=split)
        dataloaders = []
        for idx, sub_dataset in enumerate(sub_datasets):
            if n_shot[idx] != 0:
                sub_dataset._cluster_and_nshot(n_shots=n_shot[idx], n_regions=sub_dataset.n_regions, strategy=sub_dataset.n_shot_strategy, seed=SEED)
                #sub_dataset.get_n_shots(strategy='stratified', n=n_shot[idx], seed=SEED)
            sub_dataset.callback_pre_augmentation=callback_pre_augmentation[idx]
            sub_dataset.callback_post_augmentation=callback_post_augmentation[idx]
            sub_dataset.augmentations=augmentations[idx]
            dataloaders.append(
                DataLoader(
                    sub_dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    collate_fn=collate_fn,
                    pin_memory=True,
                    generator=generator
                )
            )
        return weights, pos_weights, *dataloaders
    else:
        dataset.callback_pre_augmentation=callback_pre_augmentation
        dataset.callback_post_augmentation=callback_post_augmentation
        dataset.augmentations=augmentations
        return  weights, pos_weights, DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
                generator=generator, 
                pin_memory=pin_memory, 
                drop_last=drop_last
            )
        


if __name__ == "__main__":
    zarr_path = "burned_area_dataset.zarr"
    
    dataset_set = "trainval"
    
    # Create DataLoader
    weight, pos_weight, dl_train, dl_val = get_zarr_dataloader(
        zarr_path=dataset_path,                     
        dataset_set="trainval",                 
        batch_size=16,                           
        shuffle=True,                            
        num_workers=4,                           
        metadata_keys=["sensor", "timestamp", "geolocation", "crs"],   # Include auxiliary metadata fields
        verbose = False,
        split = [.8, .2], 
        callback_pre_augmentation = [callback_pre_augmentation_training, callback_pre_augmentation_val],
        callback_post_augmentation = [callback_post_augmentation_training, callback_post_augmentation_val],
        augmentations = [augmentations_training, augmentations_val], 
        crop_images= crop_images, 
        generator= torch.Generator(device), 
        pin_memory=True, 
        drop_last=False, 
        n_shot=[50, 0],
        num_classes=num_classes
    )

    weight, pos_weight, dl_test = get_zarr_dataloader(
        zarr_path=dataset_path,                     
        dataset_set="test",                 
        batch_size=16,                           
        shuffle=True,                            
        num_workers=4,                           
        metadata_keys=["sensor", "timestamp", "geolocation", "crs"],  
        verbose = False,
        split = None, 
        callback_pre_augmentation = callback_pre_augmentation_test,
        callback_post_augmentation = callback_post_augmentation_test,
        augmentations = augmentations_test,
        crop_images= crop_images, 
        generator= torch.Generator(device), 
        pin_memory=True, 
        drop_last=False, 
        n_shot=50,
        num_classes=num_classes
    ) 
