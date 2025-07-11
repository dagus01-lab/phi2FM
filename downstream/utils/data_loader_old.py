import zarr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union, Callable, Tuple
import random
import torch.nn.functional as F
from copy import deepcopy

random.seed(42)

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
        patch_size: Optional[Tuple[int, int]] = None  # ← NEW ARG
    ):
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

        if self.dataset_set not in self.root:
            raise ValueError(f"Dataset set '{dataset_set}' not found in Zarr store")
        self.dataset_group = self.root[self.dataset_set]
        self.sample_ids = sorted(self.dataset_group.keys())

        if self.task_filter:
            self.sample_ids = [
                sid for sid in self.sample_ids 
                if self.dataset_group[sid].attrs.get('task', '') == self.task_filter
            ]
            if len(self.sample_ids) == 0:
                raise ValueError("No matching samples after task filtering.")

        if self.patch_size:
            self.patches = []  # list of tuples: (sample_id, y, x)
            for sid in self.sample_ids:
                sample_group = self.dataset_group[sid]
                img = sample_group['img']
                h, w = img.shape[1:] if len(img.shape) == 3 else img.shape[2:]
                patch_h, patch_w = self.patch_size
                for y in range(0, h, patch_h):
                    for x in range(0, w, patch_w):
                        if y + patch_h <= h and x + patch_w <= w:
                            self.patches.append((sid, y, x))
        else:
            self.patches = self.sample_ids  # fallback to normal mode
        self.get_n_shots('stratified')

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx: int) -> Dict:
        if self.patch_size:
            sample_id, y, x = self.patches[idx]
        else:
            sample_id = self.patches[idx]
            y, x = 0, 0  # Full image

        sample_group = self.dataset_group[sample_id]
        img = sample_group['img'][:]
        label_array = sample_group['label']
        label = label_array[()] if label_array.shape == () else label_array[:]

        # Transpose logic
        if img.ndim == 3:
            if img.shape[2] > img.shape[0]:
                img = img.transpose(1, 2, 0)
            elif img.shape[1] < img.shape[2]:
                img = img.transpose(0, 2, 1)
        if label.ndim == 3:
            if label.shape[2] > label.shape[0]:
                label = label.transpose(1, 2, 0)
            elif label.shape[1] < label.shape[2]:
                label = label.transpose(0, 2, 1)

        if label.ndim == 3 and label.shape[2] == 1:
            label = torch.from_numpy(label.squeeze(2)).long()
            label = F.one_hot(label, num_classes=self.num_classes).numpy()

        if self.patch_size:
            ph, pw = self.patch_size
            img = img[y:y+ph, x:x+pw, ...]
            label = label[y:y+ph, x:x+pw, ...]

        if self.crop_images:
            img = img[:64, :64, :] if img.ndim == 3 else img[:, :64, :64, :]
            label = label[:64, :64, :] if label.ndim == 3 else label[:, :64, :64, :]

        if self.callback_pre_augmentation:
            img, label = self.callback_pre_augmentation(img, label)

        if self.augmentations:
            for aug in self.augmentations:
                img, label = aug(img, label)

        if self.callback_post_augmentation:
            img, label = self.callback_post_augmentation(img, label)

        task = sample_group.attrs.get('task', '')
        sample = {
            'img': img,
            'label': label,
            'task': task,
            'sample_id': sample_id
        }

        for key in self.metadata_keys:
            if key in sample_group.attrs:
                sample[key] = sample_group.attrs[key]

        if self.patch_size:
            sample['patch_coord'] = (y, x)

        return sample

    def _infer_class_from_label(self, lbl: Union[np.ndarray, int, float]) -> Optional[int]:
        """
        Infers the dominant class from:
          - scalar label
          - one‑hot vector
          - 2D mask
          - 3D mask (C × H × W or H × W × C)
        Returns None if unable to infer.
        """
        # scalar
        if np.isscalar(lbl):
            return int(lbl)

        # zero‑dim array
        if isinstance(lbl, np.ndarray) and lbl.ndim == 0:
            return int(lbl.item())

        # one‑hot vector
        if isinstance(lbl, np.ndarray) and lbl.ndim == 1 and lbl.shape[0] == self.num_classes:
            return int(np.argmax(lbl))

        # 2D mask H×W
        if isinstance(lbl, np.ndarray) and lbl.ndim == 2:
            vals, counts = np.unique(lbl, return_counts=True)
            return int(vals[np.argmax(counts)])

        # 3D mask C×H×W or H×W×C
        if isinstance(lbl, np.ndarray) and lbl.ndim == 3:
            # channel first?
            if lbl.shape[0] == self.num_classes:
                mask = np.argmax(lbl, axis=0)
            # channel last?
            elif lbl.shape[2] == self.num_classes:
                mask = np.argmax(lbl, axis=2)
            else:
                return None
            vals, counts = np.unique(mask, return_counts=True)
            return int(vals[np.argmax(counts)])

        return None

    def get_n_shots(self,
                    strategy: str = 'random',
                    seed: Optional[int] = None
                   ) :
        """
        Subsample self.patches down to self.n_shot items.
        Supports:
          - 'random': uniform random over all patches
          - 'stratified': preserves class distribution (dominant class per patch)
        Works for both classification and segmentation masks.
        """
        if self.n_shot <= 0:
            raise ValueError("n_shot must be > 0")
        if len(self.patches) < self.n_shot:
            raise ValueError(f"n_shot={self.n_shot} > available patches={len(self.patches)}")

        rng = random.Random(seed)
        selected: List[Union[str, Tuple[str,int,int]]] = []

        if strategy == 'random':
            selected = rng.sample(self.patches, self.n_shot)

        elif strategy == 'stratified':
            # build class → list of patches
            class_to_patches: Dict[int, List] = {c: [] for c in range(self.num_classes)}
            for patch in self.patches:
                sid = patch if isinstance(patch, str) else patch[0]
                # load only the LABEL patch
                lbl_ds = self.dataset_group[sid]['label']
                full_lbl = lbl_ds[()] if lbl_ds.shape == () else lbl_ds[:]
                # if patching, crop the label
                if self.patch_size and not isinstance(patch, str):
                    y, x = patch[1], patch[2]
                    ph, pw = self.patch_size
                    full_lbl = full_lbl[y:y+ph, x:x+pw, ...] if full_lbl.ndim>=2 else full_lbl
                cls = self._infer_class_from_label(full_lbl)
                if cls is not None:
                    class_to_patches.setdefault(cls, []).append(patch)

            total = sum(len(v) for v in class_to_patches.values())
            if total < self.n_shot:
                raise ValueError("Not enough labeled patches for stratified sampling")

            # compute how many per class
            for cls, plist in class_to_patches.items():
                proportion = len(plist) / total
                k = int(round(proportion * self.n_shot))
                k = min(k, len(plist))
                selected.extend(rng.sample(plist, k))

            # pad or truncate to exactly n_shot
            if len(selected) < self.n_shot:
                remaining = [p for p in self.patches if p not in selected]
                selected.extend(rng.sample(remaining, self.n_shot - len(selected)))
            selected = selected[:self.n_shot]

        else:
            raise NotImplementedError(f"Unknown strategy '{strategy}'")

        # return new dataset instance
        self.patches = selected
            
    def split_by_percentages(self, split: List[float], num_classes: int = None) -> List['PhiSatDataset']:
        if not np.isclose(sum(split), 1.0):
            raise ValueError("Split percentages must sum to 1.0")
    
        class_to_ids: Dict[int, List[Union[str, Tuple[str, int, int]]]] = {}
        for sid in self.sample_ids:
            base_id = sid if isinstance(sid, str) else sid[0]
            grp = self.dataset_group[base_id]
            arr = grp['label']
            lbl = arr[()] if hasattr(arr, 'shape') and len(arr.shape) == 0 else arr[:]
    
            cls = -1
            if np.isscalar(lbl) or (isinstance(lbl, np.ndarray) and lbl.ndim == 0):
                cls = int(lbl)
            elif isinstance(lbl, np.ndarray) and lbl.ndim == 1 and num_classes and lbl.shape[0] == num_classes:
                one_hot = (lbl > 0)
                cls = int(np.argmax(one_hot)) if one_hot.sum() >= 1 else -1
            elif isinstance(lbl, np.ndarray) and lbl.ndim in [2, 3] and num_classes:
                if lbl.ndim == 2:
                    mask = lbl
                else:
                    if lbl.shape[0] == num_classes:
                        mask = np.argmax(lbl, axis=0)
                    elif lbl.shape[2] == num_classes:
                        mask = np.argmax(lbl, axis=2)
                    else:
                        mask = None
                if mask is not None:
                    vals, counts = np.unique(mask, return_counts=True)
                    cls = int(vals[np.argmax(counts)])
            class_to_ids.setdefault(cls, []).append(sid)
    
        if set(class_to_ids.keys()) == {-1}:
            ids = self.sample_ids.copy()
            random.shuffle(ids)
            subsets = []
            total = len(ids)
            sizes = [int(r * total) for r in split]
            sizes[-1] = total - sum(sizes[:-1])
            start = 0
            for size in sizes:
                subset = deepcopy(self)
                subset.sample_ids = ids[start:start+size]
                subsets.append(subset)
                start += size
            return subsets
    
        per_split_ids: Dict[int, List[Union[str, Tuple[str, int, int]]]] = {i: [] for i in range(len(split))}
        for cls, ids in class_to_ids.items():
            random.shuffle(ids)
            n = len(ids)
            sizes = [int(r * n) for r in split]
            sizes[-1] = n - sum(sizes[:-1])
            start = 0
            for i, size in enumerate(sizes):
                per_split_ids[i].extend(ids[start:start+size])
                start += size
    
        subsets = []
        for i in range(len(split)):
            subset = deepcopy(self)
            subset.sample_ids = per_split_ids[i]
            subsets.append(subset)
        return subsets

        
    def compute_class_and_pos_weights(self, num_classes: int):
        pixel_counts = np.zeros((num_classes,), dtype=np.int64)
        total_pixels = 0
        sample_pos = np.zeros((num_classes,), dtype=np.int64)
        sample_total = 0
    
        for sid in self.sample_ids:
            base_id = sid if isinstance(sid, str) else sid[0]
            grp = self.dataset_group[base_id]
            try:
                arr = grp['label']
                lbl = arr[()] if hasattr(arr, 'shape') and len(arr.shape) == 0 else arr[:]
    
                if np.isscalar(lbl) or (isinstance(lbl, np.ndarray) and lbl.ndim == 0):
                    cls = int(lbl)
                    if not (0 <= cls < num_classes):
                        raise ValueError(f"Label {cls} out of range for sample {sid}")
                    sample_pos[cls] += 1
                    sample_total += 1
                    pixel_counts[cls] += 1
                    total_pixels += 1
    
                elif isinstance(lbl, np.ndarray) and lbl.ndim == 1 and lbl.shape[0] == num_classes:
                    one_hot = (lbl > 0).astype(int)
                    sample_pos += one_hot
                    sample_total += 1
                    pixel_counts += one_hot
                    total_pixels += num_classes
    
                elif isinstance(lbl, np.ndarray) and lbl.ndim in (2, 3):
                    if lbl.ndim == 3 and (lbl.shape[0] == 1 or lbl.shape[-1] == 1):
                        lbl = np.squeeze(lbl, axis=0 if lbl.shape[0] == 1 else -1)
                    if lbl.ndim == 2:
                        lbl_int = lbl
                    elif lbl.shape[0] == num_classes:
                        lbl_int = np.argmax(lbl, axis=0)
                    elif lbl.shape[-1] == num_classes:
                        lbl_int = np.argmax(lbl, axis=2)
                    else:
                        raise ValueError(f"Unexpected mask shape {lbl.shape} for sample {sid}")
                    flat = lbl_int.ravel()
                    total_pixels += flat.size
                    for c in range(num_classes):
                        pixel_counts[c] += int((flat == c).sum())
                else:
                    raise ValueError(f"Unsupported label shape {getattr(lbl, 'shape', type(lbl))} for sample {sid}")
            except Exception as e:
                print(f"Sample {sid} has no 'label' field")
    
        if sample_total > 0:
            freq = sample_pos.astype(float)
            freq[freq == 0] = 1.0
            total_s = freq.sum()
            class_weights = (total_s / (freq * num_classes)).astype(np.float32)
        else:
            freq = pixel_counts.astype(float)
            freq[freq == 0] = 1.0
            total_p = freq.sum()
            class_weights = (total_p / (freq * num_classes)).astype(np.float32)
    
        class_weights = torch.from_numpy(class_weights)
    
        if sample_total > 0:
            neg = sample_total - sample_pos.astype(float)
            pos = sample_pos.astype(float)
        else:
            neg = total_pixels - pixel_counts.astype(float)
            pos = pixel_counts.astype(float)
    
        pos[pos == 0] = 1.0
        pos_weight = torch.from_numpy((neg / pos).astype(np.float32))
    
        return class_weights, pos_weight
        
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
    n_shot: int= 0
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
        assert len(split) == len(callback_pre_augmentation) == len(callback_post_augmentation) == len(augmentations), \
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
        n_shot=n_shot
        #patch_size=(256,256)
    )
    img, label = dataset[0]['img'], dataset[0]['label']
    print(f"Dataset {dataset_set} shapes: img={img.shape}, label={label.shape}")
    weights, pos_weights = dataset.compute_class_and_pos_weights(num_classes)
    print(f"weights: {weights}, pos_weights:{pos_weights}")
    if not split is None:
        sub_datasets = dataset.split_by_percentages(split=split, num_classes=num_classes)
        dataloaders = []
        for idx, sub_dataset in enumerate(sub_datasets):
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
        split = [.9, .1], 
        callback_pre_augmentation = [callback_pre_augmentation_training, callback_pre_augmentation_val],
        callback_post_augmentation = [callback_post_augmentation_training, callback_post_augmentation_val],
        augmentations = [augmentations_training, augmentations_val], 
        crop_images= crop_images, 
        generator= torch.Generator(device), 
        pin_memory=True, 
        drop_last=False, 
        n_shot=500,
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
        n_shot=500,
        num_classes=num_classes
    ) 
