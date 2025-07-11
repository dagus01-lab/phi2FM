import zarr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union, Callable
import random
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
    """
    PyTorch Dataset for loading data from a Zarr dataset created with add_sample function.
    
    Args:
        zarr_path (str): Path to the Zarr dataset.
        dataset_set (str): One of {"trainval", "test"} (case-insensitive).
        callback_pre_augmentation (callable, optional): Transformation to apply before augmentation
        callback_post_augmentation (callable, optional): Transformation to apply before augmentation
        augmentations (list[callable], optional): List of augmentations to be applied to the dataset samples
        task_filter (str, optional): If provided, only load samples with this task.
        metadata_keys (list, optional): List of metadata keys to include in sample.
        crop_images (bool, optional): boolean flag indicating whether to crop samples or not
    """
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
        crop_images: bool = False
    ):
        self.root = zarr.open(zarr_path, mode='r')
        self.dataset_set = dataset_set.lower()
        #self.transform = transform
        #self.target_transform = target_transform
        self.task_filter = task_filter.lower() if task_filter else None
        self.metadata_keys = metadata_keys or []
        self.verbose = verbose
        self.crop_images = crop_images

        self.callback_pre_augmentation = callback_pre_augmentation
        self.callback_post_augmentation = callback_post_augmentation
        self.augmentations = augmentations
        
        # Verify the dataset exists
        if self.dataset_set not in self.root:
            raise ValueError(f"Dataset set '{dataset_set}' not found in Zarr store")
            
        self.dataset_group = self.root[self.dataset_set]
        
        # Get all sample IDs
        self.sample_ids = sorted(self.dataset_group.keys())
        
        if self.verbose:
            print(f"Found {len(self.sample_ids)} total samples in {dataset_set}")
            
        # Get available tasks for debugging
        if self.verbose:
            tasks = {}
            for sid in self.sample_ids[:min(100, len(self.sample_ids))]:
                task = self.dataset_group[sid].attrs.get('task', '')
                tasks[task] = tasks.get(task, 0) + 1
            print(f"Available tasks in first 100 samples: {tasks}")
        
        # Filter by task if needed
        if self.task_filter:
            original_count = len(self.sample_ids)
            self.sample_ids = [
                sid for sid in self.sample_ids 
                if self.dataset_group[sid].attrs.get('task', '') == self.task_filter
            ]
            if self.verbose:
                print(f"Filtered to {len(self.sample_ids)} samples with task '{self.task_filter}'")
                
            if len(self.sample_ids) == 0:
                available_tasks = set()
                for sid in list(self.dataset_group.keys())[:min(100, len(list(self.dataset_group.keys())))]:
                    available_tasks.add(self.dataset_group[sid].attrs.get('task', ''))
                raise ValueError(
                    f"No samples found with task '{self.task_filter}'. "
                    f"Available tasks: {available_tasks}"
                )
    
    def __len__(self) -> int:
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        sample_id = self.sample_ids[idx]
        sample_group = self.dataset_group[sample_id]
    
        # Load image
        img = sample_group['img'][:]
    
        # Load label (handle scalar or array)
        label_array = sample_group['label']
        label = label_array[()] if label_array.shape == () else label_array[:]
        if img.ndim == 3:
            if img.shape[2] > img.shape[0] :
                img = img.transpose(1, 2, 0)
            elif img.shape[1] < img.shape[2]:
                img = img.transpose(0, 2, 1)
        if label.ndim == 3:
            if label.shape[2] > label.shape[0] :
                label = label.transpose(1, 2, 0)
            elif label.shape[1] < label.shape[2]:
                label = label.transpose(0, 2, 1)
        if self.crop_images:
            if img.ndim >= 3:
                img = img[:64, :64, :] if img.ndim == 3 else img[:, :64, :64, :]
            if label.ndim >= 3:
                label = label[:64, :64, :] if label.ndim == 3 else label[:, :64, :64, :]
        if self.callback_pre_augmentation:
            img, label = self.callback_pre_augmentation(img.copy(), label.copy())
        if not self.augmentations is None and (isinstance(self.augmentations, list) or isinstance(self.augmentations, tuple)): 
            for aug in self.augmentations:
                img, label = aug(img.copy(), label.copy())
        #print(f"item {idx} - shapes before post callback: {img.shape}, {label.shape}")
        if self.callback_post_augmentation:
            img, label = self.callback_post_augmentation(img.copy(), label.copy())
        # Apply transforms
        #img = self.transform(img) if self.transform else torch.from_numpy(img)
        #label = self.target_transform(label) if self.target_transform else torch.tensor(label)
        
    
        # Task and metadata
        task = sample_group.attrs.get('task', '')
        sample = {'img': img, 'label': label, 'task': task, 'sample_id': sample_id}
    
        for key in self.metadata_keys:
            if key in sample_group.attrs:
                sample[key] = sample_group.attrs[key]
    
        return sample
        
    def split_by_percentages(self, split: List[float], num_classes: int = None) -> List['PhiSatDataset']:
        """
        Stratified split preserving class proportions when possible.

        Args:
            split: list of floats summing to 1.0, e.g. [0.8, 0.2]
            num_classes: total number of classes (required for one-hot labels)

        Returns:
            List of PhiSatDataset subsets
        """
        if not np.isclose(sum(split), 1.0):
            raise ValueError("Split percentages must sum to 1.0")

        # Group samples by a representative label for stratification
        class_to_ids: Dict[int, List[str]] = {}
        for sid in self.sample_ids:
            grp = self.dataset_group[sid]
            arr = grp['label']
            # Load label
            if hasattr(arr, 'shape') and len(arr.shape) == 0:
                lbl = arr[()]
            else:
                lbl = arr[:]

            # Determine class for this sample
            cls = -1
            # Single-label scalar
            if np.isscalar(lbl) or (isinstance(lbl, np.ndarray) and lbl.ndim == 0):
                cls = int(lbl)
            # One-hot vector (image-level multi-class)
            elif isinstance(lbl, np.ndarray) and lbl.ndim == 1 and num_classes and lbl.shape[0] == num_classes:
                one_hot = (lbl > 0)
                if one_hot.sum() == 1:
                    cls = int(np.argmax(one_hot))
                else:
                    # multi-label: assign to first positive
                    cls = int(np.argmax(one_hot))
            # Segmentation mask (pixel-wise)
            elif isinstance(lbl, np.ndarray) and lbl.ndim in [2,3] and num_classes:
                # Convert to integer mask
                if lbl.ndim == 2:
                    mask = lbl
                else:
                    # detect channel dimension
                    if lbl.shape[0] == num_classes:
                        mask = np.argmax(lbl, axis=0)
                    elif lbl.shape[2] == num_classes:
                        mask = np.argmax(lbl, axis=2)
                    else:
                        mask = None
                if mask is not None:
                    vals, counts = np.unique(mask, return_counts=True)
                    cls = int(vals[np.argmax(counts)])
            # Fallback remains cls=-1

            class_to_ids.setdefault(cls, []).append(sid)

        # If only fallback group, do random split
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

        # Stratified allocation
        per_split_ids: Dict[int, List[str]] = {i: [] for i in range(len(split))}
        for cls, ids in class_to_ids.items():
            random.shuffle(ids)
            n = len(ids)
            sizes = [int(r * n) for r in split]
            sizes[-1] = n - sum(sizes[:-1])
            start = 0
            for i, size in enumerate(sizes):
                per_split_ids[i].extend(ids[start:start+size])
                start += size

        # Build subsets
        subsets: List[PhiSatDataset] = []
        for i in range(len(split)):
            subset = deepcopy(self)
            subset.sample_ids = per_split_ids[i]
            subsets.append(subset)
        return subsets

        # Shuffle and allocate stratified splits
        per_split_ids: Dict[int, List[str]] = {i: [] for i in range(len(split))}
        for cls, ids in class_to_ids.items():
            random.shuffle(ids)
            n = len(ids)
            sizes = [int(r * n) for r in split]
            sizes[-1] = n - sum(sizes[:-1])
            start = 0
            for i, size in enumerate(sizes):
                per_split_ids[i].extend(ids[start:start+size])
                start += size

        # Build subset datasets
        subsets: List[PhiSatDataset] = []
        for i in range(len(split)):
            subset = deepcopy(self)
            subset.sample_ids = per_split_ids[i]
            subsets.append(subset)
        return subsets


        
    def compute_class_and_pos_weights(self, num_classes: int):
    
        # Counters for segmentation (pixel‐wise) and classification (image‐wise)
        pixel_counts   = np.zeros((num_classes,), dtype=np.int64)
        total_pixels   = 0
        sample_pos     = np.zeros((num_classes,), dtype=np.int64)
        sample_total   = 0
    
        for sid in self.sample_ids:
            grp = self.dataset_group[sid]
            try:
                arr = grp['label']
        
                # Read the label, whether 0‑D or array
                if hasattr(arr, 'shape') and len(arr.shape) == 0:
                    lbl = arr[()]
                else:
                    lbl = arr[:]
        
                # Case A: scalar or 0‑D array → single‐label classification
                if np.isscalar(lbl) or (isinstance(lbl, np.ndarray) and lbl.ndim == 0):
                    cls = int(lbl)
                    if not (0 <= cls < num_classes):
                        raise ValueError(f"Label {cls} out of range for sample {sid}")
                    sample_pos[cls] += 1
                    sample_total    += 1
                    pixel_counts[cls] += 1
                    total_pixels   += 1
        
                # Case B: 1‑D one‑hot vector → multi‑label classification
                elif isinstance(lbl, np.ndarray) and lbl.ndim == 1 and lbl.shape[0] == num_classes:
                    one_hot = (lbl > 0).astype(int)
                    sample_pos += one_hot
                    sample_total += 1
                    pixel_counts += one_hot
                    total_pixels  += num_classes
        
                # Case C: pixel‐wise mask
                elif isinstance(lbl, np.ndarray) and lbl.ndim in (2, 3):
                    # If 3D with a singleton channel, squeeze it
                    if lbl.ndim == 3 and (lbl.shape[0] == 1 or lbl.shape[-1] == 1):
                        lbl = np.squeeze(lbl, axis=0 if lbl.shape[0] == 1 else -1)
        
                    # After squeezing, if 2D → integer mask
                    if lbl.ndim == 2:
                        lbl_int = lbl
                    # If still 3D → true one‑hot with C=num_classes
                    else:
                        if lbl.shape[0] == num_classes:       # (C, H, W)
                            lbl_int = np.argmax(lbl, axis=0)
                        elif lbl.shape[-1] == num_classes:    # (H, W, C)
                            lbl_int = np.argmax(lbl, axis=2)
                        else:
                            raise ValueError(f"Unexpected mask shape {lbl.shape} for sample {sid}")
        
                    flat = lbl_int.ravel()
                    total_pixels += flat.size
                    # count pixels per class
                    for c in range(num_classes):
                        pixel_counts[c] += int((flat == c).sum())
        
                else:
                    raise ValueError(f"Unsupported label shape {getattr(lbl, 'shape', type(lbl))} for sample {sid}")
            except Exception as e:
                print(f"Sample {sid} has no 'label' field")
    
        # — Compute class weights for CrossEntropyLoss ——
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
    
        # — Compute pos_weight for BCEWithLogitsLoss ——
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
    #transform: Optional[Callable] = None,
    #target_transform: Optional[Callable] = None,
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
    drop_last: bool = False
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
    dataset = PhiSatDataset(
        zarr_path=zarr_path,
        dataset_set=dataset_set,
        #transform=transform,
        #target_transform=target_transform,
        task_filter=task_filter,
        metadata_keys=metadata_keys,
        verbose=verbose, 
        crop_images=crop_images
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
                    collate_fn=adv_collate_fn,
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
                collate_fn=adv_collate_fn,
                generator=generator, 
                pin_memory=pin_memory, 
                drop_last=drop_last
            )
        

class NormalizeChannels:
    """Normalize each channel to [0, 1] or using mean/std"""
    def __init__(self, means=None, stds=None, min_max=True):
        self.means = means
        self.stds = stds
        self.min_max = min_max
        
    def __call__(self, img):
        """
        Args:
            img: numpy array of shape (C, H, W)
        """
        if self.min_max:
            # Min-max normalization per channel
            result = torch.from_numpy(img.copy())
            for c in range(img.shape[0]):
                min_val = torch.min(result[c])
                max_val = torch.max(result[c])
                if max_val > min_val:  # Avoid division by zero
                    result[c] = (result[c] - min_val) / (max_val - min_val)
            return result
        else:
            # Mean-std normalization
            img_tensor = torch.from_numpy(img)
            for c in range(img.shape[0]):
                mean = self.means[c] if self.means else img_tensor[c].mean()
                std = self.stds[c] if self.stds else img_tensor[c].std()
                if std > 0:  # Avoid division by zero
                    img_tensor[c] = (img_tensor[c] - mean) / std
            return img_tensor
        
        
if __name__ == "__main__":
    # Example usage
    zarr_path = "burned_area_dataset.zarr"
    
    dataset_set = "trainval"
    # Create DataLoader
    dataloader = get_zarr_dataloader(
        zarr_path=zarr_path,
        dataset_set=dataset_set,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        #transform=NormalizeChannels(min_max=True),
        metadata_keys=["sensor", "timestamp"],
    )
    
    # Iterate through batches
    for batch in dataloader:
        # Access data based on tasks in the batch
        for task in batch['tasks']:
            images = batch[f'{task}_img']
            labels = batch[f'{task}_label']
            # Forward pass, compute loss, etc.