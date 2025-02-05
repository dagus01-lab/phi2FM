import random
import numpy as np
import lmdb

# PyTorch
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.utils.data.distributed import DistributedSampler


class TransformX:
    def __init__(self, augmentations=True, input_size=None,
                 min_scaling=np.array([0.]*8), max_scaling=np.array([1.]*8),
                 means=np.array([0.]*8), stds=np.array([1.]*8),
                 rot_prob=0.25, flip_prob=0.25, noise_prob=0.25,
                 noise_std_range=(0.005, 0.015)):
        """
        Args:
            means (np.ndarray): Channel-wise means. Shape: (C,)
            stds (np.ndarray): Channel-wise stds. Shape: (C,)
            augmentations (bool): Whether to apply augmentations.
            rot_prob (float): Probability of applying rotation.
            flip_prob (float): Probability of applying flips.
            noise_prob (float): Probability of applying noise.
            noise_std_range (tuple): (min_std, max_std) for noise.
                NOTE: Here these values are interpreted relative to the [0,1]
                scale. Since we will add noise in the original domain,
                we multiply by (max - min) later.
            input_size (int, optional): If specified and the image is larger than 
                input_size x input_size, randomly crop the image down to that size.
        """
        self.augmentations = augmentations
        
        # Ensure arrays are 1D
        means = means.reshape(-1)
        stds = stds.reshape(-1)
        min_scaling = min_scaling.reshape(-1)
        max_scaling = max_scaling.reshape(-1)

        # Precompute scaled means and stds for later normalization.
        self.scaled_means = ((means - min_scaling) / (max_scaling - min_scaling))[:, None, None]
        self.scaled_stds = (stds / (max_scaling - min_scaling))[:, None, None]

        self.channel_range = (max_scaling - min_scaling)[:, None, None]  # Shape: (C,1,1)
        self.min_scaling = min_scaling[:, None, None]
        self.max_scaling = max_scaling[:, None, None]

        self.rot_prob = rot_prob
        self.flip_prob = flip_prob
        self.noise_prob = noise_prob
        self.noise_std_low, self.noise_std_high = noise_std_range
        
        # Determine sizes for cropping
        # We assume input_size is the original spatial size.
        # new_size is the desired crop size.
        self.input_size = input_size
        self.original_size = 256 if input_size > 128 else 128


    def __call__(self, x_np):
        # x_np is assumed to be a NumPy array of shape (C, H, W)

        # --- Spatial augmentations ---
        if self.augmentations:
            rand_vals = np.random.rand(3)

            # Random Crop if the input is larger than the desired crop size.
            if self.original_size > self.input_size:
                top = np.random.randint(0, self.original_size - self.input_size + 1)
                left = np.random.randint(0, self.original_size - self.input_size + 1)
                x_np = x_np[:, top:top+self.input_size, left:left+self.input_size]

            # Rotation: 0, 90, 180, or 270 degrees
            if rand_vals[0] < self.rot_prob:
                k = np.random.randint(0, 4)
                x_np = np.rot90(x_np, k, axes=(1, 2))

            # Horizontal flip
            if rand_vals[1] < self.flip_prob:
                x_np = np.flip(x_np, axis=2)

            # Vertical flip
            if rand_vals[2] < self.flip_prob:
                x_np = np.flip(x_np, axis=1)
        else:
            # If no augmentations, perform a center crop if needed.
            if self.original_size > self.input_size:
                top = (self.original_size - self.input_size) // 2
                left = (self.original_size - self.input_size) // 2
                x_np = x_np[:, top:top+self.input_size, left:left+self.input_size]

        # --- Data Pre-Processing ---

        # 1. Clip the raw values to the valid range in the original domain.
        #    Use broadcasting over the spatial dimensions.
        x_np = np.clip(x_np, self.min_scaling, self.max_scaling)

        # 2. Add noise in the original domain.
        if self.augmentations and np.random.rand() < self.noise_prob:
            # Because noise_std_range is defined relative to the [0,1] scale,
            # we multiply by the channel-wise range to convert to the original scale.
            # Generate a per-channel noise std (and broadcast to H,W).
            noise_std = np.random.uniform(self.noise_std_low, self.noise_std_high, size=(x_np.shape[0], 1, 1)) * self.channel_range
            noise = np.random.normal(0, noise_std, size=x_np.shape)
            x_np = x_np + noise
            # Optionally, clip again to ensure values stay in the valid range.
            x_np = np.clip(x_np, self.min_scaling, self.max_scaling)

        # 3. Scale the data from the original range to [0, 1].
        x_np = (x_np - self.min_scaling) / self.channel_range

        # 4. Normalize to zero mean and unit variance (per channel).
        x_np = (x_np - self.scaled_means) / self.scaled_stds

        return x_np




class LmdbDataset(Dataset):
    def __init__(
        self,
        lmdb_path,
        transform_x=None,
        apply_zoom_task=True,
        fixed_task=None,
        zoom_range=(1.0, 1.5),
        augment_drop=None,
        device="cpu",
        clip_values=(0.0, 1.0),
    ):
        """
        Args:
            lmdb_path (str): Path to the LMDB file.
            transform_x (callable, optional): Optional transform on x_data (NumPy based).
            apply_zoom_task (bool): Whether to apply the zoom-level prediction task.
            fixed_task (str, optional): Specific task ('coords', 'climate', or None for all tasks).
            zoom_range (tuple): Range (min_zoom, max_zoom) for zoom factor.
            augment_drop (callable): Transformations (like RandomErasing).
            device (str): 'cpu' or 'cuda' (not really used now, since we do everything in NumPy).
            clip_values (tuple): (min_val, max_val) for final clipping.
        """
        self.lmdb_path = lmdb_path
        self.transform_x = transform_x
        self.apply_zoom_task = apply_zoom_task
        self.fixed_task = fixed_task
        self.zoom_range = zoom_range
        self.augment_drop = augment_drop
        self.device = device
        self.clip_values = clip_values
        self.env = None
        self.txn = None

        # Determine dataset length using a temporary environment
        with lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False) as temp_env:
            with temp_env.begin(write=False) as temp_txn:
                self.length = temp_txn.stat()["entries"] // 3  # 3 keys per record: image, coords, climate


    def __len__(self):
        return self.length


    def init_worker(self):
        """
        Initialize the LMDB environment and transaction for this worker.
        """
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        if self.txn is None:
            self.txn = self.env.begin(write=False)


    def numpy_zero_to_noise(self, image_np):
        """
        Applies random erasing transformations (augment_drop) to the combined image and a white image to identify erased areas.
        Then replaces the erased areas in the original image with noise.

        image_np: (C, H, W) NumPy array
        """
        mean_val = image_np.mean()
        std_val = image_np.std() + 1e-6

        noise = np.random.normal(mean_val, std_val, size=image_np.shape)
        noise = np.clip(noise, image_np.min(), image_np.max())

        # Create a white image
        white = np.ones_like(image_np)

        # Concatenate original and white along the channel dimension: (2*C, H, W)
        merged = np.concatenate([image_np, white], axis=0)

        # Apply the custom augment_drop transform if available
        if self.augment_drop is not None:
            dropped = self.augment_drop(merged)  # Apply random erasing on (2*C, H, W)
        else:
            dropped = merged

        C, _, _ = image_np.shape
        # Identify erased areas in the white image part
        erased_mask = (dropped[C:2*C, :, :] == 0)

        # Replace erased areas in the original with noise
        reconstructed = np.where(erased_mask, noise, dropped[:C, :, :])

        return reconstructed
    
    def np_to_tensor(self, np_x, dtype=torch.float32):
        np_x = np.copy(np_x)
        torch_x = torch.from_numpy(np_x)
        torch_x = torch_x.to(dtype=dtype)
        if self.device != "cpu":
            torch_x = torch_x.to(self.device)
        return torch_x

    def __getitem__(self, idx):
        if self.env is None or self.txn is None:
            self.init_worker()

        # Get the record from the LMDB
        image_data = self.txn.get(f"{idx:08d}_image".encode("ascii"))
        coords_data = self.txn.get(f"{idx:08d}_coords".encode("ascii"))
        climate_data = self.txn.get(f"{idx:08d}_climate".encode("ascii"))

        # Deserialize
        x = np.frombuffer(image_data, dtype=np.uint16).reshape((8, 256, 256))
        y_coords = np.frombuffer(coords_data, dtype=np.float64)
        y_climate = np.frombuffer(climate_data, dtype=np.uint8)[0]

        y = {'coords': None, 'climate': None, 'reconstruction': None}

        # ---------------------------
        # Transform X
        # ---------------------------
        if self.transform_x is not None:
            x = self.transform_x(x)

        # ---------------------------
        # Coords Task
        # ---------------------------
        if self.fixed_task is None or self.fixed_task == 'coords':
            y['coords'] = self.np_to_tensor(y_coords)

        # ---------------------------
        # Climate Task
        # ---------------------------
        if self.fixed_task is None or self.fixed_task == 'climate': 
            y['climate'] = self.np_to_tensor(y_climate, dtype=torch.long)

        # ---------------------------
        # Reconstruction Task
        # ---------------------------
        if (self.fixed_task is None or self.fixed_task == 'reconstruction') and self.augment_drop is not None:
            x_original = x.copy()
            x = self.numpy_zero_to_noise(x)
            y['reconstruction'] = self.np_to_tensor(x_original)

        x_torch = self.np_to_tensor(x)
        return x_torch, y




class NumpyGridPatchMask:
    def __init__(self, image_size=(3, 256, 256), patch_size=16, mask_fraction=0.2, mask_value=0):
        """
        A transform that masks out a fraction of patches in a grid.

        Args:
            image_size (tuple): The shape of the image in (C, H, W) format.
            patch_size (int): The size of the patch (square) along each spatial dimension.
            mask_fraction (float): The fraction of patches to mask (e.g., 0.2 means 20% of the patches).
            mask_value (int or float): The value to fill the masked patches with.
        """
        self.C, self.H, self.W = image_size
        self.patch_size = patch_size
        self.mask_fraction = mask_fraction
        self.mask_value = mask_value
        
        # Calculate how many patches fit along each dimension
        self.num_patches_h = self.H // patch_size
        self.num_patches_w = self.W // patch_size
        
        # Precompute all patch indices in a grid
        self.patch_indices = [
            (row_idx, col_idx) 
            for row_idx in range(self.num_patches_h) 
            for col_idx in range(self.num_patches_w)
        ]

    def __call__(self, img):
        """
        Args:
            img (np.ndarray): The input image array with shape (C, H, W).
        
        Returns:
            np.ndarray: The image with certain patches masked out.
        """
        # Determine how many patches to mask
        total_patches = len(self.patch_indices)
        if type(self.mask_fraction) == tuple:
            mask_perc = random.uniform(*self.mask_fraction)
        else:
            mask_perc = self.mask_fraction

        num_to_mask = int(round(total_patches * mask_perc))
        
        # Randomly sample the patches to be masked
        patches_to_mask = random.sample(self.patch_indices, num_to_mask)
        
        # Copy the input to avoid in-place operations
        masked_img = img.copy()
        
        # Mask out the chosen patches
        for (row, col) in patches_to_mask:
            y_start = row * self.patch_size
            y_end   = y_start + self.patch_size
            x_start = col * self.patch_size
            x_end   = x_start + self.patch_size
            
            masked_img[:, y_start:y_end, x_start:x_end] = self.mask_value
        
        return masked_img





def load_foundation_data(lmdb_path_train, lmdb_path_val, lmdb_path_test, lmdb_path_inference, 
                         device_dataset, device_dataloader, with_augmentations=True, 
                         num_workers=0, batch_size=16, input_size=None, fixed_task=None,
                         use_ddp=False, rank=0, world_size=1,
                         split_ratio=1.0, n_shot=None,
                         ):
    
    # ---------------------------
    # 1. Define Transforms
    # ---------------------------
    augment_drop = NumpyGridPatchMask(
        image_size=(8, input_size, input_size), 
        patch_size=16, 
        mask_fraction=0.75, 
        mask_value=0
    )

    global_min = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
    global_max = np.array([27053., 27106., 27334., 27068., 27273., 27496., 27618., 27409.])
    global_mean = np.array([1889.57135146, 1706.35570957, 1829.54409057, 1864.05266573, 2378.13355846, 1974.74770695, 2309.17435277, 2472.06254275])
    global_std = np.array([1926.40004038, 1770.64430483, 2083.48230285, 1916.71983995, 2008.31424611, 2109.32162828, 2074.35633945, 2078.41143301])


    transform_x_train = TransformX(augmentations=with_augmentations,
                                   input_size=input_size,
                                   min_scaling=global_min,
                                   max_scaling=global_max,
                                   means=global_mean,
                                   stds=global_std,
                                   )

    transform_x_test = TransformX(augmentations=False,
                                   input_size=input_size,
                                   min_scaling=global_min,
                                   max_scaling=global_max,
                                   means=global_mean,
                                   stds=global_std,
                                   )
    
    # ---------------------------
    # 2. Load Datasets
    # ---------------------------
    dataset_train = LmdbDataset(
        lmdb_path=lmdb_path_train,
        transform_x=transform_x_train,
        augment_drop=augment_drop,
        device=device_dataset,
        apply_zoom_task=False,
        fixed_task=None # DID NOT IMPLEMENT THIS...
    )
    
    dataset_val = LmdbDataset(
        lmdb_path=lmdb_path_val,
        transform_x=transform_x_test,
        augment_drop=augment_drop,
        device=device_dataset,
        apply_zoom_task=False,
        fixed_task=None
    )
    
    dataset_test = LmdbDataset(
        lmdb_path=lmdb_path_test,
        transform_x=transform_x_test,
        augment_drop=augment_drop,
        device=device_dataset,
        apply_zoom_task=False,
        fixed_task=None
    )
    
    dataset_inference = LmdbDataset(
        lmdb_path=lmdb_path_inference,
        transform_x=transform_x_test,
        augment_drop=augment_drop,
        device=device_dataset,
        apply_zoom_task=False,
        fixed_task=None
    )


    # ---------------------------
    # 3. Create Split of Dataset
    # ---------------------------
    assert not (n_shot == None) or not (split_ratio == None), 'Please define data partition protocol!'
    
    if split_ratio is not None:
        if split_ratio < 1.0:
            train_size = int(len(dataset_train) * split_ratio)
            unused_size = len(dataset_train) - train_size
            dataset_train, _ = random_split(dataset_train, [train_size, unused_size], generator=torch.Generator(device=device_dataloader))
            
            val_size = int(len(dataset_val) * split_ratio)
            unused_size = len(dataset_val) - val_size
            dataset_val, _ = random_split(dataset_val, [val_size, unused_size], generator=torch.Generator(device=device_dataloader))
            
            test_size = int(len(dataset_test) * split_ratio)
            unused_size = len(dataset_test) - test_size
            dataset_test, _ = random_split(dataset_test, [test_size, unused_size], generator=torch.Generator(device=device_dataloader))
    
    elif n_shot is not None:
        dataset_train = Subset(dataset_train, range(n_shot))
        
        n_shot_val = int(n_shot / len(dataset_train) * len(dataset_val))
        dataset_val = Subset(dataset_val, range(n_shot_val))

    # ---------------------------
    # 4. Use DistributedSampler
    # ---------------------------
    if use_ddp:
        train_sampler = DistributedSampler(
            dataset_train, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            dataset_val, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=False
        )
        test_sampler = DistributedSampler(
            dataset_test, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=False
        )
        inference_sampler = DistributedSampler(
            dataset_inference, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=False
        )
    else:
        # Normal sampling
        train_sampler = None
        val_sampler = None
        test_sampler = None
        inference_sampler = None


    # ---------------------------
    # 5. Create DataLoaders
    # ---------------------------

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=(not use_ddp),  # let the sampler shuffle if DDP
        sampler=train_sampler,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
        generator=torch.Generator(device=device_dataloader),
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
        generator=torch.Generator(device=device_dataloader),
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
        generator=torch.Generator(device=device_dataloader),
    )
    dataloader_inference = DataLoader(
        dataset_inference,
        batch_size=batch_size,
        shuffle=False,
        sampler=inference_sampler,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
        generator=torch.Generator(device=device_dataloader),
    )
    
    return dataloader_train, dataloader_val, dataloader_test, dataloader_inference