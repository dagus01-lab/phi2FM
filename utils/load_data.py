# Standard Library
import os
from glob import glob
import lmdb
import pickle

# External Libraries
import buteo as beo
import numpy as np

# PyTorch
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.utils.data.distributed import DistributedSampler

from utils import config_lc
from utils import Prithvi_100M_config

import random
from torchvision import transforms
import math

import cv2 


# statistics used to normalize images before passing to the model
MEANS_PRITHVI = np.array(Prithvi_100M_config.data_mean).reshape(1, 1, -1)
STDS_PRITHVI = np.array(Prithvi_100M_config.data_std).reshape(1, 1, -1)
MEANS_PRITHVI_PHI2 = np.array(Prithvi_100M_config.data_mean_phi2).reshape(1, 1, -1)
STDS_PRITHVI_PHI2 = np.array(Prithvi_100M_config.data_std_phi2).reshape(1, 1, -1)

LC_MAP = config_lc.lc_model_map
# order S2 bands: 0-B02, 1-B03, 2-B04, 3-B08, 4-B05, 5-B06, 6-B07, 7-B8A, 8-B11, 9-B12
MEANS_SATMAE = np.array([1184.3824625, 1120.77120066, 1136.26026392, 1762.59530783, 1263.73947144, 1645.40315151,
                        1846.87040806, 1972.62420416, 1732.16362238, 1247.91870117])

STDS_SATMAE = np.array([650.2842772, 965.23119807,  948.9819932, 1364.38688993, 1108.06650639, 1258.36394548,
                       1233.1492281, 3545.66, 1310.36996126, 1087.6020813])

MEANS_SATMAE_PHI2 = 10000 * np.array([0.1072132 , 0.10218794, 0.0983548 , 0.22145009, 0.12199436, 0.19247645, 0.22734961, 0. , 0. , 0. ])

STDS_SATMAE_PHI2 = 10000 *  np.array([0.04608911, 0.04950009, 0.07192364, 0.10286225, 0.07146991, 0.08716079, 0.1045232 , 0. , 0. , 0. ])

MIN_MAJORTOM = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
MAX_MAJORTOM = np.array([1.41421356, 1.41421356, 1.41421356, 1.41421356, 1.41421356, 1.41421356, 1.41421356, 1.41421356])


PROCESS_PHISAT = True


def to_one_hot_lc(y, class_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])):
    y_classification = np.isin(class_labels, y).astype(np.float32)
    return y_classification


def to_one_hot_building(y):
    mean_value = np.mean(y > 0)
    if mean_value < 0 or mean_value > 1:
        raise ValueError('Invalid values in building mask')

    classes = [mean_value == 0, 0 < mean_value <= 0.3, 0.3 < mean_value <= 0.6, 0.6 < mean_value <= 0.9, mean_value > 0.9]    
    y_classification = np.array([float(x) for x in classes], dtype=np.float32)
    return y_classification



def sentinelNormalize(x):
    if PROCESS_PHISAT:
        if x.shape[2] == 8:
            x = np.delete(x, 3, axis=2)
            zeros_shape = (x.shape[0], x.shape[1], 3)
            zeros = np.zeros(zeros_shape, dtype=x.dtype)
            x = np.concatenate((x, zeros), axis=2)
            
        min_value = MEANS_SATMAE_PHI2 - 2 * STDS_SATMAE_PHI2
        max_value = MEANS_SATMAE_PHI2 + 2 * STDS_SATMAE_PHI2
    
    else:
        min_value = MEANS_SATMAE - 2 * STDS_SATMAE
        max_value = MEANS_SATMAE + 2 * STDS_SATMAE

    img = (x - min_value) / (max_value - min_value + 1e-8) * 255.0
    img = np.clip(img, 0, 255).astype(np.float32)
    return img

def preprocess_image_prithvi(image):
    if PROCESS_PHISAT:
        if image.shape[2] == 8:
            image = np.delete(image, 3, axis=2)
            zeros_shape = (image.shape[0], image.shape[1], 3)
            zeros = np.zeros(zeros_shape, dtype=image.dtype)
            image = np.concatenate((image, zeros), axis=2)
        
        normalized = image.copy()
        normalized = ((image - MEANS_PRITHVI_PHI2) / STDS_PRITHVI_PHI2)

    else:
        # normalize image
        normalized = image.copy()
        normalized = ((image - MEANS_PRITHVI) / STDS_PRITHVI)
    normalized = normalized.astype(np.float32, copy=False)

    # normalized = torch.from_numpy(normalized.reshape(1, normalized.shape[0], 1, *normalized.shape[-2:])).to(torch.float32)
    return normalized

def callback_preprocess(x, y):
    if PROCESS_PHISAT:
        if x.shape[2] == 8:
            x = np.delete(x, 3, axis=2)
            zeros_shape = (x.shape[0], x.shape[1], 3)
            zeros = np.zeros(zeros_shape, dtype=x.dtype)
            x = np.concatenate((x, zeros), axis=2)

    x_norm = np.empty_like(x, dtype=np.float32)
    np.divide(x, 10000.0, out=x_norm)

    y = y.astype(np.float32, copy=False)

    return x_norm, y


def callback_preprocess_satmae(x, y):
    if PROCESS_PHISAT:
        if x.shape[2] == 8:
            x = np.delete(x, 3, axis=2)
            zeros_shape = (x.shape[0], x.shape[1], 3)
            zeros = np.zeros(zeros_shape, dtype=x.dtype)
            x = np.concatenate((x, zeros), axis=2)
    x_norm = sentinelNormalize(x)
    y = y.astype(np.float32, copy=False)

    x_norm = x_norm[16:-16, 16:-16, :]
    if len(y.shape) > 2:
        y = y[16:-16, 16:-16, :]
    return x_norm, y


def callback_preprocess_prithvi(x, y):
    # order S2 bands: 0-B02, 1-B03, 2-B04, 3-B08, 4-B05, 5-B06, 6-B07, 7-B8A, 8-B11, 9-B12
    # HLS bands: 0-B02, 1-B03, 2-B04, 4-B05, 5-B06, 6-B07,
    if PROCESS_PHISAT:
        x = x[:, :, (0, 1, 2, 5, 6, 7)] 
    else:
        x = x[:, :, (0, 1, 2, 4, 5, 6)] 
    x_norm = preprocess_image_prithvi(x)
    y = y.astype(np.float32, copy=False)

    return x_norm, y


def callback_preprocess_landcover(x, y):
    if PROCESS_PHISAT:
        if x.shape[2] == 8:
            x = np.delete(x, 3, axis=2)
            zeros_shape = (x.shape[0], x.shape[1], 3)
            zeros = np.zeros(zeros_shape, dtype=x.dtype)
            x = np.concatenate((x, zeros), axis=2)

    x_norm = np.empty_like(x, dtype=np.float32)
    np.divide(x, 10000.0, out=x_norm)

    u, inv = np.unique(y, return_inverse=True)
    y = np.array([LC_MAP[val] for val in u])[inv].reshape(y.shape)
    
    return x_norm, y


def callback_preprocess_building_classification(x, y):
    if PROCESS_PHISAT:
        if x.shape[2] == 8:
            x = np.delete(x, 3, axis=2)
            zeros_shape = (x.shape[0], x.shape[1], 3)
            zeros = np.zeros(zeros_shape, dtype=x.dtype)
            x = np.concatenate((x, zeros), axis=2)

    x_norm = np.empty_like(x, dtype=np.float32)
    np.divide(x, 10000.0, out=x_norm)

    # y = to_one_hot_building(y)

    return x_norm, y



def callback_preprocess_landcover_satmae(x, y):
    x_norm = sentinelNormalize(x)

    u,inv = np.unique(y,return_inverse = True)
    y = np.array([LC_MAP[x] for x in u])[inv].reshape(y.shape)

    x_norm = x_norm[16:-16, 16:-16, :]
    y = y[16:-16, 16:-16, :]
    return x_norm, y


def callback_preprocess_landcover_prithvi(x, y):
    # order S2 bands: 0-B02, 1-B03, 2-B04, 3-B08, 4-B05, 5-B06, 6-B07, 7-B8A, 8-B11, 9-B12
    # HLS bands: 0-B02, 1-B03, 2-B04, 4-B05, 5-B06, 6-B07,
    if PROCESS_PHISAT:
        x = x[:, :, (0, 1, 2, 5, 6, 7)] # throw away unused bands
    else:
        x = x[:, :, (0, 1, 2, 4, 5, 6)] # throw away unused bands

    x_norm = preprocess_image_prithvi(x)
    u, inv = np.unique(y, return_inverse=True)
    y = np.array([LC_MAP[x] for x in u])[inv].reshape(y.shape)

    return x_norm, y


def callback_preprocess_phisat2_classifier(x, y):
    assert x.shape[2] == 8, "Input x must have 8 channels for PHISAT2 classifier."
    
    x_norm = np.empty_like(x, dtype=np.float32)
    np.divide(x, 10000.0, out=x_norm)
    np.clip(x_norm, 0, 1, out=x_norm)
    np.power(x_norm, 0.5, out=x_norm)
    x_norm = (x_norm - MIN_MAJORTOM) / (MAX_MAJORTOM - MIN_MAJORTOM)
    np.clip(x_norm, 0, 1, out=x_norm)

    y = y.astype(np.float32, copy=False)
    
    return x_norm, y


def callback_postprocess_decoder(x, y):
    x = beo.channel_last_to_first(x)
    if len(y.shape) > 2:
        y = beo.channel_last_to_first(y)

    return torch.from_numpy(x), torch.from_numpy(y)


def callback_postprocess_decoder_geo(x, y):
    x = beo.channel_last_to_first(x)

    return torch.from_numpy(x), torch.from_numpy(y)


def callback_decoder(x, y):
    x, y = callback_preprocess(x, y)
    x, y = callback_postprocess_decoder(x, y)

    return x, y


def callback_decoder_landcover(x, y):
    x, y = callback_preprocess_landcover(x, y)
    x, y = callback_postprocess_decoder(x, y)

    return x, y


def callback_decoder_building_classification(x, y):
    x, y = callback_preprocess(x, y)
    x, y = callback_postprocess_decoder(x, y)

    return x, y


def callback_decoder_satmae(x, y):
    x, y = callback_preprocess_satmae(x, y)
    x, y = callback_postprocess_decoder(x, y)

    return x, y


def callback_decoder_landcover_satmae(x, y):
    x, y = callback_preprocess_landcover_satmae(x, y)
    x, y = callback_postprocess_decoder(x, y)

    return x, y

def callback_decoder_prithvi(x, y):
    x, y = callback_preprocess_prithvi(x, y)
    x, y = callback_postprocess_decoder(x, y)

    return x, y

def callback_decoder_landcover_prithvi(x, y):
    x, y = callback_preprocess_landcover_prithvi(x, y)
    x, y = callback_postprocess_decoder(x, y)

    return x, y


def callback_decoder_geo(x, y):
    x, y = callback_preprocess(x, y)
    x, y = callback_postprocess_decoder_geo(x, y)

    return x, y


def callback_decoder_phisat2_classifier(x, y):
    x, y = callback_preprocess_phisat2_classifier(x, y)
    x, y = callback_postprocess_decoder(x, y)

    return x, y


def load_data(x_train, y_train, x_val, y_val, x_test, y_test, x_inference, y_inference, device, with_augmentations=False, num_workers=0,
              batch_size=16, downstream_task=None, model_name=None, pad_to_10_bands=False):
    
    """
    Loads the data from the data folder.
    """
    global PROCESS_PHISAT
    PROCESS_PHISAT = pad_to_10_bands
    
    if model_name == 'SatMAE' or model_name == 'SatMAE_classifier':
        if downstream_task == 'lc':
            cb_decoder = callback_decoder_landcover_satmae
        else:
            cb_decoder = callback_decoder_satmae
    elif model_name == 'prithvi':
        if downstream_task == 'lc':
            cb_decoder = callback_decoder_landcover_prithvi
        else:
            cb_decoder = callback_decoder_prithvi
    elif model_name == 'phisat2_Classifier':
        cb_decoder = callback_decoder_phisat2_classifier
    else:
        if downstream_task=='lc':
            cb_decoder = callback_decoder_landcover
        elif downstream_task == 'building_classification':
            cb_decoder = callback_decoder_building_classification
        elif downstream_task == 'geo':
            cb_decoder = callback_decoder_geo
        else:
            cb_decoder = callback_decoder

    if with_augmentations:
        aug = [
                beo.AugmentationRotationXY(p=0.2, inplace=True),
                beo.AugmentationMirrorXY(p=0.2, inplace=True),
                # beo.AugmentationCutmix(p=0.2, inplace=True),
                beo.AugmentationNoiseNormal(p=0.2, inplace=True),
            ]

        if model_name == 'SatMAE':
            if downstream_task == 'lc':
                cb_preprocess = callback_preprocess_landcover_satmae
            else:
                cb_preprocess = callback_preprocess_satmae
        
        elif model_name == 'prithvi':
            if downstream_task == 'lc':
                cb_preprocess = callback_preprocess_landcover_prithvi
            else:
                cb_preprocess = callback_preprocess_prithvi
        else:
            if downstream_task=='lc':
                cb_preprocess = callback_preprocess_landcover
            else:
                cb_preprocess = callback_preprocess

        if downstream_task in ['geo', 'lc_classification', 'building_classification', 'roads_regression', 'coords']:
            cb_postprocess = callback_postprocess_decoder_geo
            aug = [
                beo.AugmentationRotation(p=0.2, inplace=True),
                beo.AugmentationMirror(p=0.2, inplace=True),
                # beo.AugmentationCutmix(p=0.2, inplace=True),
                beo.AugmentationNoiseNormal(p=0.2, inplace=True),
            ]
        else:
            cb_postprocess = callback_postprocess_decoder

        ds_train = beo.DatasetAugmentation(
            x_train, y_train,
            callback_pre_augmentation=cb_preprocess,
            callback_post_augmentation=cb_postprocess,
            augmentations=aug
        )
    else:
        ds_train = beo.Dataset(x_train, y_train, callback=cb_decoder)

    ds_test = beo.Dataset(x_test, y_test, callback=cb_decoder)
    ds_val = beo.Dataset(x_val, y_val, callback=cb_decoder)
    ds_inference = beo.Dataset(x_inference, y_inference, callback=cb_decoder)
    
    # import pdb; pdb.set_trace()

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers,
                          drop_last=False, generator=torch.Generator(device))
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers,
                         drop_last=False, generator=torch.Generator(device))
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers,
                        drop_last=False, generator=torch.Generator(device))
    dl_inference = DataLoader(ds_inference, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers,
                              drop_last=False, generator=torch.Generator(device))

    return dl_train, dl_test, dl_val, dl_inference











class TransformX:
    def __init__(self, means, stds, mins, maxs, augmentations=True, 
                 clip_values=(0., 1.), rot_prob=0.25, flip_prob=0.25, 
                 noise_prob=0.25, noise_std_range=(0.005, 0.015), input_size=None,):
        """
        Args:
            means (np.ndarray): Channel-wise means. Shape: (C,)
            stds (np.ndarray): Channel-wise stds. Shape: (C,)
            augmentations (bool): Whether to apply augmentations.
            clip_values (tuple): (min_val, max_val) for clipping.
            rot_prob (float): Probability of applying rotation.
            flip_prob (float): Probability of applying flips.
            noise_prob (float): Probability of applying noise.
            noise_std_range (tuple): (min_std, max_std) for noise std.
            input_size (int, optional): If specified and the image is larger than 
                input_size x input_size, randomly crop the image down to that size.
        """
        self.augmentations = augmentations
        self.clip_min, self.clip_max = clip_values
        self.rot_prob = rot_prob
        self.flip_prob = flip_prob
        self.noise_prob = noise_prob
        self.noise_std_low, self.noise_std_high = noise_std_range
        self.rotations = 4  # 0, 90, 180, 270 degrees
        
        self.input_size = input_size

        # Store means and stds as NumPy arrays with shape (C,1,1) for broadcasting
        self.means = means.reshape(-1, 1, 1)
        self.stds = stds.reshape(-1, 1, 1)
        
        # Normalize to [0, 1] range
        self.min_ = mins.reshape(-1, 1, 1)
        self.max_ = maxs.reshape(-1, 1, 1)

    def __call__(self, x_np, y_climate_np):
        # x_np: (C, H, W) NumPy array
        # y_climate_np: (C', H, W) NumPy array or possibly (H, W) if squeezed later
        C, H, W = x_np.shape

        # Spatial transforms
        if self.augmentations:
            
            # Random Crop
            if (self.input_size is not None) and (H >= self.input_size) and (W >= self.input_size):
                top = np.random.randint(0, H - self.input_size + 1)
                left = np.random.randint(0, W - self.input_size + 1)
                x_np = x_np[:, top:top+self.input_size, left:left+self.input_size]
                
                if y_climate_np is not None:
                    y_climate_np = y_climate_np[:, top:top+self.input_size, left:left+self.input_size]

                C, H, W = x_np.shape
            
            # Rotation
            if np.random.rand() < self.rot_prob:
                k = random.randint(0, self.rotations - 1)
                x_np = np.rot90(x_np, k, axes=(1, 2))  # rotate along H,W axes
                if y_climate_np is not None:
                    y_climate_np = np.rot90(y_climate_np, k, axes=(1, 2))

            # Horizontal flip
            if np.random.rand() < self.flip_prob:
                x_np = np.flip(x_np, axis=2)  # flip width axis
                if y_climate_np is not None:
                    y_climate_np = np.flip(y_climate_np, axis=2 if y_climate_np.ndim == 3 else 1)

            # Vertical flip
            if np.random.rand() < self.flip_prob:
                x_np = np.flip(x_np, axis=1)  # flip height axis
                if y_climate_np is not None:
                    y_climate_np = np.flip(y_climate_np, axis=1 if y_climate_np.ndim == 3 else 0)
        
        else:
            top = (H - self.input_size) // 2
            left = (W - self.input_size) // 2
            
            x_np = x_np[:, top:top+self.input_size, left:left+self.input_size]
            if y_climate_np is not None:
                y_climate_np = y_climate_np[:, top:top+self.input_size, left:left+self.input_size]

            C, H, W = x_np.shape

        # Non-spatial transforms
        # Normalize x
        x_np = x_np / 10000.0
        
        # x_np = (x_np - self.means) / self.stds

        x_np = np.clip(x_np, 0, 2.)
        # x_np = np.log1p(x_np)
        x_np = x_np ** 0.5
        x_np = (x_np - self.min_) / (self.max_ - self.min_)

        # Add noise
        if self.augmentations and np.random.rand() < self.noise_prob:
            noise_std = random.uniform(self.noise_std_low, self.noise_std_high)
            noise = np.random.normal(0, noise_std, x_np.shape)
            x_np = x_np + noise

        # Clip
        x_np = np.clip(x_np, self.clip_min, self.clip_max)

        return x_np, y_climate_np



class TransformX_LMDB:
    def __init__(self, augmentations=True, 
                 clip_values=(0., 1.), rot_prob=0.25, flip_prob=0.25, 
                 noise_prob=0.25, noise_std_range=(0.005, 0.015), input_size=None,):
        """
        Args:
            means (np.ndarray): Channel-wise means. Shape: (C,)
            stds (np.ndarray): Channel-wise stds. Shape: (C,)
            augmentations (bool): Whether to apply augmentations.
            clip_values (tuple): (min_val, max_val) for clipping.
            rot_prob (float): Probability of applying rotation.
            flip_prob (float): Probability of applying flips.
            noise_prob (float): Probability of applying noise.
            noise_std_range (tuple): (min_std, max_std) for noise std.
            input_size (int, optional): If specified and the image is larger than 
                input_size x input_size, randomly crop the image down to that size.
        """
        self.augmentations = augmentations
        self.clip_min, self.clip_max = clip_values
        self.rot_prob = rot_prob
        self.flip_prob = flip_prob
        self.noise_prob = noise_prob
        self.noise_std_low, self.noise_std_high = noise_std_range
        self.input_size = 256 if input_size > 128 else 128
        self.new_size = input_size
        self.sqrt10000 = np.sqrt(10000)  # Precompute constant for normalization
        
    def __call__(self, x_np):
        # Spatial transforms
        if self.augmentations:
            
            rand_vals = np.random.rand(3)
            
            # Random Crop
            if self.input_size > self.new_size:
                top = np.random.randint(0, self.input_size - self.new_size + 1)
                left = np.random.randint(0, self.input_size - self.new_size + 1)
                x_np = x_np[:, top:top+self.new_size, left:left+self.new_size]
                
            # Rotation
            if rand_vals[0] < self.rot_prob:
                k = np.random.randint(0, 4)  # Random rotation: 0, 90, 180, or 270 degrees
                x_np = np.rot90(x_np, k, axes=(1, 2))  # Rotate along H, W axes

            # Horizontal flip
            if rand_vals[1] < self.flip_prob:
                x_np = np.flip(x_np, axis=2)  # Flip along width axis

            # Vertical flip
            if rand_vals[2] < self.flip_prob:
                x_np = np.flip(x_np, axis=1)  # Flip along height axis
        
        else:
            if self.input_size > self.new_size:
                top = (self.input_size - self.new_size) // 2
                left = (self.input_size - self.new_size) // 2
                x_np = x_np[:, top:top+self.new_size, left:left+self.new_size]

        # Non-spatial transforms
        x_np = np.sqrt(x_np) / self.sqrt10000 # same as x_np = ( sqrt(x_np / 10000) - 0 ) / (sqrt2 - 0)

        # Add noise
        if self.augmentations and np.random.rand() < self.noise_prob:
            noise_std = np.random.uniform(self.noise_std_low, self.noise_std_high)
            noise = np.random.normal(0, noise_std, x_np.shape)
            x_np += noise

        # Clip
        x_np = np.clip(x_np, self.clip_min, self.clip_max)

        return x_np



class TransformY:
    """
    Remap climate classification from 31 Köppen classes (0..30) 
    down to 6 classes (0..5) (5 major climate zones + water/no data):

    reduced_climate_map = {
        0: water / no data
        1: tropical
        2: arid
        3: temperate
        4: cold
        5: polar
    }
    """
    def __init__(self):
        # Create a lookup array indexed by 0..30. 
        # Each position in the array tells us the new class label (0..5).
        # For example: 
        #   index = old_class (Köppen)
        #   value = new_class (reduced)
        # 
        # Mapping:
        #   0 -> 0 (water/no data)
        #   1,2,3 -> 1 (tropical)
        #   4,5,6,7 -> 2 (arid)
        #   8..16 -> 3 (temperate)
        #   17..28 -> 4 (cold)
        #   29..30 -> 5 (polar)
        #
        # NOTE: We are using np.uint8 as a convenient small-integer dtype. 
        #       You can also return int64 if preferred.

        self.map_array = np.array([
            0,  # 0 -> water/no data
            1,  # 1 -> tropical
            1,  # 2 -> tropical
            1,  # 3 -> tropical
            2,  # 4 -> arid
            2,  # 5 -> arid
            2,  # 6 -> arid
            2,  # 7 -> arid
            3,  # 8 -> temperate
            3,  # 9 -> temperate
            3,  # 10 -> temperate
            3,  # 11 -> temperate
            3,  # 12 -> temperate
            3,  # 13 -> temperate
            3,  # 14 -> temperate
            3,  # 15 -> temperate
            3,  # 16 -> temperate
            4,  # 17 -> cold
            4,  # 18 -> cold
            4,  # 19 -> cold
            4,  # 20 -> cold
            4,  # 21 -> cold
            4,  # 22 -> cold
            4,  # 23 -> cold
            4,  # 24 -> cold
            4,  # 25 -> cold
            4,  # 26 -> cold
            4,  # 27 -> cold
            4,  # 28 -> cold
            5,  # 29 -> polar
            5   # 30 -> polar
        ], dtype=np.uint8)

    def __call__(self, y_climate: np.ndarray) -> np.ndarray:
        """
        y_climate is a NumPy array (C', H, W) or (H, W) of type uint8 
        with values in [0..30]. This method maps each value to [0..5].

        Returns:
            A NumPy array of the same shape with values in [0..5].
        """
        # Ensure y_climate is within bounds 0..30
        if y_climate.min() < 0 or y_climate.max() > 30:
            raise ValueError(f"y_climate contains values outside the range [0..30] -- {y_climate.min():.3f} - {y_climate.max():.3f}")
        
        # Map to reduced classes
        mapped = self.map_array[y_climate]

        # You can return int64 if needed. For example:
        # mapped = mapped.astype(np.int64)

        return mapped





class MultiArrayDataset(Dataset):
    def __init__(
        self, 
        x_data, 
        y_data, 
        transform_x=None, 
        transform_y=None,
        apply_zoom_task=True,
        fixed_task=None,
        zoom_range=(1.0, 1.5),
        augment_drop=None,
        device='cpu',
        clip_values=(0., 1.)
    ):
        """
        Args:
            x_data (MultiArray): Input features as NumPy arrays (H, W, C).
            y_data (dict): Dictionary with keys like 'coords', 'climate', also NumPy arrays.
            transform_x (callable, optional): Optional transform on x_data (NumPy based).
            transform_y (callable, optional): Optional transform on y_data (NumPy based).
            apply_zoom_task (bool): Whether to apply the zoom-level prediction task.
            fixed_task (bool): Whether to apply a specific task.
            zoom_range (tuple): Range (min_zoom, max_zoom) for zoom factor.
            augment_drop (callable): Transformations (like RandomErasing). Should be adapted or removed if needed.
            device (str): 'cpu' or 'cuda' (not really used now, since we do everything in NumPy).
            clip_values (tuple): (min_val, max_val) for final clipping.
        """
        self.x_data = x_data
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.apply_zoom_task = apply_zoom_task
        self.fixed_task = fixed_task
        self.zoom_range = zoom_range
        self.augment_drop = augment_drop
        self.device = device
        self.clip_values = clip_values
        self.num_classes = 31

        if self.fixed_task is None or self.fixed_task == 'coords':
            self.y_coords = y_data['coords']
        if self.fixed_task is None or self.fixed_task == 'climate':
            self.y_climate = y_data['climate']

        if self.fixed_task is None:
            if not (len(self.x_data) == len(self.y_coords) == len(self.y_climate)):
                raise ValueError("x_data, y_coords, and y_climate must have the same length.")


    def __len__(self):
        return len(self.x_data)

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

        C, H, W = image_np.shape
        # Identify erased areas in the white image part
        erased_mask = (dropped[C:2*C, :, :] == 0)

        # Replace erased areas in the original with noise
        reconstructed = np.where(erased_mask, noise, dropped[:C, :, :])

        return reconstructed

    def augment_drop_fn(self, image_np):
        # Just apply zero_to_noise using NumPy
        return self.numpy_zero_to_noise(image_np)

    def numpy_resize(self, img_np, new_h, new_w):
        """
        Resize (C,H,W) image using cv2.
        img_np: (C,H,W)
        returns: (C,new_H,new_W)
        """
        C, H, W = img_np.shape
        # cv2.resize expects (H,W,C), so transpose
        img_transposed = np.transpose(img_np, (1,2,0))
        resized = cv2.resize(img_transposed, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]

        # Now back to (C, new_H, new_W)
        resized = np.transpose(resized, (2, 0, 1))
        
        return resized

    def numpy_one_hot(self, y_np):
        """
        Emulate torch one-hot logic:
        y_np: (C', H, W) with C'=1 or possibly (H, W)
        Original code used Ftorch.one_hot(y_climate_zoomed.to(int64), num_classes=self.num_classes)
        and then permute(3,1,2,0).squeeze(3).

        We'll assume C' = 1: (1, H, W). Then one-hot each pixel:
        Steps:
        - Flatten (1, H, W) -> (H * W)
        - One-hot -> (H * W, classes)
        - Reshape -> (H, W, classes)
        - Add C' dim -> (H, W, 1, classes)
        - Permute to (classes, H, W, 1)
        - Squeeze(1) -> (classes, H, W)
        """

        # Ensure y_np is int64
        y_np = y_np.astype(np.int64)
        
        # y_np: (1, H, W) -> Flatten to (H * W)
        C_prime, H, W = y_np.shape
        if C_prime != 1:
            raise ValueError("Expected y_climate to have shape (1, H, W) for one-hot encoding.")
        flat = y_np.reshape(-1)
        
        # One-hot encoding
        one_hot_flat = np.eye(self.num_classes, dtype=np.float32)[flat]  # (H * W, classes)
        
        # Reshape to (H, W, classes)
        one_hot_hw = one_hot_flat.reshape(H, W, self.num_classes)  # (H, W, classes)
        
        # Add C' dimension -> (H, W, 1, classes)
        one_hot_hw = np.expand_dims(one_hot_hw, axis=2)  # (H, W, 1, classes)
        
        # Permute to (classes, H, W, 1)
        one_hot_perm = np.transpose(one_hot_hw, (3, 0, 1, 2))  # Correct permutation
        
        # Squeeze the last dimension -> (classes, H, W)
        one_hot_final = np.squeeze(one_hot_perm, axis=-1)
        
        return one_hot_final

    def __getitem__(self, idx):
        x = np.load(self.x_data[idx])
        y_coords = np.load(self.y_coords[idx])
        y_climate = np.load(self.y_climate[idx])
        
        # Load raw data as NumPy
        # x = self.x_data[idx]           # (H, W, C) NumPy
        # Convert to (C,H,W)
        x = np.transpose(x, (2,0,1))   # (C,H,W)

        if self.fixed_task is None or self.fixed_task == 'coords':
            # y_coords = self.y_coords[idx]  # NumPy array (coords_dim,)
            y = {'coords': y_coords}
        else:
            y = {}

        if self.fixed_task is None or self.fixed_task == 'climate':
            # y_climate = self.y_climate[idx]# (H, W, C') NumPy
            y_climate = np.transpose(y_climate, (2,0,1)) # (C',H,W)
        else:
            y_climate = None

        # print(f'x.shape: {x.shape}, y_coords.shape: {y_coords.shape}, y_climate.shape: {y_climate.shape}')

        # Apply transform_x if available (NumPy based)
        if self.transform_x is not None:
            x, y_climate = self.transform_x(x, y_climate)

        # transform_y if needed
        if self.transform_y and (self.fixed_task is None or self.fixed_task == 'climate'):
            y_climate = self.transform_y(y_climate)

        # ---------------------------
        # Self-Supervised: Zoom Task
        # ---------------------------
        if self.apply_zoom_task:
            raise NotImplementedError('While the zoom task is implemented, it should not be used in this model')
            zoom_factor = random.uniform(*self.zoom_range)
            C, H, W = x.shape
            new_H, new_W = int(H * zoom_factor), int(W * zoom_factor)

            # Resize x
            zoomed = self.numpy_resize(x, new_H, new_W)

            # Resize y_climate similarly
            zoomed_climate = self.numpy_resize(y_climate, new_H, new_W)

            if zoom_factor >= 1.0:
                # Center crop to original size
                top = (new_H - H) // 2
                left = (new_W - W) // 2
                x_zoomed = zoomed[:, top:top+H, left:left+W]
                y_climate_zoomed = zoomed_climate[:, top:top+H, left:left+W]
            else:
                # If zoom_factor < 1.0, pad instead
                x_zoomed = np.zeros((C,H,W), dtype=np.float32)
                y_climate_zoomed = np.zeros_like(y_climate)
                pad_h = (H - new_H) // 2
                pad_w = (W - new_W) // 2
                x_zoomed[:, pad_h:pad_h+new_H, pad_w:pad_w+new_W] = zoomed
                y_climate_zoomed[:, pad_h:pad_h+new_H, pad_w:pad_w+new_W] = zoomed_climate

            x = x_zoomed
            
            
            # One-hot encode y_climate_zoomed
            # y_climate_one_hot = self.numpy_one_hot(y_climate_zoomed)
            # y['climate'] = y_climate_one_hot
            y['climate'] = y_climate_zoomed.squeeze(0)
            y['zoom_factor'] = np.array([zoom_factor], dtype=np.float32)
        else:
            # One-hot encode y_climate
            # y_climate_one_hot = self.numpy_one_hot(y_climate)
            # y['climate'] = y_climate_one_hot
            y['zoom_factor'] = np.array([1.0], dtype=np.float32)
            
            if self.fixed_task is None or self.fixed_task == 'climate': 
                y['climate'] = y_climate.squeeze(0)
                y['climate'] = np.ascontiguousarray(y['climate'])

        # ---------------------------
        # Self-Supervised: Reconstruction Task
        # ---------------------------
        if (self.fixed_task is None or self.fixed_task == 'reconstruction') and self.augment_drop is not None:
            x_original = x.copy()
            # Mask out areas in the image using random erasing (NumPy)
            x_masked = self.augment_drop_fn(x)
            y['reconstruction'] = x_original
            x = np.clip(x_masked, self.clip_values[0], self.clip_values[1])
        else:
            x = np.clip(x, self.clip_values[0], self.clip_values[1])
            y['reconstruction'] = None

        # Convert to torch tensors at the very end
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        y['zoom_factor'] = torch.tensor(y['zoom_factor'], dtype=torch.float32, device=self.device)

        if self.fixed_task is None or self.fixed_task == 'reconstruction':
            y['reconstruction'] = torch.tensor(y['reconstruction'], dtype=torch.float32, device=self.device)

        if self.fixed_task is None or self.fixed_task == 'coords':
            y['coords'] = torch.tensor(y['coords'], dtype=torch.float32, device=self.device)

        if self.fixed_task is None or self.fixed_task == 'climate':
            y['climate'] = torch.tensor(y['climate'], dtype=torch.int64, device=self.device)
        
        return x, y


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





class NumpyRandomErasing:
    def __init__(self, p=1.0, scale=(0.01, 0.025), ratio=(0.3, 3.3), value=0, inplace=False):
        """
        Mimics PyTorch's RandomErasing behavior in NumPy.
        
        Args:
            p (float): Probability of applying the transform.
            scale (tuple): Range of proportion of erased area against input image area.
            ratio (tuple): Range of aspect ratio of the erased area.
            value (int or float): Erasing value to fill the region.
            inplace (bool): If True, do erasing in place; otherwise, create a copy.
        """
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    def __call__(self, img):
        """
        img: NumPy array (C,H,W)
        """
        if random.random() > self.p:
            return img if self.inplace else img.copy()

        C, H, W = img.shape
        area = H * W

        # Attempt to find a valid region to erase
        for _ in range(100):
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

            h_erase = int(round(math.sqrt(target_area * aspect_ratio)))
            w_erase = int(round(math.sqrt(target_area / aspect_ratio)))

            if h_erase < H and w_erase < W:
                y0 = random.randint(0, H - h_erase)
                x0 = random.randint(0, W - w_erase)
                
                if not self.inplace:
                    img = img.copy()
                img[:, y0:y0 + h_erase, x0:x0 + w_erase] = self.value
                return img

        # If no valid region found, just return the image
        return img if self.inplace else img.copy()


class NumpyCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img




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
                         split_ratio=1.0
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



    transform_x_train = TransformX_LMDB(augmentations=with_augmentations,
                                                         input_size=input_size,
                                                         )

    transform_x_test = TransformX_LMDB(augmentations=False,
                                                        input_size=input_size,
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
    if split_ratio < 1.0:
        train_size = int(len(dataset_train) * split_ratio)
        unused_size = len(dataset_train) - train_size
        dataset_train, _ = random_split(dataset_train, [train_size, unused_size], generator=torch.Generator(device=device_dataloader))
        
        val_size = int(len(dataset_val) * split_ratio)
        unused_size = len(dataset_val) - val_size
        dataset_val, _ = random_split(dataset_val, [val_size, unused_size], generator=torch.Generator(device=device_dataloader))

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



def load_foundation_data_from_np(x_train, y_train, x_val, y_val, x_test, y_test, x_inference, y_inference, device_dataset,
                         device_dataloader, with_augmentations=True, num_workers=0, batch_size=16, input_size=None, fixed_task=None,
                         use_ddp=False, rank=0, world_size=1):
    
    augment_drop = NumpyCompose([
        NumpyRandomErasing(p=1.0, scale=(0.075, 0.100), ratio=(0.30, 3.30), value=0, inplace=False),
        NumpyRandomErasing(p=1.0, scale=(0.075, 0.100), ratio=(0.30, 3.30), value=0, inplace=True),
        NumpyRandomErasing(p=1.0, scale=(0.075, 0.100), ratio=(0.30, 3.30), value=0, inplace=True),
        NumpyRandomErasing(p=1.0, scale=(0.075, 0.100), ratio=(0.30, 3.30), value=0, inplace=True),
        NumpyRandomErasing(p=1.0, scale=(0.075, 0.100), ratio=(0.30, 3.30), value=0, inplace=True),
        NumpyRandomErasing(p=1.0, scale=(0.075, 0.100), ratio=(0.30, 3.30), value=0, inplace=True),
    ])

    # MEANS_MAJORTOM = np.array([0.1890831 , 0.17103081, 0.18347583, 0.18679002, 0.23836178, 0.1979685 , 0.23144282, 0.24777563])
    # STD_MAJORTOM = np.array([0.03045127, 0.03341764, 0.04389473, 0.0362088 , 0.05299489, 0.04338668, 0.04811415, 0.0512637])
    means_majortom = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
    std_majortom = np.array([1., 1., 1., 1., 1., 1., 1., 1.])

    min_maortom = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
    max_majortom = np.array([1.41421356, 1.41421356, 1.41421356, 1.41421356, 1.41421356, 1.41421356, 1.41421356, 1.41421356])


    transform_x_train = TransformX(means = means_majortom,
                                stds = std_majortom,
                                mins = min_maortom,
                                maxs = max_majortom,
                                augmentations=with_augmentations,
                                input_size=input_size,
                                )

    transform_x_test = TransformX(means = means_majortom,
                                stds = std_majortom,
                                mins = min_maortom,
                                maxs = max_majortom,
                                augmentations=False,
                                input_size=input_size,
                                )
    
    transform_y = None
    # transform_y = TransformY()
    
    dataset_train = MultiArrayDataset(
        x_data=x_train,
        y_data=y_train,
        transform_x=transform_x_train,
        transform_y=transform_y,
        augment_drop=augment_drop,
        device=device_dataset,
        apply_zoom_task=False,
        fixed_task=None # DID NOT IMPLEMENT THIS...
    )
    
    dataset_val = MultiArrayDataset(
        x_data=x_val,
        y_data=y_val,
        transform_x=transform_x_test,
        transform_y=transform_y,
        augment_drop=augment_drop,
        device=device_dataset,
        apply_zoom_task=False,
        fixed_task=None
    )
    
    dataset_test = MultiArrayDataset(
        x_data=x_test,
        y_data=y_test,
        transform_x=transform_x_test,
        transform_y=transform_y,
        augment_drop=augment_drop,
        device=device_dataset,
        apply_zoom_task=False,
        fixed_task=None
    )
    
    dataset_inference = MultiArrayDataset(
        x_data=x_inference,
        y_data=y_inference,
        transform_x=transform_x_test,
        transform_y=transform_y,
        augment_drop=augment_drop,
        device=device_dataset,
        apply_zoom_task=False,
        fixed_task=None
    )


    if use_ddp:
        # Typically, you'll shuffle the training dataset only
        train_sampler = DistributedSampler(
            dataset_train, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True
        )
        # For validation/test/inference, set shuffle=False or True as needed
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


    # -------------------------------------------------------
    # 4) Build DataLoaders
    #    - If using DistributedSampler, DO NOT pass shuffle=True
    # -------------------------------------------------------
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

    # dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=False, generator=torch.Generator(device=device_dataloader))
    # dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=False, generator=torch.Generator(device=device_dataloader))
    # dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, drop_last=False, generator=torch.Generator(device=device_dataloader))
    # dataloader_inference = DataLoader(dataset_inference, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, drop_last=False, generator=torch.Generator(device=device_dataloader))
    
    return dataloader_train, dataloader_val, dataloader_test, dataloader_inference