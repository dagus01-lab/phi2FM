
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
from utils.data_loader import get_zarr_dataloader, AugmentationRotationXY, AugmentationMirrorXY, AugmentationNoiseNormal

import random
from torchvision import transforms
import math
import torch.nn.functional as F


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


PHISAT_MIN = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
PHISAT_MAX = np.array([100., 100., 100., 100., 100., 100., 100., 100.])
PHISAT_MEAN = np.array([39.91732045, 37.5492021, 37.54950869, 39.21091477, 44.2665634, 39.50358262, 43.62563718, 45.28759192])
PHISAT_STD = np.array([17.06368142, 17.08672835, 20.21215486, 17.8629414, 20.11975944, 20.02886564, 19.79381833, 20.16760416])

# PHISAT_MIN = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
# PHISAT_MAX = np.array([10000., 10000., 10000., 10000., 10000., 10000., 10000., 10000.])
# PHISAT_MEAN = np.array([1884.56169544, 1701.8988641, 1818.49680678, 1856.58051233, 2364.33335501, 1961.68849886, 2294.99146283, 2457.69823862])
# PHISAT_STD = np.array([1899.72067083, 1743.80445286, 2020.09785262, 1873.41863641, 1924.71680909, 2034.2549607, 1992.56097028, 1996.09805038])

CLOUDS_MEAN = [0, 0, 0, 0, 0, 0, 0, 0]
CLOUDS_STD = [1, 1, 1, 1, 1, 1, 1, 1]


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



def pad_bands(x):
    if x.shape[2] == 8:
        if PROCESS_PHISAT == 10:
            x = np.delete(x, 3, axis=2)
            zeros_shape = (x.shape[0], x.shape[1], 3)
            zeros = np.zeros(zeros_shape, dtype=x.dtype)
            x = np.concatenate((x, zeros), axis=2)
    
        elif PROCESS_PHISAT == 13:
            # Remove PAN (assumed to be at index 3)
            x = np.delete(x, 3, axis=2)  # Now x has 7 channels: [B02, B03, B04, B08, B05, B06, B07]
            H, W, _ = x.shape

            # Create an output array of zeros with 13 channels
            out = np.zeros((H, W, 13), dtype=x.dtype)

            # Map the available channels to the proper positions:
            # Sentinel-2 L1C ordering: 0: B1 (pad), 1: B02, 2: B03, 3: B04, 4: B05, 5: B06, 6: B07, 7: B08, 8: B8A, 9: B9, 10: B10, 11: B11, 12: B12
            out[..., 1] = x[..., 0]  # B02
            out[..., 2] = x[..., 1]  # B03
            out[..., 3] = x[..., 2]  # B04
            out[..., 7] = x[..., 3]  # B08
            out[..., 4] = x[..., 4]  # B05
            out[..., 5] = x[..., 5]  # B06
            out[..., 6] = x[..., 6]  # B07
            # The remaining bands (B1, B8A, B9, B10, B11, B12) remain 0.
            x = out
        
        elif PROCESS_PHISAT == 3:
            # RGB bands are B04, B03, B02
            x = x[:, :, (2, 1, 0)]

        elif PROCESS_PHISAT == 4:
            # RGB bands are B04, B03, B02
            x = x[:, :, (3, 2, 1, 0)]
    elif x.shape[2] == 7:
        # (Assume these 7 are HLS: [B02, B03, B04, B05, B06, B07, B8A])
        if PROCESS_PHISAT == 10:
            # We want 10 bands. Simply pad with three zero‐channels.
            # e.g. [B02, B03, B04, B05, B06, B07, B8A] + [0,0,0] → 10
            zeros_shape = (x.shape[0], x.shape[1], 3)
            zeros = np.zeros(zeros_shape, dtype=x.dtype)
            x = np.concatenate((x, zeros), axis=2)

        elif PROCESS_PHISAT == 13:
            # Create a 13‐band “Sentinel-2 L1C” output, mapping these 7 channels.
            H, W, _ = x.shape
            out = np.zeros((H, W, 13), dtype=x.dtype)
            # Suppose our 7 bands are HLS in order: 
            #    x[...,0]=B02, x[...,1]=B03, x[...,2]=B04, x[...,3]=B05, x[...,4]=B06, x[...,5]=B07, x[...,6]=B8A
            out[..., 1] = x[..., 0]  # B02
            out[..., 2] = x[..., 1]  # B03
            out[..., 3] = x[..., 2]  # B04
            # We don’t have B08 in HLS, so out[...,7] stays 0
            out[..., 4] = x[..., 3]  # B05
            out[..., 5] = x[..., 4]  # B06
            out[..., 6] = x[..., 5]  # B07
            out[..., 8] = x[..., 6]  # B8A
            # (Bands B1, B9, B10, B11, B12 remain zero)
            x = out

        elif PROCESS_PHISAT == 3:
            # If the model only needs RGB, assume HLS indexing:
            # HLS:  0→B02, 1→B03, 2→B04  ⇒ RGB = [B04, B03, B02]
            x = x[:, :, (2, 1, 0)]

        elif PROCESS_PHISAT == 4:
            # If model needs RGB+NIR, but HLS has no B08 as separate channel,
            # you could treat B8A (x[...,6]) as “approximate NIR”:
            #   [B04, B03, B02, B8A]
            x = x[:, :, (2, 1, 0, 6)]
        elif PROCESS_PHISAT == 8:
            H, W, _ = x.shape
            zeros = np.zeros((H, W, 1), dtype=x.dtype)
            x = np.concatenate((x, zeros), axis=2)

        # If PROCESS_PHISAT is something else, you may want to raise an error or
        # simply leave ‘x’ as is. For example:
        else:
            raise ValueError(f"Unsupported PROCESS_PHISAT={PROCESS_PHISAT} when x has 7 channels")

    return x


def sentinelNormalize(x):
    if PROCESS_PHISAT is not None:
        x = pad_bands(x)
            
        min_value = MEANS_SATMAE_PHI2 - 2 * STDS_SATMAE_PHI2
        max_value = MEANS_SATMAE_PHI2 + 2 * STDS_SATMAE_PHI2
    
    else:
        min_value = MEANS_SATMAE - 2 * STDS_SATMAE
        max_value = MEANS_SATMAE + 2 * STDS_SATMAE

    img = (x - min_value) / (max_value - min_value + 1e-8) * 255.0
    img = np.clip(img, 0, 255).astype(np.float32)
    return img

def preprocess_image_prithvi(image):
    if PROCESS_PHISAT is not None:
        image = pad_bands(image)
        
        normalized = image.copy()
        #normalized = ((image - MEANS_PRITHVI_PHI2) / STDS_PRITHVI_PHI2)

    else:
        # normalize image
        normalized = image.copy()
        #normalized = ((image - MEANS_PRITHVI) / STDS_PRITHVI)
    normalized = normalized.astype(np.float32, copy=False)

    # normalized = torch.from_numpy(normalized.reshape(1, normalized.shape[0], 1, *normalized.shape[-2:])).to(torch.float32)
    return normalized

def callback_preprocess(x, y):
    x = pad_bands(x)
    #x_norm = np.empty_like(x, dtype=np.float32)
    #np.divide(x, 10000.0, out=x_norm)

    y = y.astype(np.float32, copy=False)

    return x, y#x_norm, y


def callback_preprocess_satmae(x, y):
    x = pad_bands(x)
    x_norm = sentinelNormalize(x)
    y = y.astype(np.float32, copy=False)

    x_norm = x_norm[16:-16, 16:-16, :]
    if len(y.shape) > 2:
        y = y[16:-16, 16:-16, :]
    return x_norm, y


def callback_preprocess_prithvi(x, y):
    # order S2 bands: 0-B02, 1-B03, 2-B04, 3-B08, 4-B05, 5-B06, 6-B07, 7-B8A, 8-B11, 9-B12
    # HLS bands: 0-B02, 1-B03, 2-B04, 4-B05, 5-B06, 6-B07,
    #print(x.shape)
    if PROCESS_PHISAT == 10:
        x = x[:, :, (0, 1, 2, 5, 6, 7)] 
    else:
        x = x[:, :, (0, 1, 2, 4, 5, 6)] 
    x_norm = preprocess_image_prithvi(x)
    y = y.astype(np.float32, copy=False)

    return x_norm, y


def callback_preprocess_landcover(x, y):
    x = pad_bands(x)
    x_norm = np.empty_like(x, dtype=np.float32)
    np.divide(x, 10000.0, out=x_norm)

    u, inv = np.unique(y, return_inverse=True)
    y = np.array([LC_MAP[val] for val in u])[inv].reshape(y.shape)
    
    return x_norm, y


def callback_preprocess_building_classification(x, y):
    x = pad_bands(x)
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
    if PROCESS_PHISAT == 10:
        x = x[:, :, (0, 1, 2, 5, 6, 7)] # throw away unused bands
    else:
        x = x[:, :, (0, 1, 2, 4, 5, 6)] # throw away unused bands

    x_norm = preprocess_image_prithvi(x)
    u, inv = np.unique(y, return_inverse=True)
    y = np.array([LC_MAP[x] for x in u])[inv].reshape(y.shape)

    return x_norm, y


def callback_preprocess_phisatnet(x, y):
    assert x.shape[2] == 8, "Input x must have 8 channels for phisatnet model."
    
    #x = np.sqrt(x)
    #x = np.clip(x, PHISAT_MIN, PHISAT_MAX)
    #x = (x - PHISAT_MEAN) / PHISAT_STD
    
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)
    
    return x, y


def callback_preprocess_phisatnet_lc(x, y):
    assert x.shape[2] == 8, "Input x must have 8 channels for phisatnet model."
    
    x = np.sqrt(x)
    x = np.clip(x, PHISAT_MIN, PHISAT_MAX)
    x = (x - PHISAT_MEAN) / PHISAT_STD
    
    x = x.astype(np.float32, copy=False)
    
    u, inv = np.unique(y, return_inverse=True)
    y = np.array([LC_MAP[val] for val in u])[inv].reshape(y.shape)
    
    return x, y


def callback_postprocess_decoder(x, y):
    x = beo.channel_last_to_first(x)
    if y.ndim > 2:
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
    #x, y = callback_postprocess_decoder(x, y)
    x = beo.channel_last_to_first(x)
    if y.ndim > 2:
        y = beo.channel_last_to_first(y)
    #x, y = callback_postprocess_decoder(x, y)
    
    return torch.from_numpy(x), torch.from_numpy(np.array(y))
    #return x, y


def callback_decoder_landcover_satmae(x, y):
    x, y = callback_preprocess_landcover_satmae(x, y)
    x, y = callback_postprocess_decoder(x, y)

    return x, y

def callback_decoder_prithvi(x, y):
    x, y = callback_preprocess_prithvi(x, y)
    #x, y = callback_postprocess_decoder(x, y)
    x = beo.channel_last_to_first(x)
    if y.ndim > 2:
        y = beo.channel_last_to_first(y)
    #x, y = callback_postprocess_decoder(x, y)
    
    return torch.from_numpy(x), torch.from_numpy(np.array(y))
    #return x, y

def callback_decoder_landcover_prithvi(x, y):
    x, y = callback_preprocess_landcover_prithvi(x, y)
    x, y = callback_postprocess_decoder(x, y)

    return x, y


def callback_decoder_geo(x, y):
    x, y = callback_preprocess(x, y)
    x, y = callback_postprocess_decoder_geo(x, y)

    return x, y


def callback_decoder_phisatnet(x, y):
    x, y = callback_preprocess_phisatnet(x, y)
    x = beo.channel_last_to_first(x)
    if y.ndim > 2:
        y = beo.channel_last_to_first(y)
    #x, y = callback_postprocess_decoder(x, y)
    
    return torch.from_numpy(x), torch.from_numpy(np.array(y))

def callback_decoder_phisatnet_lc(x, y):
    x, y = callback_preprocess_phisatnet_lc(x, y)
    x, y = callback_postprocess_decoder(x, y)
    return x, y
    
def callback_decoder_phisatnet_clouds(x, y):
    x, y = callback_preprocess(x, y)
    x = beo.channel_last_to_first(x)
    if y.ndim > 2:
        y = beo.channel_last_to_first(y)
    x = minmax_normalize_image(x)#(x-CLOUDS_MEAN)/CLOUDS_STD #normalize_image_burned_area(x)
    return torch.from_numpy(x), torch.from_numpy(y)
    
def callback_decoder_phisatnet_burned_area(x, y):
    x, y = callback_preprocess(x, y)
    x = beo.channel_last_to_first(x)
    if y.ndim > 2:
        y = beo.channel_last_to_first(y)
    x = minmax_normalize_image(x) #normalize_image_burned_area(x)
    return torch.from_numpy(x), torch.from_numpy(y)
    
def callback_decoder_phisatnet_fire(x, y):
    x, y = callback_preprocess(x, y) 
    x = beo.channel_last_to_first(x)
    x = minmax_normalize_image(x) #normalize_image_fire(x)
    return torch.from_numpy(x), torch.from_numpy(np.array(y))
    
def minmax_normalize_image(x):
    # Normalize each channel to [0, 1]
    print(x.shape)
    for c in range(x.shape[0]):
        min_val = np.min(x[c])
        max_val = np.max(x[c])
        if max_val > min_val:  # Avoid division by zero
            x[c] = (x[c] - min_val) / (max_val - min_val)
    return x
    
def normalize_image_burned_area(x):
    means = [0.5692603492540789, 0.5233146455770651, 0.49774728208504626, 0.5614061973077787, 0.5094977101466148, 0.5503450336828751, 0.5719299002762076, 0, 0, 0]
    std = [0.24279108867296925, 0.25451220952717407, 0.277410560398893, 0.28924007207410934, 0.2766535835665443, 0.2841112679453489, 0.28949325669342035, 1, 1, 1]
    for c in range(x.shape[0]):
        x[c] = (x[c]-means[c]) / std[c]
    return x
def normalize_image_fire(x):
    #print(x.shape)
    means = [0.13257149824361583, 0.11341501907740992, 0.10651690894124867, 0.11917635482784486, 0.14930889600288982, 0.11450899398166578, 0.1397788301338759, 0.15386646124067438, 0, 0, 0]
    std =  [0.059953769519909654, 0.05968709520228296, 0.06689720179336182, 0.062324357124935545, 0.08827480561589907, 0.06881834767281576, 0.07911373795071863, 0.08765243590683403, 1, 1, 1]
    for c in range(x.shape[0]):
        x[c] = (x[c]-means[c]) / std[c]
    return x

def callback_decoder_burned_area(x, y):
    x, y = callback_preprocess(x, y) 
    x = beo.channel_last_to_first(x)
    if y.ndim > 2:
        y = beo.channel_last_to_first(y)
    x = minmax_normalize_image(x) #normalize_image_burned_area(x)
    return torch.from_numpy(x), torch.from_numpy(y)

def callback_decoder_clouds(x, y):
    x, y = callback_preprocess(x, y)
    x = beo.channel_last_to_first(x)
    if y.ndim > 2:
        y = beo.channel_last_to_first(y)
    x = minmax_normalize_image(x)#(x - CLOUDS_MEAN ) /CLOUDS_STD
    return torch.from_numpy(x), torch.from_numpy(y)

def callback_decoder_fire(x, y):
    x, y = callback_preprocess(x, y)
    x = beo.channel_last_to_first(x)
    x = minmax_normalize_image(x) #normalize_image_fire(x)
    return torch.from_numpy(x), torch.from_numpy(np.array(y))

def callback_decoder_satmae_fire(x, y):
    x = pad_bands(x)
    x_norm = minmax_normalize_image(x) #sentinelNormalize(x)
    y = y.astype(np.float32, copy=False)
    x_norm = beo.channel_last_to_first(x_norm)
    x_norm = x_norm[:, 80:-80, 80:-80]
    #x_norm = x_norm[16:-16, 16:-16, :]
    if len(y.shape) > 2:
        y = y[:, 80:-80, 80:-80]
    return torch.from_numpy(x_norm), torch.from_numpy(np.array(y)) #x_norm, y
def callback_decoder_satmae_burned_area(x, y):
    x = pad_bands(x)
    x_norm = sentinelNormalize(x)
    y = y.astype(np.float32, copy=False)
    x_norm = beo.channel_last_to_first(x_norm)
    x_norm = x_norm[:, 80:-80, 80:-80]
    #x_norm = x_norm[16:-16, 16:-16, :]
    if y.ndim > 2:
        y = beo.channel_last_to_first(y)
    if len(y.shape) > 2:
        y = y[:, 80:-80, 80:-80]
    return torch.from_numpy(x_norm), torch.from_numpy(np.array(y)) #x_norm, y

def callback_decoder_satmae_clouds(x, y):
    x = pad_bands(x)
    x_norm = minmax_normalize_image(x)#(x - CLOUDS_MEAN) / CLOUDS_STD
    y = y.astype(np.float32, copy=False)
    x_norm = beo.channel_last_to_first(x_norm)
    x_norm = x_norm[:, 80:-80, 80:-80]
    #x_norm = x_norm[16:-16, 16:-16, :]
    if y.ndim > 2:
        y = beo.channel_last_to_first(y)
    #if len(y.shape) > 2:
    #    y = y[16:-16, 16:-16, :]
    return torch.from_numpy(x_norm), torch.from_numpy(np.array(y))

def callback_decoder_satmae_worldfloods(x, y):
    x = pad_bands(x)
    #x_norm = sentinelNormalize(x)
    y = y.astype(np.float32, copy=False)
    x = beo.channel_last_to_first(x)
    x = x[:, 80:-80, 80:-80]
    #x_norm = x_norm[16:-16, 16:-16, :]
    if y.ndim > 2:
        y = beo.channel_last_to_first(y)
    #if len(y.shape) > 2:
    #    y = y[16:-16, 16:-16, :]
    return torch.from_numpy(x), torch.from_numpy(np.array(y)) #x_norm, y

def callback_preprocess_fire_satmae(x, y):
    #x_norm = sentinelNormalize(x)
    return x, y
def callback_preprocess_burned_area_satmae(x, y):
    #x_norm = sentinelNormalize(x)
    return x, y
def callback_preprocess_fire_prithvi(x, y):
    return x, y
def callback_preprocess_prithvi_burned_area(x, y):
    return x, y
def callback_decoder_prithvi_fire(x, y):
    x = pad_bands(x)
    x, y =  callback_decoder_prithvi(x, y)
    x = x[:, 10:-10, 10:-10]
    return x, y
def callback_decoder_prithvi_clouds(x, y):
    x = pad_bands(x)
    x, y =  callback_decoder_prithvi(x, y)
    x = x[:, 16:-16, 16:-16]
    x_norm = minmax_normalize_image(x)#(x-CLOUDS_MEAN)/CLOUDS_STD
    y = y[:, 16:-16, 16:-16]
    return x_norm, y
def callback_decoder_prithvi_worldfloods(x, y):
    x = pad_bands(x)
    x, y =  callback_decoder_prithvi(x, y)
    x = x[:, 16:-16, 16:-16]
    y = y[:, 16:-16, 16:-16]
    return x, y
def callback_decoder_prithvi_burned_area(x, y):
    x = pad_bands(x)
    x, y =  callback_decoder_prithvi(x, y)
    x = x[:, 16:-16, 16:-16]
    y = y[:, 16:-16, 16:-16]
    return x, y

def callback_preprocess_phisatnet_burned_area(x, y):
    return x, y

def load_data(dataset_path, device, with_augmentations=False, num_workers=0, batch_size=16, downstream_task=None, model_name=None, pad_bands=False, 
             crop_images: bool = False, n: int = None, regions: list= None, y: str='lc', data_selection: str = 'create', name: str = None, split_percentage: list=None, by_region: bool=False, num_classes: int = 4, weights_dir: str = None):
    
    """
    Loads the data from the data folder.
    """
    global PROCESS_PHISAT
    PROCESS_PHISAT = pad_bands
    
    
    if model_name == 'SatMAE' or model_name == 'SatMAE_classifier':
        if downstream_task == 'lc':
            cb_decoder = callback_decoder_landcover_satmae
        elif downstream_task == 'fire':
            cb_decoder = callback_decoder_satmae_fire
        elif downstream_task == 'burned_area' or downstream_task == 'worldfloods':
            cb_decoder = callback_decoder_satmae_burned_area
        elif downstream_task == 'clouds':
            cb_decoder = callback_decoder_satmae_clouds
        else:
            cb_decoder = callback_decoder_satmae
    elif model_name == 'prithvi':
        if downstream_task == 'lc':
            cb_decoder = callback_decoder_landcover_prithvi
        elif downstream_task == 'fire':
            cb_decoder = callback_decoder_prithvi_fire
        elif downstream_task == 'burned_area' or downstream_task == 'worldfloods':
            cb_decoder = callback_decoder_prithvi_burned_area
        elif downstream_task == "clouds":
            cb_decoder = callback_decoder_prithvi_clouds
        else:
            cb_decoder = callback_decoder_prithvi
    elif model_name == 'phisatnet' or model_name == 'phisatnet_classifier':
        if downstream_task == 'lc':
            cb_decoder = callback_decoder_phisatnet_lc
        elif downstream_task == 'fire':
            cb_decoder = callback_decoder_phisatnet_fire
        elif downstream_task == 'burned_area' or downstream_task == 'worldfloods':
            cb_decoder = callback_decoder_phisatnet_burned_area
        elif downstream_task == "clouds":
            cb_decoder = callback_decoder_phisatnet_clouds
        else:
            cb_decoder = callback_decoder_phisatnet
    else:
        if downstream_task=='lc':
            cb_decoder = callback_decoder_landcover
        elif downstream_task == 'building_classification':
            cb_decoder = callback_decoder_building_classification
        elif downstream_task == 'geo':
            cb_decoder = callback_decoder_geo
        elif downstream_task == 'burned_area' or downstream_task == 'worldfloods':
            cb_decoder = callback_decoder_burned_area
        elif downstream_task == "clouds":
            cb_decoder = callback_decoder_clouds
        elif downstream_task == 'fire':
            cb_decoder = callback_decoder_fire
        else:
            cb_decoder = callback_decoder
    if with_augmentations:
        aug = [
                AugmentationRotationXY(p=0.2, inplace=True),
                AugmentationMirrorXY(p=0.2, inplace=True),
                # beo.AugmentationCutmix(p=0.2, inplace=True),
                AugmentationNoiseNormal(p=0.2, inplace=True),
            ]

        if model_name == 'SatMAE':
            #print("SATMAE preprocessing")
            if downstream_task == 'lc':
                cb_preprocess = callback_preprocess_landcover_satmae
            elif downstream_task == 'fire':
                cb_preprocess = callback_preprocess_fire_satmae
            elif downstream_task == 'burned_area' or downstream_task == "worldfloods":
                cb_preprocess = callback_preprocess_burned_area_satmae
            else:
                cb_preprocess = callback_preprocess_satmae
        
        elif model_name == 'prithvi':
            #print("prithvi preprocessing")
            if downstream_task == 'lc':
                cb_preprocess = callback_preprocess_landcover_prithvi
            elif downstream_task == 'fire':
                cb_preprocess = callback_preprocess_fire_prithvi
            elif downstream_task == 'burned_area' or downstream_task == "worldfloods":
                cb_preprocess = callback_preprocess_prithvi_burned_area
            else:
                cb_preprocess = callback_preprocess_prithvi
        elif model_name == 'phisatnet' or model_name == 'phisatnet_classifier':
            #print("phisatnet preprocessing")
            if downstream_task == 'lc':
                cb_preprocess = callback_preprocess_phisatnet_lc
            elif downstream_task == "burned_area" or downstream_task == "worldfloods":
                cb_preprocess = callback_preprocess_phisatnet_burned_area
            else:
                cb_preprocess = callback_preprocess_phisatnet
        else:
            #print("other preprocessing")
            if downstream_task=='lc':
                cb_preprocess = callback_preprocess_landcover
            else:
                cb_preprocess = callback_preprocess

        if downstream_task in ['geo', 'lc_classification', 'building_classification', 'roads_regression', 'coords', ]:
            cb_postprocess = callback_postprocess_decoder_geo
            aug = [
                AugmentationRotation(p=0, inplace=True),
                AugmentationMirror(p=0.2, inplace=True),
                # beo.AugmentationCutmix(p=0.2, inplace=True),
                AugmentationNoiseNormal(p=0.2, inplace=True),
            ]
        else:
            cb_postprocess = callback_postprocess_decoder

        callback_pre_augmentation_training = cb_preprocess
        callback_post_augmentation_training = cb_decoder  #cb_postprocess
        augmentations_training = aug
    else:
        callback_pre_augmentation_training = None
        callback_post_augmentation_training = cb_decoder
        augmentations_training = None
        
    callback_pre_augmentation_test = None
    callback_post_augmentation_test = cb_decoder
    augmentations_test = None

    callback_pre_augmentation_val = None
    callback_post_augmentation_val = cb_decoder
    augmentations_val = None

    callback_pre_augmentation_inference = None
    callback_post_augmentation_inference = cb_decoder
    augmentations_inference = None
    
    if downstream_task == "clouds":
        weight, pos_weight, dl_train, dl_val, dl_test = get_zarr_dataloader(
            zarr_path=dataset_path,                     # Path to the Zarr archive
            dataset_set="trainval",                 # Dataset subset to use
            batch_size=16,                           # Number of samples per batch
            shuffle=True,                            # Enable shuffling (useful for training)
            num_workers=4,                           # Number of parallel workers for loading
            #transform=NormalizeChannels(min_max=True),  # Normalize input channels to [0, 1]
            metadata_keys=["sensor", "timestamp", "geolocation", "crs"],   # Include auxiliary metadata fields
            verbose = False,
            split = [.8, .02, .18], 
            save = True,
            split_names = ["train", "validation", "test"],
            callback_pre_augmentation = [callback_pre_augmentation_training, callback_pre_augmentation_val, callback_pre_augmentation_test],
            callback_post_augmentation = [callback_post_augmentation_training, callback_post_augmentation_val, callback_post_augmentation_test],
            augmentations = [augmentations_training, augmentations_val, augmentations_test], 
            crop_images= crop_images, 
            generator= torch.Generator(device), 
            pin_memory=True, 
            drop_last=False, 
            num_classes=num_classes, 
            n_shot=[n, 0, 0], 
            weights_dir=weights_dir
        )

        dl_inference = dl_test
    
    else:
        # import pdb; pdb.set_trace()
        weight, pos_weight, dl_train, dl_val = get_zarr_dataloader(
            zarr_path=dataset_path,                     # Path to the Zarr archive
            dataset_set="trainval",                 # Dataset subset to use
            batch_size=16,                           # Number of samples per batch
            shuffle=True,                            # Enable shuffling (useful for training)
            num_workers=4,                           # Number of parallel workers for loading
            #transform=NormalizeChannels(min_max=True),  # Normalize input channels to [0, 1]
            metadata_keys=["sensor", "timestamp", "geolocation", "crs"],   # Include auxiliary metadata fields
            verbose = False,
            split = [.9, .1], 
            split_names = ["train", "validation"],
            callback_pre_augmentation = [callback_pre_augmentation_training, callback_pre_augmentation_val],
            callback_post_augmentation = [callback_post_augmentation_training, callback_post_augmentation_val],
            augmentations = [augmentations_training, augmentations_val], 
            crop_images= crop_images, 
            generator= torch.Generator(device), 
            pin_memory=True, 
            drop_last=False, 
            num_classes=num_classes, 
            n_shot=[n, 0], 
            weights_dir=weights_dir
        )

        _, _, dl_test = get_zarr_dataloader(
            zarr_path=dataset_path,                     # Path to the Zarr archive
            dataset_set="test",                 # Dataset subset to use
            batch_size=16,                           # Number of samples per batch
            shuffle=True,                            # Enable shuffling (useful for training)
            num_workers=4,                           # Number of parallel workers for loading
            #transform=NormalizeChannels(min_max=True),  # Normalize input channels to [0, 1]
            metadata_keys=["sensor", "timestamp", "geolocation", "crs"],   # Include auxiliary metadata fields
            verbose = False,
            split = None, 
            split_names = ["test"],
            callback_pre_augmentation = callback_pre_augmentation_test,
            callback_post_augmentation = callback_post_augmentation_test,
            augmentations = augmentations_test,
            crop_images= crop_images, 
            generator= torch.Generator(device), 
            pin_memory=True, 
            drop_last=False, 
            num_classes=num_classes, 
            n_shot=0, 
            weights_dir=None
        ) 
        dl_inference = dl_test

    return weight, pos_weight, dl_train, dl_test, dl_val, dl_inference



