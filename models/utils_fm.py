import numpy as np

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.distributed as dist

from collections import OrderedDict

from .uniphi_foundation import phisat2net_uniphi
from .geoaware_foundation import phisat2net_geoaware

# -------------------------------------------------------------------
# MULTI-TASK LOSS FUNCTION
# -------------------------------------------------------------------

class MultiTaskLoss(nn.Module):
    def __init__(self, apply_zoom=False, fixed_task=None, device='cuda', climate_segm=False, perceptual_loss=False):
        super(MultiTaskLoss, self).__init__()
        
        # Initialize log variances as learnable parameters
        self.apply_zoom = apply_zoom
        self.fixed_task = fixed_task
        self.climate_segm = climate_segm
        self.perceptual_loss = perceptual_loss
        
        if fixed_task is None:
            self.log_sigma_recon = nn.Parameter(torch.zeros(1, device=device)) # For reconstruction
            self.log_sigma_clim = nn.Parameter(torch.zeros(1, device=device)) # For climate
            self.log_sigma_geo = nn.Parameter(torch.zeros(1, device=device)) # For geolocation
            

            # Initialize scales
            self.scale_recon = 1.
            self.scale_seg = 1.
            self.scale_geo = 1.

            if self.apply_zoom:
                raise NotImplementedError('While the zoom task is implemented, it should not be used in this model')
                self.log_sigma_zoom = nn.Parameter(torch.zeros(1)) # For zoom level
                self.scale_zoom = 1.
        
            if self.climate_segm:
                self.log_sigma_tv = nn.Parameter(torch.zeros(1, device=device))  # For TV loss
                self.scale_tv = 1.

            if self.perceptual_loss:
                self.log_sigma_perc = nn.Parameter(torch.zeros(1, device=device)) # For perceptual loss
                self.scale_perc = 1.

        # CLIMATE SEGMENTATION
        if fixed_task is None or fixed_task == "climate":
            
            class_counts = {0: 44899, 1: 2179, 2: 2544, 3: 11093,
                            4: 18101, 5: 6713, 6: 6509, 7: 8397,
                            8: 1123, 9: 886, 10: 15, 11: 3357,
                            12: 1658, 13: 10, 14: 5067, 15: 1924,
                            16: 63, 17: 193, 18: 564, 19: 1439,
                            20: 55, 21: 1328, 22: 1190, 23: 3111,
                            24: 371, 25: 2088, 26: 5867, 27: 14610,
                            28: 608, 29: 7865, 30: 397,
                            }

            reduced_class_counts = {
                0: class_counts[0], # water/no-data
                1: class_counts[1] + class_counts[2] + class_counts[3], # tropical
                2: class_counts[4] + class_counts[5] + class_counts[6] + class_counts[7], # arid
                3: sum(class_counts[i] for i in range(8, 17)), # temperate
                4: sum(class_counts[i] for i in range(17, 29)), # cold
                5: class_counts[29] + class_counts[30] # polar
            }
            
            # class_counts = reduced_class_counts

            alpha = 0.5 # choose value between 0.1 and 0.9 --> 0.5 means sqrt

            counts_array = np.array(list(class_counts.values()), dtype=np.float32)
            weights = 1.0 / np.power(counts_array, alpha)
            weights /= weights.sum()
            weights *= len(class_counts)  # Adjust so that weights sum to number of classes

            # Convert to a tensor to pass into PyTorch loss function
            self.class_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

            if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
                print("Class Weights:", self.class_weights_tensor)

            self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights_tensor, label_smoothing=0.1)
        
        if fixed_task is None or fixed_task == "coords" or fixed_task == "reconstruction":
            self.mse_loss = nn.MSELoss()
        
        if (fixed_task is None or fixed_task == "reconstruction") and self.perceptual_loss:
            self.perceptual_loss = PerceptualLoss(device=device)

    def total_variation_loss(self, logits):
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1)  # (B, 30, H, W)
        
        # Compute differences along spatial dimensions
        diff_x = probs[:, :, 1:, :] - probs[:, :, :-1, :]
        diff_y = probs[:, :, :, 1:] - probs[:, :, :, :-1]
        
        # TV loss is mean of absolute differences
        tv_loss = diff_x.abs().mean() + diff_y.abs().mean()
        return tv_loss

    def forward(self, output, labels):
        '''
        output: Dict containing model outputs
        labels: Dict containing ground truth labels

        output keys: "reconstruction", "climate", "coords", "zoom_factor"
        label keys: "reconstruction", "climate", "coords", "zoom_factor"
        '''
        if self.fixed_task is None:
            # Ensure that log_sigma values are within a reasonable range
            self.log_sigma_recon.data.clamp_(min=-2, max=1)
            self.log_sigma_clim.data.clamp_(min=-2, max=1)
            self.log_sigma_geo.data.clamp_(min=-2, max=1)
            
            if self.climate_segm:
                self.log_sigma_tv.data.clamp_(min=-2, max=1)

            if self.apply_zoom:
                self.log_sigma_zoom.data.clamp_(min=-2, max=1)
            
            if self.perceptual_loss:
                self.log_sigma_perc.data.clamp_(min=-2, max=1)

        # Reconstruction Loss (Pixel-wise and Perceptual)
        if self.fixed_task is None or self.fixed_task == "reconstruction":
            loss_recon = self.mse_loss(output["reconstruction"], labels["reconstruction"])
            if self.perceptual_loss:
                loss_perceptual = self.perceptual_loss(output["reconstruction"], labels["reconstruction"])
        
        # Climate Segmentation Loss (Cross-entropy and Total Variation)
        if self.fixed_task is None or self.fixed_task == "climate":
            loss_climate = self.ce_loss(output["climate"], labels["climate"])
            if self.climate_segm:
                loss_tv = self.total_variation_loss(output["climate"])
        
        # Geolocation Loss
        if self.fixed_task is None or self.fixed_task == "coords":
            loss_geo = self.mse_loss(output["coords"], labels["coords"])
        
        # Zoom Level Loss
        if self.apply_zoom:
            loss_zoom = self.mse_loss(output["zoom_factor"], labels["zoom_factor"])
        
        # Combine all losses with uncertainty-based weighting
        # Using the formulation: (1/(2*sigma^2)) * loss + log(sigma)
        if self.fixed_task is None:
            loss = (
                  (0.5 * torch.exp(-2 * self.log_sigma_recon)  * loss_recon  * self.scale_recon    + self.log_sigma_recon)
                + (0.5 * torch.exp(-2 * self.log_sigma_clim)   * loss_climate * self.scale_seg     + self.log_sigma_clim)
                + (0.5 * torch.exp(-2 * self.log_sigma_geo)    * loss_geo * self.scale_geo         + self.log_sigma_geo)
                )
            
            # print(f"Weighted reconstruction loss: {0.5 * torch.exp(-2 * self.log_sigma_recon)  * loss_recon  * self.scale_recon}")
            # print(f"Weighted climate loss: {0.5 * torch.exp(-2 * self.log_sigma_clim)   * loss_climate * self.scale_seg}")
            # print(f"Weighted geolocation loss: {0.5 * torch.exp(-2 * self.log_sigma_geo)    * loss_geo * self.scale_geo}")
            
            if self.climate_segm:
                loss += (0.5 * torch.exp(-2 * self.log_sigma_tv) * loss_tv * self.scale_tv + self.log_sigma_tv)
            
            if self.apply_zoom:
                loss += (0.5 * torch.exp(-2 * self.log_sigma_zoom) * loss_zoom * self.scale_zoom + self.log_sigma_zoom)
            
            if self.perceptual_loss:
                loss += (0.5 * torch.exp(-2 * self.log_sigma_perc) * loss_perceptual * self.scale_perc + self.log_sigma_perc)
                
        elif self.fixed_task == "reconstruction":
            loss = loss_recon
            if self.perceptual_loss:
                loss += loss_perceptual * 0.01
        elif self.fixed_task == "climate":
            loss = loss_climate
            if self.climate_segm:
                loss += loss_tv * 0.01
        elif self.fixed_task == "coords":
            loss = loss_geo
        else:
            raise ValueError(f"Task {self.fixed_task} is not among the tasks available")

        if self.fixed_task is None:
            log_loss = {
                'total_loss': loss.item(),
                'loss_components': {
                    'reconstruction': loss_recon.item(),
                    'climate': loss_climate.item(),
                    'geolocation': loss_geo.item(),
                    },
                'log_sigmas': {
                    'log_sigma_recon': self.log_sigma_recon.item(),
                    'log_sigma_clim': self.log_sigma_clim.item(),
                    'log_sigma_geo': self.log_sigma_geo.item(),
                    },
                'scaled_loss':{
                    'reconstruction': (0.5 * torch.exp(-2 * self.log_sigma_recon)  * loss_recon  * self.scale_recon).item(),
                    'climate': (0.5 * torch.exp(-2 * self.log_sigma_clim)  * loss_climate * self.scale_seg).item(),
                    'geolocation': (0.5 * torch.exp(-2 * self.log_sigma_geo)  * loss_geo * self.scale_geo).item(),
                    }
            }
            
            if self.climate_segm:
                log_loss['loss_components']['total_variation'] = loss_tv.item()
                log_loss['log_sigmas']['log_sigma_tv'] = self.log_sigma_tv.item()
                log_loss['scaled_loss']['total_variation'] = (0.5 * torch.exp(-2 * self.log_sigma_tv)   * loss_tv * self.scale_tv).item()

            if self.apply_zoom:
                log_loss['loss_components']['zoom_level'] = loss_zoom.item()
                log_loss['log_sigmas']['log_sigma_zoom'] = self.log_sigma_zoom.item()
                log_loss['scaled_loss']['zoom_level'] = (0.5 * torch.exp(-2 * self.log_sigma_zoom) * loss_zoom * self.scale_zoom).item()
            
            if self.perceptual_loss:
                log_loss['loss_components']['perceptual'] = loss_perceptual.item()
                log_loss['log_sigmas']['log_sigma_perc'] = self.log_sigma_perc.item()
                log_loss['scaled_loss']['perceptual'] = (0.5 * torch.exp(-2 * self.log_sigma_perc) * loss_perceptual * self.scale_perc).item()

        elif self.fixed_task == "reconstruction":
            log_loss = {
                'total_loss': loss.item(),
                'loss_components': {
                    'reconstruction': loss_recon.item()
                }
            }
            
            if self.perceptual_loss:
                log_loss['loss_components']['perceptual'] = loss_perceptual.item()
        
        elif self.fixed_task == "climate":
            log_loss = {
                'total_loss': loss.item(),
                'loss_components': {
                    'climate': loss_climate.item()
                }
            }
            
            if self.climate_segm:
                log_loss['loss_components']['total_variation'] = loss_tv.item()

        elif self.fixed_task == "coords":
            log_loss = {
                'total_loss': loss.item(),
                'loss_components': {
                    'coords': loss_geo.item()
                }
            }
        
        return loss, log_loss



class PerceptualLoss(nn.Module):
    """
    A perceptual loss class that:
    - Uses the first three channels (BGR) of 8-channel input images.
    - Converts BGR to RGB.
    - Feeds the 3-channel RGB image into a pre-trained VGG16.
    - Extracts features from specified layers.
    - Computes L1 loss between the features of the reconstructed and original images.
    """
    def __init__(self, 
                 layers=None, 
                 layer_weights=None,
                 requires_grad=False,
                 use_rgb=True,
                 device='cuda'):
        super(PerceptualLoss, self).__init__()
        
        # Default layers: Commonly used layers in VGG16
        if layers is None:
            # Layers named according to PyTorch's VGG16 structure:
            # '3': relu1_2, '8': relu2_2, '15': relu3_3, '22': relu4_3
            layers = ['3', '8', '15', '22']  # mid-level layers
        self.layers = layers
        
        # Default weights
        if layer_weights is None:
            layer_weights = [1.0 for _ in layers]
        self.layer_weights = layer_weights

        # Load pre-trained VGG16
        vgg = models.vgg16(weights=True).features
        vgg.to(device)
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = requires_grad
        self.vgg = vgg

        # Save normalization parameters (ImageNet)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        self.use_rgb = use_rgb  # Flag to indicate if using RGB channels

    def extract_features(self, x):
        """
        Pass the input through VGG and extract features at specified layers.
        """
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                features.append(x)
            if len(features) == len(self.layers):
                break
        return features

    def forward(self, x, y):
        """
        x, y: Tensors of shape (B, 8, H, W)
        """
        # Extract RGB channels (first three channels)
        x_rgb = x[:, :3, :, :]  # (B, 3, H, W)
        y_rgb = y[:, :3, :, :]  # (B, 3, H, W)

        # Convert BGR to RGB by reversing the channel order
        x_rgb = x_rgb[:, [2, 1, 0], :, :]  # (B, 3, H, W)
        y_rgb = y_rgb[:, [2, 1, 0], :, :]  # (B, 3, H, W)

        # Normalize using ImageNet statistics
        x_rgb = (x_rgb - self.mean) / self.std
        y_rgb = (y_rgb - self.mean) / self.std

        # Extract features
        x_feats = self.extract_features(x_rgb)
        y_feats = self.extract_features(y_rgb)

        # Compute weighted L1 loss over the chosen layers
        loss = 0.0
        for (x_f, y_f, w) in zip(x_feats, y_feats, self.layer_weights):
            loss += w * F.l1_loss(x_f, y_f)

        return loss




# -------------------------------------------------------------------
# GET FOUNDATION MODEL
# -------------------------------------------------------------------
def get_phisat2_model(
    model_size = 'nano',
    apply_zoom = False,
    return_model=None, # 'classifier', 'pretrain', 'pretrain_compatible', None
    fixed_task=None,
    unet_type='uniphi', # 'uniphi', 'geoaware'
    **kwargs
    ):
    
    if model_size == 'nano':            # Full mode: 298.00 MB -- Encoder: 73.690 MB
        depths = [2, 2, 8, 2]
        dims = [80, 160, 320, 640]
    
    elif model_size == 'mini':          # Full mode: 298.00 MB -- Encoder: 73.690 MB
        depths = [3, 3, 9, 3]
        dims = [92, 184, 368, 736]
        
    elif model_size == 'tiny':          # Full mode: 504.50 MB -- Encoder: 139.37 MB
        depths = [3, 3, 9, 3]
        dims = [96, 192, 384, 768]
    
    elif model_size == 'xsmall':
        depths = [3, 3, 10, 3]
        dims = [100, 200, 400, 800]

    elif model_size == 'small':         # Full mode: 635.31 MB -- Encoder: 185.63 MB
        depths = [3, 3, 12, 3]
        dims = [104, 208, 416, 832]
    
    elif model_size == 'light':         # Full mode: 1130.5 MB -- Encoder: 368.39 MB
        depths = [3, 3, 22, 3]
        dims = [124, 248, 496, 992]
    
    elif model_size == 'base':          # Full mode: 1343.1 MB -- Encoder: 448.23 MB
        depths = [3, 3, 27, 3]
        dims = [128, 256, 512, 1024]
    
    elif model_size == 'large':
        depths = [3, 3, 30, 3]
        dims = [132, 264, 528, 1056]
    
    elif model_size == 'xlarge':
        depths = [3, 3, 33, 3]
        dims = [136, 272, 544, 1088]
        
    elif model_size == 'xxlarge':
        depths = [4, 4, 28, 4]
        dims = [192, 384, 768, 1536]

    elif model_size == 'debugNano':          # Full mode: 298.00 MB -- Encoder: 73.690 MB
        depths = [2, 2]
        dims = [2, 3]

    else:
        raise ValueError(f"Invalid model size: {model_size}")
    

    if return_model == 'pretrain':
        if unet_type == 'uniphi':
            return phisat2net_uniphi(depths=depths, 
                                     dims=dims, 
                                     ov_compatiblity=False, 
                                     dropout=True, 
                                     apply_zoom=apply_zoom, 
                                     fixed_task=fixed_task,
                                     **kwargs
                                    )

        elif unet_type == 'geoaware':
            return phisat2net_geoaware(depths=depths,
                                       dims=dims,
                                       dropout=True,
                                       fixed_task=fixed_task,
                                       )
        
        
    # elif return_model == 'downstream_segmentation':
    #     return DownstreamPhiSat2(depths=depths, dims=dims, ov_compatiblity=True, task='segmentation', **kwargs)

    # elif return_model == 'downstream_classification':
    #     return DownstreamPhiSat2(depths=depths, dims=dims, ov_compatiblity=True, task='classification', **kwargs)

    elif return_model is None:
        updated_kwargs = kwargs.copy()
        updated_kwargs.update({'depths': depths, 'dims': dims})
        return updated_kwargs


