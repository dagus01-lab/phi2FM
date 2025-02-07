import torch
import torch.nn as nn
from .blocks import CNNBlock, ScaleSkip2D, ChannelGLU, make_bilinear_upsample
from torchvision import models
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from tabulate import tabulate
import torch.distributed as dist

# FX Graph Mode QAT
# from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
# from torch.ao.quantization import get_default_qat_qconfig

# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from model_CoreCNN import CoreCNNBlock, CoreEncoderBlock, CoreDecoderBlock


# -------------------------------------------------------------------
# FoundationEncoder without flattening
# -------------------------------------------------------------------
class FoundationEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim=3,
        depths=None,
        dims=None,
        img_size=64,
        activation=nn.ReLU6(),
        norm='batch',
        encoder_drop_probs=None,
    ):
        """
        A U-Net style encoder that:
         1) progressively downsamples
         2) returns final spatial features (B, dims[-1], H_enc, W_enc)
         3) returns skip connections
        """
        super().__init__()

        self.depths = depths
        self.dims = dims
        self.input_dim = input_dim
        self.img_size = img_size
        self.activation = activation

        # Compute how many steps (downsampling stages) we have
        self.steps = len(depths)
        self.sizes = [img_size // (2**i) for i in range(self.steps)]

        if encoder_drop_probs is None:
            encoder_drop_probs = [0.0]*self.steps

        # -------------------------------------------------------------
        # Downsample modules
        # -------------------------------------------------------------
        self.downsample = nn.ModuleList()
        for i in range(self.steps - 1):
            # e.g. dims[i] -> dims[i+1], with a stride-2 or maxpool down
            self.downsample.append(
                nn.Sequential(
                    nn.Conv2d(self.dims[i], self.dims[i + 1], 1, padding=0),
                    nn.MaxPool2d(2, stride=2),
                )
            )

        # -------------------------------------------------------------
        # Blocks + Skip-scale modules
        # -------------------------------------------------------------
        self.block_scalers = nn.ModuleList([ScaleSkip2D(self.dims[i]) for i in range(self.steps)])
        self.blocks_down = nn.ModuleList()

        for i in range(self.steps):
            stage_blocks = nn.ModuleList()
            # We'll use the same drop probs for each block at stage i
            drop_prob_main = encoder_drop_probs[i]

            for _ in range(depths[i]):
                stage_blocks.append(
                    CNNBlock(
                        channels_in=self.dims[i],
                        channels_out=self.dims[i],
                        chw=None if norm=='batch' else [self.dims[i], self.sizes[i], self.sizes[i]],
                        activation=self.activation,
                        drop_prob_main=drop_prob_main
                    )
                )
            self.blocks_down.append(stage_blocks)

    def forward(self, x):
        """
        Returns:
          final_feats: spatial feature map at the bottom of U-Net (B, dims[-1], H_enc, W_enc)
          skips: list of skip connection feature maps
        """
        skips = []

        for i in range(self.steps):
            
            # (1) Local blocks
            pre_block = x
            
            for j in range(self.depths[i]):
                x = self.blocks_down[i][j](x)

            # If we stacked more than one block on this stage, do the skip-scaler
            if self.depths[i] > 1:
                x = self.block_scalers[i](x, pre_block)

            # (2) Collect skip at this stage
            skips.append(x)

            # (3) Downsample, except for the last iteration i=steps-1
            if i < self.steps - 1:
                x = self.downsample[i](x)

        # x is now the bottom-most representation: (B, dims[-1], H_enc, W_enc)
        return x, skips

# -------------------------------------------------------------------
# FoundationDecoder
# -------------------------------------------------------------------
class FoundationDecoder(nn.Module):
    def __init__(
        self,
        *,
        depths=None,
        dims=None,
        img_size=64,
        dropout=None,
        activation=nn.ReLU6(),
        norm='batch',
        ov_compatiblity=False,
        decoder_drop_probs=None
    ):
        """
        A U-Net style decoder that:
         1) up-samples the bottom feature map
         2) merges with skip connections
         3) returns final high-res feature map (B, dims[0], H, W)
        """
        super().__init__()
        self.depths = depths
        self.dims = dims
        self.img_size = img_size
        self.activation = activation
        

        self.steps = len(depths)
        # sizes in reverse for the decoder
        self.sizes = [img_size // (2**(self.steps - i - 1)) for i in range(self.steps)]

        # Handle dropout (optional)
        if dropout is None:
            dropout = [None] * self.steps
        elif isinstance(dropout, (int, float)):
            dropout = [dropout] * self.steps
        self.dropout = dropout

        if decoder_drop_probs is None:
            decoder_drop_probs = [0.0]*len(depths)

        # -------------------------------------------------------------
        # Skip & block scalers, blocks
        # -------------------------------------------------------------
        self.skip_scalers = nn.ModuleList()
        self.block_scalers = nn.ModuleList([ScaleSkip2D(self.dims[i]) for i in range(self.steps)])
        self.blocks_up = nn.ModuleList()

        for i in range(self.steps):
            # self.skip_scalers.append(ScaleSkip2D(self.dims[i], drop_y=self.dropout[i], signal_to_noise=(0.1, 0.9)))
            self.skip_scalers.append(ScaleSkip2D(self.dims[i]))

            stage_blocks = nn.ModuleList()
            # We'll use the same drop probs for each block at stage i
            drop_prob_main = decoder_drop_probs[i]
            
            for _ in range(depths[i]):
                stage_blocks.append(
                    CNNBlock(
                        channels_in=self.dims[i],
                        channels_out=self.dims[i],
                        chw=None if norm=='batch' else [self.dims[i], self.sizes[i], self.sizes[i]],
                        activation=self.activation,
                        drop_prob_main=drop_prob_main
                    )
                )
            self.blocks_up.append(stage_blocks)


        # -------------------------------------------------------------
        # Upsamplers
        # -------------------------------------------------------------
        self.upsamplers = nn.ModuleList()
        for i in range(self.steps - 1):
            self.upsamplers.append(
                nn.Sequential(
                    make_bilinear_upsample(self.dims[i]) if ov_compatiblity else nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.Conv2d(self.dims[i], self.dims[i + 1], 3, padding=1, bias=False, padding_mode='replicate'),
                    nn.LayerNorm([self.dims[i + 1], self.sizes[i + 1], self.sizes[i + 1]]),
                    self.activation,
                )
            )

        self.prehead_norm = nn.LayerNorm([self.dims[-1], self.sizes[-1], self.sizes[-1]])

    def forward(self, x, skips):
        """
        x: (B, dims[-1], H_enc, W_enc) bottom_feats from the encoder
        skips: list of skip features from each encoder stage
        returns: (B, dims[0], H, W)
        """

        for i in range(self.steps):
            # (1) Upsample, except for the first iteration i=0:
            if i > 0:
                x = self.upsamplers[i-1](x)

            # (2) Merge skip from the corresponding stage
            skip_x = skips[-(i + 1)]
            x = self.skip_scalers[i](x, skip_x)

            # (3) Local blocks
            pre_block = x
            for block in self.blocks_up[i]:
                x = block(x)
            if self.depths[i] > 1:
                x = self.block_scalers[i](x, pre_block)

        x = self.prehead_norm(x)
        return x


# -------------------------------------------------------------------
# The combined multi-task model
# -------------------------------------------------------------------
class FoundationModel4Task(nn.Module):
    def __init__(
        self,
        *,
        input_dim=3,
        output_dim=None,
        depths=None,
        dims=None,
        img_size=128,
        dropout=True,
        activation=nn.GELU(),
        ov_compatiblity=False,
        apply_zoom=False,
        fixed_task=None
    ):
        """
        A U-Net-like model with extra heads for:
          - climate zone segmentation
          - geolocation
          - zoom level
          - reconstruction
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim if output_dim is None else output_dim
        self.depths = depths
        self.dims = dims
        self.img_size = img_size
        self.apply_zoom = apply_zoom
        self.fixed_task = fixed_task

        self.dropout = dropout
        
        if dropout:
            self.encoder_drop_probs=[0.0, 0.05, 0.1, 0.15]
            self.decoder_drop_probs=[0.1, 0.15, 0.15, 0.2]
            self.decoder_drop = 0.1
            self.geo_dropout = 0.1
            self.climate_dropout = 0.1
        else:
            self.encoder_drop_probs=None
            self.decoder_drop_probs=None
            self.decoder_drop = None
            self.geo_dropout = 0.
            self.climate_dropout = 0.
        
        self.activation = activation
        self.channel_activation = ChannelGLU()

        # Stem
        self.stem = CNNBlock(
            channels_in=input_dim,
            channels_out=dims[0],
            chw=None,
            activation=self.activation,
        )

        # Encoder (no flattening)
        self.encoder = FoundationEncoder(
            input_dim=dims[0],
            depths=depths,
            dims=dims,
            img_size=img_size,
            activation=self.channel_activation,  # or self.activation
            encoder_drop_probs=self.encoder_drop_probs,
        )

        # Decoder
        if fixed_task != 'coords':
            self.decoder = FoundationDecoder(
                depths=depths[::-1],
                dims=dims[::-1],
                img_size=img_size,
                dropout=dropout,
                activation=self.channel_activation,  # or self.activation
                ov_compatiblity=ov_compatiblity,
                decoder_drop_probs=self.decoder_drop_probs,
            )
        else:
            self.decoder = nn.Conv2d(4, 2, kernel_size=1)

        # HEADS
        # 1) Reconstruction head
        if self.fixed_task is None or self.fixed_task == "reconstruction":
            self.head_recon = CNNBlock(
                channels_in=self.dims[0],
                channels_out=self.output_dim,
                chw=[self.output_dim, self.img_size, self.img_size],
                activation=self.activation,
                activation_out=nn.Sigmoid(),  # or None, depending on your use-case
            )
        else:
            self.head_recon = nn.Conv2d(4, 2, kernel_size=1)

        # 2) Climate zone segmentation head (31 classes)
        # self.head_seg = CNNBlock(
        #     channels_in=self.dims[0],
        #     channels_out=31,
        #     chw=None,
        #     activation=self.activation,
        #     activation_out=nn.Identity(),  # <-- ensures raw logits
        #     drop_prob_main=self.climate_dropout
        # )
        if self.fixed_task is None or self.fixed_task == "climate":
            head_seg_channels = [self.dims[0], 128, 256, 128, 31]
            self.head_seg = nn.Sequential(
                *[
                    CNNBlock(
                        channels_in=head_seg_channels[i],
                        channels_out=head_seg_channels[i + 1],
                        chw=None,
                        activation=self.activation,
                        activation_out=nn.Identity() if i == len(head_seg_channels) - 2 else None,
                        drop_prob_main=0.0 if i == len(head_seg_channels) - 2 else self.climate_dropout,
                    )
                    for i in range(len(head_seg_channels) - 1)
                ]
            )
        else:
            self.head_seg = nn.Conv2d(4, 2, kernel_size=1)

        # 3) Geolocation head (latitude, sin(lon), cos(lon), optional extra?)
        #    We'll pool the final encoder features globally to get (B, C).
        if self.fixed_task is None or self.fixed_task == "coords":
            self.head_geo = nn.Sequential(
                nn.Linear(self.dims[-1]  * 2, 128),
                self.activation,
                *([] if self.geo_dropout == 0 else [nn.Dropout(p=self.geo_dropout)]),
                nn.Linear(128, 4),
                nn.Tanh()
            )
        else:
            self.head_geo = nn.Conv2d(4, 2, kernel_size=1)

        # 4) Zoom level head (single scalar)
        if self.apply_zoom:
            self.head_zoom = nn.Sequential(
                nn.Linear(self.dims[-1] * 2, 64),
                self.activation,
                nn.Linear(64, 1)
            )

    def pool_feats(self, x):
        avg_pooled_feats = nn.AdaptiveAvgPool2d(1)(x)  # Shape: (B, C, 1, 1)
        max_pooled_feats = nn.AdaptiveMaxPool2d(1)(x)  # Shape: (B, C, 1, 1)

        combined_pooled_feats = torch.cat((avg_pooled_feats, max_pooled_feats), dim=1)  # Shape: (B, 2*C, 1, 1)
        pooled_feats = combined_pooled_feats.view(combined_pooled_feats.size(0), -1)  # Shape: (B, 2*C)
        return pooled_feats

    def create_dummy_tensor(self, B, C, H, W, device=None):
        dummy = torch.zeros(B, C, H, W, device=device)
        half = H // 2
        dummy[:, 0, :half, :] = 10.0
        dummy[:, 1, half:, :] = 10.0
        return dummy

    def forward(self, x, print_outs = False):
        """
        Args:
            x: (B, 8, 128, 128)
        Returns:
            dict of tasks:
              coords -> (B, 4)
              climate -> (B, 31, 128, 128)
              zoom_factor -> (B, 1)
              reconstruction -> (B, 3, 128, 128)
        """
        # Stem
        x = self.stem(x)  # (B, dims[0], H, W)

        # Encoder
        bottom_feats, skips = self.encoder(x)
        # bottom_feats shape: (B, dims[-1], H_enc, W_enc)
        # H_enc, W_enc = 128 / 2^(steps-1) if each stage does 2x downsample

        # Global average pooling for geolocation / zoom
        # This yields (B, dims[-1])
        # pooled_feats = bottom_feats.mean(dim=(2, 3))  # global average across spatial dims
        pooled_feats = self.pool_feats(bottom_feats)

        # Heads for global tasks
        if self.fixed_task is None or self.fixed_task == "coords":
            geo_pred = self.head_geo(pooled_feats)  # (B, 4)
        else:
            # geo_pred = torch.zeros(x.size(0), 4, device=x.device)
            geo_pred = None

        # Decoder
        if self.fixed_task is None or self.fixed_task in ["reconstruction", "climate"]:
            decoded_feats = self.decoder(bottom_feats, skips)  # (B, dims[0], 128, 128)
        else:
            decoded_feats = None

        # Reconstruction
        if self.fixed_task is None or self.fixed_task == "reconstruction":
            reconstruction = self.head_recon(decoded_feats)  # (B, output_dim, 128, 128)
        else:
            # reconstruction = torch.zeros(x.size(0), self.output_dim, self.img_size, self.img_size, device=x.device)
            reconstruction = None

        # Segmentation
        if self.fixed_task is None or self.fixed_task == "climate":
            climate_logits = self.head_seg(decoded_feats)  # (B, 31, 128, 128)
        else:
            # climate_logits = self.create_dummy_tensor(x.size(0), 31, self.img_size, self.img_size, device=x.device)
            climate_logits = None
        
        if print_outs: 
            outputs = [
                ("stem out", x.shape, x.mean(), x.std(), x.min(), x.max()),
                ("encoder out", bottom_feats.shape, bottom_feats.mean(), bottom_feats.std(), bottom_feats.min(), bottom_feats.max()),
                ("pool out", pooled_feats.shape, pooled_feats.mean(), pooled_feats.std(), pooled_feats.min(), pooled_feats.max()),
                ("coords out", geo_pred.shape, geo_pred.mean(), geo_pred.std(), geo_pred.min(), geo_pred.max()),
                ("decoder out", decoded_feats.shape, decoded_feats.mean(), decoded_feats.std(), decoded_feats.min(), decoded_feats.max()),
                ("reconstruction out", reconstruction.shape, reconstruction.mean(), reconstruction.std(), reconstruction.min(), reconstruction.max()),
                ("climate out", climate_logits.shape, climate_logits.mean(), climate_logits.std(), climate_logits.min(), climate_logits.max()),
            ]

            # Prepare data for tabulate
            table_data = []
            for name, shape, mean, std, min_, max_ in outputs:
                table_data.append([
                    name,
                    str(shape),
                    f"{mean:.3e}",
                    f"{std:.3e}",
                    f"{min_:.3e} - {max_:.3e}"
                ])

            # Table headers
            headers = ["Layer", "Shape", "Mean", "Std Dev", "Min - Max"]
            print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".3e"))

        if self.apply_zoom:
            raise NotImplementedError('While the zoom task is implemented, it should not be used in this model')
            zoom_pred = self.head_zoom(pooled_feats) # (B, 1)

            return {
                "coords": geo_pred,              # (B, 4)
                "climate": climate_logits,       # (B, 31, 128, 128)
                "zoom_factor": zoom_pred,        # (B, 1)
                "reconstruction": reconstruction # (B, 8, 128, 128)
            }
        else:
            return {
                "coords": geo_pred,              # (B, 4)
                "climate": climate_logits,       # (B, 31, 128, 128)
                "reconstruction": reconstruction # (B, 8, 128, 128)
            }










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


class MultiTaskLoss(nn.Module):
    def __init__(self, apply_zoom=False, fixed_task=None, device='cuda'):
        super(MultiTaskLoss, self).__init__()
        
        # Initialize log variances as learnable parameters
        self.apply_zoom = apply_zoom
        self.fixed_task = fixed_task
        
        if fixed_task is None:
            self.log_sigma_recon = nn.Parameter(torch.zeros(1, device=device)) # For reconstruction
            self.log_sigma_perc = nn.Parameter(torch.zeros(1, device=device)) # For perceptual loss
            self.log_sigma_seg = nn.Parameter(torch.zeros(1, device=device)) # For climate segmentation
            self.log_sigma_geo = nn.Parameter(torch.zeros(1, device=device)) # For geolocation
            self.log_sigma_tv = nn.Parameter(torch.zeros(1, device=device))  # For TV loss

            # Initialize scales
            self.scale_recon = 1.
            self.scale_perc = 1.
            self.scale_seg = 1.
            self.scale_geo = 1.
            self.scale_tv = 1.

            if self.apply_zoom:
                raise NotImplementedError('While the zoom task is implemented, it should not be used in this model')
                self.log_sigma_zoom = nn.Parameter(torch.zeros(1)) # For zoom level
                self.scale_zoom = 1.
        
        # CLIMATE SEGMENTATION
        if fixed_task is None or fixed_task == "climate":
            # IF PATCH SIZE 128
            if False:
                class_counts = {0: 1549247552, 1: 142519569, 2: 166608107, 3: 727792659,
                                4: 1186275975, 5: 440328614, 6: 427634848, 7: 550045685, 
                                8: 73479446, 9: 57963957, 10: 1020444, 11: 220679379, 
                                12: 108179283, 13: 732136, 14: 332133077, 15: 126554991, 
                                16: 4698469, 17: 12629170, 18: 36806304, 19: 93155052, 
                                20: 3677890, 21: 87460863, 22: 78153774, 23: 204252025, 
                                24: 23775016, 25: 136446293, 26: 384372285, 27: 952531678, 
                                28: 40078754, 29: 514620810, 30: 26158823}

            # IF PATCH SIZE 256
            else:
                class_counts = {0: 1549444778, 1: 142382282, 2: 166764223, 3: 726361887, 
                                4: 1185407623, 5: 440376776, 6: 427143328, 7: 549611278, 
                                8: 73397526, 9: 57980341, 10: 1020444, 11: 220250563, 
                                12: 108221291, 13: 732136, 14: 331937481, 15: 126418136, 
                                16: 4698469, 17: 12563634, 18: 36796947, 19: 93112211, 
                                20: 3677890, 21: 87395947, 22: 78061994, 23: 203878488, 
                                24: 23744073, 25: 136426482, 26: 384078333, 27: 952557416, 
                                28: 40013218, 29: 514598398, 30: 26158823}


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
        
        if fixed_task is None or fixed_task == "coords":
            self.mse_loss = nn.MSELoss()
        
        if fixed_task is None or fixed_task == "reconstruction":
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
            self.log_sigma_perc.data.clamp_(min=-2, max=1)
            self.log_sigma_seg.data.clamp_(min=-2, max=1)
            self.log_sigma_geo.data.clamp_(min=-2, max=1)
            self.log_sigma_tv.data.clamp_(min=-2, max=1)

            if self.apply_zoom:
                self.log_sigma_zoom.data.clamp_(min=-2, max=1)

        # Reconstruction Loss (Pixel-wise and Perceptual)
        if self.fixed_task is None or self.fixed_task == "reconstruction":
            loss_recon = F.smooth_l1_loss(output["reconstruction"], labels["reconstruction"])
            loss_perceptual = self.perceptual_loss(output["reconstruction"], labels["reconstruction"])
        
        # Climate Segmentation Loss (Cross-entropy and Total Variation)
        if self.fixed_task is None or self.fixed_task == "climate":
            loss_climate = self.ce_loss(output["climate"], labels["climate"])
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
                + (0.5 * torch.exp(-2 * self.log_sigma_perc) * loss_perceptual * self.scale_perc + self.log_sigma_perc)
                + (0.5 * torch.exp(-2 * self.log_sigma_seg)  * loss_climate * self.scale_seg     + self.log_sigma_seg)
                + (0.5 * torch.exp(-2 * self.log_sigma_geo)  * loss_geo * self.scale_geo         + self.log_sigma_geo)
                + (0.5 * torch.exp(-2 * self.log_sigma_tv)   * loss_tv * self.scale_tv           + self.log_sigma_tv)
            )
            
            if self.apply_zoom:
                loss += (0.5 * torch.exp(-2 * self.log_sigma_zoom) * loss_zoom * self.scale_zoom + self.log_sigma_zoom)
                
        elif self.fixed_task == "reconstruction":
            loss = loss_recon + loss_perceptual * 0.01
        elif self.fixed_task == "climate":
            loss = loss_climate + loss_tv * 0.01
        elif self.fixed_task == "coords":
            loss = loss_geo
        else:
            raise ValueError(f"Task {self.fixed_task} is not among the tasks available")

        if self.fixed_task is None:
            log_loss = {
                'total_loss': loss.item(),
                'loss_components': {
                'reconstruction': loss_recon.item(),
                'perceptual': loss_perceptual.item(),
                'climate_segmentation': loss_climate.item(),
                'total_variation': loss_tv.item(),
                'geolocation': loss_geo.item(),
                },
                'log_sigmas': {
                'log_sigma_recon': self.log_sigma_recon.item(),
                'log_sigma_perc': self.log_sigma_perc.item(),
                'log_sigma_seg': self.log_sigma_seg.item(),
                'log_sigma_tv': self.log_sigma_tv.item(),
                'log_sigma_geo': self.log_sigma_geo.item(),
                },
                'scaled_loss':{
                'reconstruction': (0.5 * torch.exp(-2 * self.log_sigma_recon)  * loss_recon  * self.scale_recon).item(),
                'perceptual': (0.5 * torch.exp(-2 * self.log_sigma_perc) * loss_perceptual * self.scale_perc).item(),
                'climate_segmentation': (0.5 * torch.exp(-2 * self.log_sigma_seg)  * loss_climate * self.scale_seg).item(),
                'total_variation': (0.5 * torch.exp(-2 * self.log_sigma_tv)   * loss_tv * self.scale_tv).item(),
                'geolocation': (0.5 * torch.exp(-2 * self.log_sigma_geo)  * loss_geo * self.scale_geo).item(),
                }
            }

            if self.apply_zoom:
                log_loss['loss_components']['zoom_level'] = loss_zoom.item()
                log_loss['log_sigmas']['log_sigma_zoom'] = self.log_sigma_zoom.item()
                log_loss['scaled_loss']['zoom_level'] = (0.5 * torch.exp(-2 * self.log_sigma_zoom) * loss_zoom * self.scale_zoom).item()

        elif self.fixed_task == "reconstruction":
            log_loss = {
                'total_loss': loss.item(),
                'loss_components': {
                    'reconstruction': loss_recon.item(),
                    'perceptual': loss_perceptual.item()
                }
            }
        
        elif self.fixed_task == "climate":
            log_loss = {
                'total_loss': loss.item(),
                'loss_components': {
                    'climate_segmentation': loss_climate.item(),
                    'total_variation': loss_tv.item()
                }
            }
        
        elif self.fixed_task == "coords":
            log_loss = {
                'total_loss': loss.item(),
                'loss_components': {
                    'coords': loss_geo.item()
                }
            }
        
        return loss, log_loss




# -------------------------------------------------------------------
# CLASSIFIER
# -------------------------------------------------------------------


class DownstreamPhiSat2(nn.Module):
    def __init__(
        self,
        *,
        input_dim=3,
        output_dim=None,
        depths=None,
        dims=None,
        img_size=128,
        dropout=True,
        activation=nn.GELU(),
        ov_compatiblity=False,
        task='classification'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim if output_dim is None else output_dim
        self.depths = depths
        self.dims = dims
        self.img_size = img_size
        self.task = task
        if ov_compatiblity:
            self.activation = activation
            self.channel_activation = ChannelGLU()
        else:
            self.activation = activation
            self.channel_activation = ChannelGLU()

        self.dropout = dropout

        if dropout:
            # Example: same pattern as your Foundation Model
            self.encoder_drop_probs = [0.0, 0.05, 0.1, 0.15]
            self.decoder_drop_probs = [0.1, 0.15, 0.15, 0.2]
            self.decoder_drop = 0.1

            self.class_dropout = 0.1
            self.segm_dropout  = 0.15
        else:
            self.encoder_drop_probs = None
            self.decoder_drop_probs = None
            self.decoder_drop = None

            self.class_dropout = 0.0
            self.segm_dropout  = 0.0

        # -------------------------------
        # Stem (no flattening)
        # -------------------------------
        self.stem = CNNBlock(
            channels_in=input_dim,
            channels_out=dims[0],
            chw=None,
            activation=self.activation,
        )

        # -------------------------------
        # Encoder
        # -------------------------------
        self.encoder = FoundationEncoder(
            input_dim=dims[0],
            depths=depths,
            dims=dims,
            img_size=img_size,
            activation=self.channel_activation,
            encoder_drop_probs=self.encoder_drop_probs,
        )

        # -------------------------------
        # Decoder
        # -------------------------------
        self.decoder = FoundationDecoder(
            depths=depths[::-1],
            dims=dims[::-1],
            img_size=img_size,
            dropout=self.decoder_drop,
            activation=self.channel_activation,
            ov_compatiblity=ov_compatiblity,
            decoder_drop_probs=self.decoder_drop_probs,
        )

        # -------------------------------
        # Heads (choose by task)
        # -------------------------------
        if self.task == 'classification':
            # Example: classification head with optional dropout
            # after Flatten and before final Linear
            # self.head_clas = nn.Sequential(
            #     nn.AdaptiveAvgPool2d((1, 1)),
            #     nn.Flatten(),
            #     *([] if self.class_dropout == 0 else [nn.Dropout(p=self.class_dropout)]),
            #     nn.Linear(self.dims[-1], self.output_dim),
            # )

            self.head_clas = nn.Sequential(
                nn.Linear(self.dims[-1]  * 2, 128),
                nn.GELU(),
                *([] if self.class_dropout == 0 else [nn.Dropout(p=self.class_dropout)]),
                nn.Linear(128, self.output_dim),
            )

        elif self.task == 'segmentation':
            # Example: segmentation head with dropout in the CNNBlock
            self.head_segm = CNNBlock(
                channels_in=self.dims[0],
                channels_out=self.output_dim,
                # chw=[self.output_dim, self.img_size, self.img_size],
                chw=None,
                activation=self.activation,
                activation_out=nn.Identity(),
                drop_prob_main=self.segm_dropout   # apply main dropout in the block
            )
        
        else:
            raise ValueError(f"Invalid task: {self.task}")

    def pool_feats(self, x):
        avg_pooled_feats = nn.AdaptiveAvgPool2d(1)(x)  # Shape: (B, C, 1, 1)
        max_pooled_feats = nn.AdaptiveMaxPool2d(1)(x)  # Shape: (B, C, 1, 1)

        combined_pooled_feats = torch.cat((avg_pooled_feats, max_pooled_feats), dim=1)  # Shape: (B, 2*C, 1, 1)
        pooled_feats = combined_pooled_feats.view(combined_pooled_feats.size(0), -1)  # Shape: (B, 2*C)
        return pooled_feats

    def forward(self, x):
        x = self.stem(x)
        x, skips = self.encoder(x)
        if self.task == 'classification':
            x = self.pool_feats(x)
            x = self.head_clas(x)
        elif self.task == 'segmentation':
            x = self.decoder(x, skips)
            x = self.head_segm(x)
        else:
            raise ValueError(f"Invalid task: {self.task}")
        return x



def pretrained_phisat2_downstream(
    checkpoint_path: str,
    freeze_body: bool = True,
    output_dim_fm: int = 8,
    output_dim_dw: int = 11,
    ov_compatiblity: bool = False,
    task: str = 'classification',
    **kwargs
):

    # 1. Instantiate the foundation model (with matching hyperparams)
    foundation_model = FoundationModel4Task(output_dim=output_dim_fm, **kwargs)

    # 2. Load the foundation model's state dict from checkpoint
    foundation_state_dict = torch.load(checkpoint_path, map_location="cpu")

    new_state_dict = OrderedDict()
    for key, val in foundation_state_dict.items():
        # Remove "module." if it exists
        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key
        if 'zoom' in new_key:
            continue
        new_state_dict[new_key] = val

    foundation_model.load_state_dict(new_state_dict, strict=True)

    # 3. Instantiate the downstream with same hyperparams
    print(f"ov_compatiblity: {ov_compatiblity}")
    print(f"task: {task}")
    downstream = DownstreamPhiSat2(output_dim = output_dim_dw, ov_compatiblity=ov_compatiblity, task=task, **kwargs)

    # 4. Optional dimension checks before copying over weights
    assert foundation_model.dims == downstream.dims, (
        "Mismatch in dims between foundation model and downstream: "
        f"{foundation_model.dims} vs {downstream.dims}"
    )
    assert foundation_model.depths == downstream.depths, (
        "Mismatch in depths between foundation model and downstream: "
        f"{foundation_model.depths} vs {downstream.depths}"
    )

    # 5. Copy stem weights
    downstream.stem.load_state_dict(foundation_model.stem.state_dict(), strict=True)

    # 6. Copy encoder weights
    downstream.encoder.load_state_dict(foundation_model.encoder.state_dict(), strict=True)

    # 7. Optionally freeze the stem and encoder weights
    if freeze_body:
        for param in downstream.stem.parameters():
            param.requires_grad = False
            
        for param in downstream.encoder.parameters():
            param.requires_grad = False
            
        downstream.stem.eval()
        downstream.encoder.eval()

    return downstream



def get_phisat2_model(
    model_size = 'nano',
    apply_zoom = False,
    return_model=None, # 'classifier', 'pretrain', 'pretrain_compatible', None
    fixed_task=None,
    **kwargs
    ):
    
    if model_size == 'nano':            # Full mode: 298.00 MB -- Encoder: 73.690 MB
        # depths = [2, 2]
        # dims = [2, 4]
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
        dims = [2, 4]

    else:
        raise ValueError(f"Invalid model size: {model_size}")
    
    if return_model == 'downstream_segmentation':
        return DownstreamPhiSat2(depths=depths, dims=dims, ov_compatiblity=True, task='segmentation', **kwargs)

    if return_model == 'downstream_classification':
        return DownstreamPhiSat2(depths=depths, dims=dims, ov_compatiblity=True, task='classification', **kwargs)

    elif return_model == 'pretrain':
        return FoundationModel4Task(depths=depths, 
                                    dims=dims, 
                                    ov_compatiblity=False, 
                                    dropout=True, 
                                    apply_zoom=apply_zoom, 
                                    fixed_task=fixed_task,
                                    **kwargs
                                    )

    elif return_model == 'pretrain_compatible':
        return FoundationModel4Task(depths=depths, 
                                    dims=dims, 
                                    ov_compatiblity=True, 
                                    dropout=True, 
                                    apply_zoom=apply_zoom, 
                                    fixed_task=fixed_task,
                                    **kwargs
                                    )
    
    elif return_model is None:
        updated_kwargs = kwargs.copy()
        updated_kwargs.update({'depths': depths, 'dims': dims})
        return updated_kwargs









import torch
import torch.nn as nn

from tabulate import tabulate

NORM_SIZE_THRESHOLD = 64

def get_stage_norm(spatial_size, norm_size_threshold):
    if spatial_size >= norm_size_threshold:
        return 'batch'  # Parameter-efficient for large spatial sizes
    else:
        return 'layer'  # Stable for small sizes


# -------------------------------------------------------------------
# FoundationEncoder without flattening
# -------------------------------------------------------------------
class FoundationEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim=3,
        depths=None,
        dims=None,
        img_size=64,
        activation=nn.ReLU6(),
        norm='batch',
        encoder_drop_probs=None,
    ):
        """
        A U-Net style encoder that:
         1) progressively downsamples
         2) returns final spatial features (B, dims[-1], H_enc, W_enc)
         3) returns skip connections
        """
        super().__init__()

        self.depths = depths
        self.dims = dims
        self.input_dim = input_dim
        self.img_size = img_size
        self.activation = activation

        # Compute how many steps (downsampling stages) we have
        self.steps = len(depths)
        self.sizes = [img_size // (2**i) for i in range(self.steps)]

        if encoder_drop_probs is None:
            encoder_drop_probs = [0.0]*self.steps

        # -------------------------------------------------------------
        # Downsample modules
        # -------------------------------------------------------------
        self.downsample = nn.ModuleList()
        for i in range(self.steps - 1):
            # e.g. dims[i] -> dims[i+1], with a stride-2 or maxpool down
            self.downsample.append(
                nn.Sequential(
                    nn.Conv2d(self.dims[i], self.dims[i + 1], 1, padding=0),
                    nn.MaxPool2d(2, stride=2),
                )
            )

        # -------------------------------------------------------------
        # Blocks + Skip-scale modules
        # -------------------------------------------------------------
        self.block_scalers = nn.ModuleList([ScaleSkip2D(self.dims[i]) for i in range(self.steps)])
        self.blocks_down = nn.ModuleList()

        for i in range(self.steps):
            stage_blocks = nn.ModuleList()
            # We'll use the same drop probs for each block at stage i
            drop_prob_main = encoder_drop_probs[i]

            for _ in range(depths[i]):
                if norm == 'batch_layer':
                    norm_depth = get_stage_norm(self.sizes[i], NORM_SIZE_THRESHOLD)
                else:
                    norm_depth = norm
                    
                if norm_depth == 'batch':
                    chw = None
                elif norm_depth == 'layer':
                    chw = [self.dims[i], self.sizes[i], self.sizes[i]]
                else:
                    chw = 'group_norm'

                stage_blocks.append(
                    CNNBlock(
                        channels_in=self.dims[i],
                        channels_out=self.dims[i],
                        chw=chw,
                        activation=self.activation,
                        drop_prob_main=drop_prob_main
                    )
                )
            self.blocks_down.append(stage_blocks)

    def forward(self, x):
        """
        Returns:
          final_feats: spatial feature map at the bottom of U-Net (B, dims[-1], H_enc, W_enc)
          skips: list of skip connection feature maps
        """
        skips = []

        for i in range(self.steps):
            
            # (1) Local blocks
            pre_block = x
            
            for j in range(self.depths[i]):
                x = self.blocks_down[i][j](x)

            # If we stacked more than one block on this stage, do the skip-scaler
            if self.depths[i] > 1:
                x = self.block_scalers[i](x, pre_block)

            # (2) Collect skip at this stage
            skips.append(x)

            # (3) Downsample, except for the last iteration i=steps-1
            if i < self.steps - 1:
                x = self.downsample[i](x)
        # x is now the bottom-most representation: (B, dims[-1], H_enc, W_enc)
        return x, skips

# -------------------------------------------------------------------
# FoundationDecoder
# -------------------------------------------------------------------
class FoundationDecoder(nn.Module):
    def __init__(
        self,
        *,
        depths=None,
        dims=None,
        img_size=64,
        dropout=None,
        activation=nn.ReLU6(),
        norm='batch',
        ov_compatiblity=False,
        decoder_drop_probs=None
    ):
        """
        A U-Net style decoder that:
         1) up-samples the bottom feature map
         2) merges with skip connections
         3) returns final high-res feature map (B, dims[0], H, W)
        """
        super().__init__()
        self.depths = depths
        self.dims = dims
        self.img_size = img_size
        self.activation = activation
        

        self.steps = len(depths)
        # sizes in reverse for the decoder
        self.sizes = [img_size // (2**(self.steps - i - 1)) for i in range(self.steps)]

        # Handle dropout (optional)
        if dropout is None:
            dropout = [None] * self.steps
        elif isinstance(dropout, (int, float)):
            dropout = [dropout] * self.steps
        self.dropout = dropout

        if decoder_drop_probs is None:
            decoder_drop_probs = [0.0]*len(depths)

        # -------------------------------------------------------------
        # Skip & block scalers, blocks
        # -------------------------------------------------------------
        self.skip_scalers = nn.ModuleList()
        self.block_scalers = nn.ModuleList()
        self.blocks_up = nn.ModuleList()

        for i in range(self.steps):
            # self.skip_scalers.append(ScaleSkip2D(self.dims[i], drop_y=self.dropout[i], signal_to_noise=(0.1, 0.9)))
            self.skip_scalers.append(ScaleSkip2D(self.dims[i]))
            self.block_scalers.append(ScaleSkip2D(self.dims[i]))

            stage_blocks = nn.ModuleList()
            # We'll use the same drop probs for each block at stage i
            drop_prob_main = decoder_drop_probs[i]
            
            for _ in range(depths[i]):

                if norm == 'batch_layer':
                    norm_depth = get_stage_norm(self.sizes[i], NORM_SIZE_THRESHOLD)
                else:
                    norm_depth = norm
                    
                if norm_depth == 'batch':
                    chw = None
                elif norm_depth == 'layer':
                    chw = [self.dims[i], self.sizes[i], self.sizes[i]]
                else:
                    chw = 'group_norm'

                stage_blocks.append(
                    CNNBlock(
                        channels_in=self.dims[i],
                        channels_out=self.dims[i],
                        chw=chw,
                        activation=self.activation,
                        drop_prob_main=drop_prob_main
                    )
                )
            self.blocks_up.append(stage_blocks)


        # -------------------------------------------------------------
        # Upsamplers
        # -------------------------------------------------------------
        self.upsamplers = nn.ModuleList()
        for i in range(self.steps - 1):
            upsample_size = self.sizes[i + 1]
            upsample_norm = 'batch' if upsample_size >= NORM_SIZE_THRESHOLD else 'layer'

            if upsample_norm == 'batch':
                norm_layer = nn.BatchNorm2d(self.dims[i + 1])
            else:
                norm_layer = nn.LayerNorm([self.dims[i + 1], upsample_size, upsample_size])

            self.upsamplers.append(
                nn.Sequential(
                    make_bilinear_upsample(self.dims[i]) if ov_compatiblity else nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.Conv2d(self.dims[i], self.dims[i + 1], 3, padding=1, bias=False, padding_mode='replicate'),
                    norm_layer,
                    self.activation,
                )
            )

        self.prehead_norm = nn.LayerNorm([self.dims[-1], self.sizes[-1], self.sizes[-1]])

    def forward(self, x, skips):
        """
        x: (B, dims[-1], H_enc, W_enc) bottom_feats from the encoder
        skips: list of skip features from each encoder stage
        returns: (B, dims[0], H, W)
        """

        for i in range(self.steps):
            # (1) Upsample, except for the first iteration i=0:
            if i > 0:
                x = self.upsamplers[i-1](x)

            # (2) Merge skip from the corresponding stage
            skip_x = skips[-(i + 1)]
            x = self.skip_scalers[i](x, skip_x)

            # (3) Local blocks
            pre_block = x
            for block in self.blocks_up[i]:
                x = block(x)
            if self.depths[i] > 1:
                x = self.block_scalers[i](x, pre_block)

        x = self.prehead_norm(x)
        return x


# -------------------------------------------------------------------
# The combined multi-task model
# -------------------------------------------------------------------
class phisat2net_uniphi(nn.Module):
    def __init__(
        self,
        *,
        input_dim=3,
        output_dim=None,
        depths=None,
        dims=None,
        img_size=128,
        dropout=True,
        activation=nn.GELU(),
        ov_compatiblity=False,
        apply_zoom=False,
        fixed_task=None,
        climate_segm=False,
    ):
        """
        A U-Net-like model with extra heads for:
          - climate zone segmentation
          - geolocation
          - zoom level
          - reconstruction
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim if output_dim is None else output_dim
        self.depths = depths
        self.dims = dims
        self.img_size = img_size
        self.apply_zoom = apply_zoom
        self.fixed_task = fixed_task
        self.climate_segm = climate_segm

        self.dropout = dropout
        
        if dropout:
            self.encoder_drop_probs=[0.05, 0.1, 0.15, 0.2]
            self.decoder_drop_probs=[0.05, 0.1, 0.15, 0.2]
            self.decoder_drop = 0.15
            self.geo_dropout = 0.15
            self.climate_dropout = 0.15
        else:
            self.encoder_drop_probs=None
            self.decoder_drop_probs=None
            self.decoder_drop = None
            self.geo_dropout = 0.
            self.climate_dropout = 0.
        
        self.activation = activation
        self.channel_activation = ChannelGLU()

        # Stem
        self.stem = CNNBlock(
            channels_in=input_dim,
            channels_out=dims[0],
            chw=None,
            activation=self.activation,
        )

        # Encoder (no flattening)
        self.encoder = FoundationEncoder(
            input_dim=dims[0],
            depths=depths,
            dims=dims,
            img_size=img_size,
            activation=self.channel_activation,  # or self.activation
            encoder_drop_probs=self.encoder_drop_probs,
            norm='batch_layer',
        )

        # Decoder
        if fixed_task != 'coords':
            self.decoder = FoundationDecoder(
                depths=depths[::-1],
                dims=dims[::-1],
                img_size=img_size,
                dropout=dropout,
                activation=self.channel_activation,  # or self.activation
                ov_compatiblity=ov_compatiblity,
                decoder_drop_probs=self.decoder_drop_probs,
                norm='batch_layer',
            )
        else:
            self.decoder = nn.Conv2d(4, 2, kernel_size=1)

        # HEADS
        
        # 1) Reconstruction head
        if self.fixed_task is None or self.fixed_task == "reconstruction":
            self.head_recon = CNNBlock(
                channels_in=self.dims[0],
                channels_out=self.output_dim,
                chw=[self.output_dim, self.img_size, self.img_size],
                activation=self.activation,
                activation_out=nn.Sigmoid(),  # or None, depending on your use-case
            )
        else:
            self.head_recon = nn.Conv2d(4, 2, kernel_size=1)

        # 2) Climate zone segmentation head (31 classes)
        if self.fixed_task is None or self.fixed_task == "climate":
            if self.climate_segm:
                head_seg_channels = [self.dims[0], 128, 256, 128, 31]
                self.head_seg = nn.Sequential(
                    *[
                        CNNBlock(
                            channels_in=head_seg_channels[i],
                            channels_out=head_seg_channels[i + 1],
                            chw=None,
                            activation=self.activation,
                            activation_out=nn.Identity() if i == len(head_seg_channels) - 2 else None,
                            drop_prob_main=0.0 if i == len(head_seg_channels) - 2 else self.climate_dropout,
                        )
                        for i in range(len(head_seg_channels) - 1)
                    ]
                )
            else:
                self.head_seg = nn.Sequential(
                    nn.Linear(self.dims[-1]  * 2, 128),
                    self.activation,
                    *([] if self.climate_dropout == 0 else [nn.Dropout(p=self.climate_dropout)]),
                    nn.Linear(128, 31),
                )

        else:
            self.head_seg = nn.Conv2d(4, 2, kernel_size=1)

        # 3) Geolocation head (latitude, sin(lon), cos(lon), optional extra?)
        #    We'll pool the final encoder features globally to get (B, C).
        if self.fixed_task is None or self.fixed_task == "coords":
            self.head_geo = nn.Sequential(
                nn.Linear(self.dims[-1]  * 2, 128),
                self.activation,
                *([] if self.geo_dropout == 0 else [nn.Dropout(p=self.geo_dropout)]),
                nn.Linear(128, 4),
                nn.Tanh()
            )
        else:
            self.head_geo = nn.Conv2d(4, 2, kernel_size=1)

        # 4) Zoom level head (single scalar)
        if self.apply_zoom:
            self.head_zoom = nn.Sequential(
                nn.Linear(self.dims[-1] * 2, 64),
                self.activation,
                nn.Linear(64, 1)
            )

    def pool_feats(self, x):
        avg_pooled_feats = nn.AdaptiveAvgPool2d(1)(x)  # Shape: (B, C, 1, 1)
        max_pooled_feats = nn.AdaptiveMaxPool2d(1)(x)  # Shape: (B, C, 1, 1)

        combined_pooled_feats = torch.cat((avg_pooled_feats, max_pooled_feats), dim=1)  # Shape: (B, 2*C, 1, 1)
        pooled_feats = combined_pooled_feats.view(combined_pooled_feats.size(0), -1)  # Shape: (B, 2*C)
        return pooled_feats

    def create_dummy_tensor(self, B, C, H, W, device=None):
        dummy = torch.zeros(B, C, H, W, device=device)
        half = H // 2
        dummy[:, 0, :half, :] = 10.0
        dummy[:, 1, half:, :] = 10.0
        return dummy

    def forward(self, x, print_outs = False):
        """
        Args:
            x: (B, 8, 128, 128)
        Returns:
            dict of tasks:
              coords -> (B, 4)
              climate -> (B, 31, 128, 128)
              zoom_factor -> (B, 1)
              reconstruction -> (B, 3, 128, 128)
        """
        # Stem
        x = self.stem(x)  # (B, dims[0], H, W)

        # Encoder
        bottom_feats, skips = self.encoder(x)
        # bottom_feats shape: (B, dims[-1], H_enc, W_enc)
        # H_enc, W_enc = 128 / 2^(steps-1) if each stage does 2x downsample

        # Global average pooling for geolocation / zoom
        # This yields (B, dims[-1])
        # pooled_feats = bottom_feats.mean(dim=(2, 3))  # global average across spatial dims
        pooled_feats = self.pool_feats(bottom_feats)

        # Heads for global tasks
        if self.fixed_task is None or self.fixed_task == "coords":
            geo_pred = self.head_geo(pooled_feats)  # (B, 4)
        else:
            # geo_pred = torch.zeros(x.size(0), 4, device=x.device)
            geo_pred = None

        # Decoder
        if self.fixed_task is None or self.fixed_task in ["reconstruction", "climate"]:
            decoded_feats = self.decoder(bottom_feats, skips)  # (B, dims[0], 128, 128)
        else:
            decoded_feats = None

        # Reconstruction
        if self.fixed_task is None or self.fixed_task == "reconstruction":
            reconstruction = self.head_recon(decoded_feats)  # (B, output_dim, 128, 128)
        else:
            # reconstruction = torch.zeros(x.size(0), self.output_dim, self.img_size, self.img_size, device=x.device)
            reconstruction = None

        # Climate
        if self.fixed_task is None or self.fixed_task == "climate":
            if self.climate_segm:
                climate_logits = self.head_seg(decoded_feats)  # (B, 31, 128, 128)
            else:
                climate_logits = self.head_seg(pooled_feats)
        else:
            # climate_logits = self.create_dummy_tensor(x.size(0), 31, self.img_size, self.img_size, device=x.device)
            climate_logits = None
        
        if print_outs: 
            outputs = [
                ("stem out", x.shape, x.mean(), x.std(), x.min(), x.max()),
                ("encoder out", bottom_feats.shape, bottom_feats.mean(), bottom_feats.std(), bottom_feats.min(), bottom_feats.max()),
                ("pool out", pooled_feats.shape, pooled_feats.mean(), pooled_feats.std(), pooled_feats.min(), pooled_feats.max()),
                ("coords out", geo_pred.shape, geo_pred.mean(), geo_pred.std(), geo_pred.min(), geo_pred.max()),
                ("decoder out", decoded_feats.shape, decoded_feats.mean(), decoded_feats.std(), decoded_feats.min(), decoded_feats.max()),
                ("reconstruction out", reconstruction.shape, reconstruction.mean(), reconstruction.std(), reconstruction.min(), reconstruction.max()),
                ("climate out", climate_logits.shape, climate_logits.mean(), climate_logits.std(), climate_logits.min(), climate_logits.max()),
            ]

            # Prepare data for tabulate
            table_data = []
            for name, shape, mean, std, min_, max_ in outputs:
                table_data.append([
                    name,
                    str(shape),
                    f"{mean:.3e}",
                    f"{std:.3e}",
                    f"{min_:.3e} - {max_:.3e}"
                ])

            # Table headers
            headers = ["Layer", "Shape", "Mean", "Std Dev", "Min - Max"]
            print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".3e"))

        if self.apply_zoom:
            raise NotImplementedError('While the zoom task is implemented, it should not be used in this model')
            zoom_pred = self.head_zoom(pooled_feats) # (B, 1)

            return {
                "coords": geo_pred,              # (B, 4)
                "climate": climate_logits,       # (B, 31, 128, 128)
                "zoom_factor": zoom_pred,        # (B, 1)
                "reconstruction": reconstruction # (B, 8, 128, 128)
            }
        else:
            return {
                "coords": geo_pred,              # (B, 4)
                "climate": climate_logits,       # (B, 31, 128, 128)
                "reconstruction": reconstruction # (B, 8, 128, 128)
            }




class phisat2net_uniphi_downstream(nn.Module):
    def __init__(
        self,
        *,
        input_dim=3,
        output_dim=None,
        depths=None,
        dims=None,
        img_size=128,
        dropout=True,
        activation=nn.GELU(),
        ov_compatiblity=False,
        task='classification',
        norm='batch_layer'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim if output_dim is None else output_dim
        self.depths = depths
        self.dims = dims
        self.img_size = img_size
        self.task = task
        self.norm = norm
        if ov_compatiblity:
            self.activation = activation
            self.channel_activation = ChannelGLU()
        else:
            self.activation = activation
            self.channel_activation = ChannelGLU()

        self.dropout = dropout

        if dropout:
            # Example: same pattern as your Foundation Model
            self.encoder_drop_probs = [0.0, 0.05, 0.1, 0.15]
            self.decoder_drop_probs = [0.1, 0.15, 0.15, 0.2]
            self.decoder_drop = 0.1

            self.class_dropout = 0.1
            self.segm_dropout  = 0.15
        else:
            self.encoder_drop_probs = None
            self.decoder_drop_probs = None
            self.decoder_drop = None

            self.class_dropout = 0.0
            self.segm_dropout  = 0.0

        # -------------------------------
        # Stem (no flattening)
        # -------------------------------
        self.stem = CNNBlock(
            channels_in=input_dim,
            channels_out=dims[0],
            chw=None,
            activation=self.activation,
        )

        # -------------------------------
        # Encoder
        # -------------------------------
        self.encoder = FoundationEncoder(
            input_dim=dims[0],
            depths=depths,
            dims=dims,
            img_size=img_size,
            activation=self.channel_activation,
            encoder_drop_probs=self.encoder_drop_probs,
            norm=norm,
        )

        # -------------------------------
        # Decoder
        # -------------------------------
        self.decoder = FoundationDecoder(
            depths=depths[::-1],
            dims=dims[::-1],
            img_size=img_size,
            dropout=self.decoder_drop,
            activation=self.channel_activation,
            ov_compatiblity=ov_compatiblity,
            decoder_drop_probs=self.decoder_drop_probs,
            norm=norm,
        )

        # -------------------------------
        # Heads (choose by task)
        # -------------------------------
        if self.task == 'classification':
            # Example: classification head with optional dropout
            # after Flatten and before final Linear
            # self.head_clas = nn.Sequential(
            #     nn.AdaptiveAvgPool2d((1, 1)),
            #     nn.Flatten(),
            #     *([] if self.class_dropout == 0 else [nn.Dropout(p=self.class_dropout)]),
            #     nn.Linear(self.dims[-1], self.output_dim),
            # )

            self.head_clas = nn.Sequential(
                nn.Linear(self.dims[-1]  * 2, 128),
                nn.GELU(),
                *([] if self.class_dropout == 0 else [nn.Dropout(p=self.class_dropout)]),
                nn.Linear(128, self.output_dim),
            )

        elif self.task == 'segmentation':
            # Example: segmentation head with dropout in the CNNBlock
            self.head_segm = CNNBlock(
                channels_in=self.dims[0],
                channels_out=self.output_dim,
                # chw=[self.output_dim, self.img_size, self.img_size],
                chw=None,
                activation=self.activation,
                activation_out=nn.Identity(),
                drop_prob_main=self.segm_dropout   # apply main dropout in the block
            )
        
        else:
            raise ValueError(f"Invalid task: {self.task}")

    def pool_feats(self, x):
        avg_pooled_feats = nn.AdaptiveAvgPool2d(1)(x)  # Shape: (B, C, 1, 1)
        max_pooled_feats = nn.AdaptiveMaxPool2d(1)(x)  # Shape: (B, C, 1, 1)

        combined_pooled_feats = torch.cat((avg_pooled_feats, max_pooled_feats), dim=1)  # Shape: (B, 2*C, 1, 1)
        pooled_feats = combined_pooled_feats.view(combined_pooled_feats.size(0), -1)  # Shape: (B, 2*C)
        return pooled_feats

    def forward(self, x):
        x = self.stem(x)
        x, skips = self.encoder(x)
        if self.task == 'classification':
            x = self.pool_feats(x)
            x = self.head_clas(x)
        elif self.task == 'segmentation':
            x = self.decoder(x, skips)
            x = self.head_segm(x)
        else:
            raise ValueError(f"Invalid task: {self.task}")
        return x







def load_pretrained_model(
    pretrained_path: str,
    core_args: dict,
    downstream_task: str = "classification",
    downstream_output_dim: int = 10,
    device: str = "cpu",
    freeze_body: bool = False,  # <--- new argument
):
    """
    Loads a pretrained phisat2net_uniphi model from `pretrained_path`
    (which contains only `state_dict`, no config) and creates a 
    phisat2net_uniphi_downstream with matching stem & encoder.

    Args:
        pretrained_path (str): Path to the .pt file with the pretrained state_dict.
        core_args (dict): Dictionary specifying the essential architecture args, e.g.:
            {
                'input_dim': 8,
                'img_size': 224,
                'depths': [2, 2, 8, 2],
                'dims': [80, 160, 320, 640]
            }
        downstream_task (str): "classification" or "segmentation".
        downstream_output_dim (int): Number of classes or channels for the head.
        device (str): "cpu" or "cuda" device.
        freeze_body (bool): If True, freeze the stem and encoder parameters so they 
                            do not update during training.

    Returns:
        downstream_model (nn.Module): A phisat2net_uniphi_downstream model
            with pretrained stem & encoder weights.
    """

    # 1. Load the state_dict from disk
    checkpoint = torch.load(pretrained_path, map_location=device)

    # Remove the "module." prefix in all keys
    pretrained_state_dict = {}
    for k, v in checkpoint.items():
        new_key = k.replace("module.", "")  # remove "module."
        pretrained_state_dict[new_key] = v

    # 2. Provide defaults and merge user-specified `core_args`.
    default_args = {
        "output_dim": 8,
        "dropout": True,
        "activation": nn.GELU(),
        "ov_compatiblity": True,
        "apply_zoom": False,
        "fixed_task": None,
        "climate_segm": False,
    }
    model_args = {**default_args, **core_args}

    # 3. Instantiate the upstream model with these args
    upstream_model = phisat2net_uniphi(**model_args).to(device)

    # 4. Load pretrained weights into the upstream model
    upstream_model.load_state_dict(pretrained_state_dict, strict=True)
    upstream_model.eval()

    # 5. Create the downstream model, matching the upstream stem & encoder
    downstream_model = phisat2net_uniphi_downstream(
        input_dim=model_args["input_dim"],
        output_dim=downstream_output_dim,
        depths=model_args["depths"],
        dims=model_args["dims"],
        img_size=model_args["img_size"],
        dropout=model_args["dropout"],
        activation=model_args["activation"],
        ov_compatiblity=model_args["ov_compatiblity"],
        task=downstream_task,
    ).to(device)

    # 6. Copy the 'stem' and 'encoder' weights from the upstream model
    downstream_model.stem.load_state_dict(upstream_model.stem.state_dict())
    downstream_model.encoder.load_state_dict(upstream_model.encoder.state_dict())

    # 7. Optionally freeze stem and encoder parameters
    if freeze_body:
        for param in downstream_model.stem.parameters():
            param.requires_grad = False
        for param in downstream_model.encoder.parameters():
            param.requires_grad = False

    return downstream_model
