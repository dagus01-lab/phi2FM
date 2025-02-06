import torch
import torch.nn as nn

from typing import List # need because using python 3.8

from utils.training_utils import get_activation
from .geoaware_blocks import CoreCNNBlock, CoreAttentionBlock
from .util_tools import make_bilinear_upsample


# -------------------------------------------------------------------
# FOUNDATION MODEL
# -------------------------------------------------------------------

class phisat2net_geoaware(nn.Module):
    def __init__(
        self,
        *,
        input_dim=3,
        output_dim=None,
        depths=None,
        dims=None,
        img_size=128,
        dropout=True,      # optional, not really used by the Core blocks
        activation="gelu",
        fixed_task=None,
    ):
        """
        A U-Net-like model with extra heads for:
          - climate zone classification
          - geolocation
          - zoom level
          - reconstruction
        """
        super().__init__()

        # Basic model parameters
        self.input_dim = input_dim
        self.output_dim = input_dim if output_dim is None else output_dim
        self.depths = depths
        self.dims = dims
        self.img_size = img_size
        self.fixed_task = fixed_task

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

        # ---------------------
        # 1) Stem
        # ---------------------
        # We will go from input_dim -> dims[0]
        self.stem = CoreCNNBlock(
            in_channels=self.input_dim,
            out_channels=self.dims[0],
            norm="group",
            activation=activation,
            residual=True
        )

        # ---------------------
        # 2) Encoder
        # ---------------------
        # We'll create an encoder from dims[0] -> dims[1] -> dims[2] -> ...
        # with the specified depths
        self.encoder = FoundationEncoder(
            input_dim=self.dims[0],
            depths=self.depths,
            dims=self.dims,  # we already consumed dims[0] in the stem
            norm="group",
            activation=activation,
        )

        # ---------------------
        # Bridge
        # ---------------------
        # Typically we do a small bridging block at the bottom
        # to let the model "breathe" in the bottleneck:
        self.bridge = CoreCNNBlock(
            in_channels=self.dims[-1],
            out_channels=self.dims[-1],
            norm="group",
            activation=activation,
            residual=True
        )

        # ---------------------
        # 3) Decoder
        # ---------------------
        if self.fixed_task != 'coords':
            # We decode from dims[-1] -> dims[-2] -> ... -> dims[0]
            # The FoundationDecoder expects the same # of depths as dims it receives.
            # Note: reversed(rev_dims) will go from highest to lowest
            # but the FoundationDecoder build logic also does reversed() internally.
            # So we pass them in the forward order, but it constructs them in reverse.
            self.decoder = FoundationDecoder(
                depths=self.depths,
                dims=self.dims,
                norm="group",
                activation=activation,
            )
        else:
            # If the only fixed task is coords, skip building a normal decoder
            self.decoder = nn.Identity()


        # ---------------------
        # 4) Heads
        # ---------------------
        # 4.1 Reconstruction head
        if self.fixed_task is None or self.fixed_task == "reconstruction":
            # We'll produce self.output_dim channels (usually 3) 
            # from the final decoder output (which is dims[0] channels).
            self.head_recon = nn.Sequential(
                CoreCNNBlock(
                    in_channels=self.dims[0],
                    out_channels=self.output_dim,
                    norm="group",
                    activation=activation,
                    residual=True
                ),
                nn.Tanh(),  # Apply Tanh activation to be in [-1, 1] range
                ScalingLayer(4)  # Scale by 4
            )


        else:
            self.head_recon = nn.Identity()

        # 4.2 Climate zone segmentation head (31 classes)
        if self.fixed_task is None or self.fixed_task == "climate":
            # segmentation as a global classification (B, 31) 
            # so we do a global pooling => FC => 31 
            self.head_seg = nn.Sequential(
                nn.Linear(self.dims[-1] * 2, 128),
                get_activation(activation),
                *( [nn.Dropout(p=self.climate_dropout)] if self.climate_dropout > 0 else [] ),
                nn.Linear(128, 31),
            )
        else:
            self.head_seg = nn.Identity()

        # 4.3 Geolocation head
        # We'll do global pooling on the bottom feature map
        # shape => (B, 2 * dims[-1]) => -> 128 => -> 4 => Tanh
        if self.fixed_task is None or self.fixed_task == "coords":
            self.head_geo = nn.Sequential(
                nn.Linear(self.dims[-1] * 2, 128),
                get_activation(activation),
                *( [nn.Dropout(p=self.geo_dropout)] if self.geo_dropout > 0 else [] ),
                nn.Linear(128, 4),
                nn.Tanh()
            )
        else:
            self.head_geo = nn.Identity()

    def pool_feats(self, x):
        """
        Pools a 4D feature map (B, C, H, W) into a 1D (B, 2*C) vector
        by concatenating global avg + global max. 
        """
        avg_pooled_feats = nn.AdaptiveAvgPool2d(1)(x)
        max_pooled_feats = nn.AdaptiveMaxPool2d(1)(x)
        combined_pooled_feats = torch.cat((avg_pooled_feats, max_pooled_feats), dim=1)
        pooled_feats = combined_pooled_feats.view(combined_pooled_feats.size(0), -1)
        return pooled_feats

    def forward(self, x):
        """
        Args:
            x: (B, input_dim, H, W) e.g. (B, 8, 128, 128)

        Returns:
            dict with keys: "coords", "climate", "reconstruction"
        """
        
        # ---------------------
        # 1) Stem
        # ---------------------
        # print(f"Input shape: {x.shape}, mean={x.mean().item():.3f}, max={x.max().item():.3f}")
        x_stem = self.stem(x)  # (B, dims[0], H, W)
        # print(f"Stem shape: {x_stem.shape}, mean={x_stem.mean().item():.3f}, max={x_stem.max().item():.3f}")


        # ---------------------
        # 2) Encoder
        # ---------------------
        bottom, skips = self.encoder(x_stem)
        # print(f"Bottom shape: {bottom.shape}, mean={bottom.mean().item():.3f}, max={bottom.max().item():.3f}")
        # bottom: (B, dims[-1], H//(2^num_stages), W//(2^num_stages))
        # skips: list of intermediate features


        # ---------------------
        # 3) Bridge
        # ---------------------
        bottom_feats = self.bridge(bottom)
        # print(f"Bridge shape: {bottom_feats.shape}, mean={bottom_feats.mean().item():.3f}, max={bottom_feats.max().item():.3f}")


        # ---------------------
        # 4) Decoder
        # ---------------------
        if self.fixed_task is None or self.fixed_task == "reconstruction":
            decoded_feats = self.decoder(bottom_feats, skips)  # (B, dims[0], H, W)
            # print(f"Decoded shape: {decoded_feats.shape}, mean={decoded_feats.mean().item():.3f}, max={decoded_feats.max().item():.3f}")  
        else:
            decoded_feats = None


        # ---------------------
        # 5) Task Heads
        # ---------------------
        pooled_feats = self.pool_feats(bottom_feats) # (B, 2*dims[-1])

        # 5.1 Reconstruction
        if self.fixed_task is None or self.fixed_task == "reconstruction":
            reconstruction = self.head_recon(decoded_feats)  # (B, output_dim, H, W)
        else:
            reconstruction = None

        # 5.2 Climate
        if self.fixed_task is None or self.fixed_task == "climate":
            climate_logits = self.head_seg(pooled_feats) # (B, 31)
        else:
            climate_logits = None

        # 5.3 Geolocation
        if self.fixed_task is None or self.fixed_task == "coords":
            geo_pred = self.head_geo(pooled_feats)  # (B, 4)
        else:
            geo_pred = None


        # ---------------------------------------
        # Return dictionary
        # ---------------------------------------
        out_dict = {
            "coords": geo_pred,                # (B, 4)
            "climate": climate_logits,         # (B, 31, H, W) or (B, 31)
            "reconstruction": reconstruction,  # (B, output_dim, H, W)
        }

        return out_dict





class ScalingLayer(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return x * self.scale_factor



# -----------------------------------------
# ENCODER
# -----------------------------------------

class CoreEncoderBlock(nn.Module):
    def __init__(self, depth, in_channels, out_channels, norm="batch", activation="relu"):
        super(CoreEncoderBlock, self).__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.norm = norm

        self.blocks = []
        for i in range(self.depth):
            _in_channels = self.in_channels if i == 0 else self.out_channels
            block = CoreCNNBlock(_in_channels, self.out_channels, norm=self.norm, activation=self.activation)

            self.blocks.append(block)

        self.blocks = nn.Sequential(*self.blocks)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        for i in range(self.depth):
            x = self.blocks[i](x)

        before_downsample = x
        x = self.downsample(x)

        return x, before_downsample




class FoundationEncoder(nn.Module):
    """
    A wrapper around multiple CoreEncoderBlocks. 
    Each stage will reduce the spatial dimension by 2.
    """
    def __init__(
        self,
        input_dim: int,
        depths: List[int],
        dims: List[int],
        norm="batch",
        activation="relu",
    ):
        """
        Args:
            input_dim: number of channels after the 'stem'.
            depths: e.g. [2, 2, 2, 2], how many repeated CoreCNNBlocks per stage.
            dims: e.g. [64, 128, 256, 512].
                  The i-th encoder block goes from dims[i-1] -> dims[i] 
                  except for i == 0, which is input_dim -> dims[0].
        """
        super().__init__()
        assert len(depths) == len(dims), f"depths and dims must have the same length. dims={dims}, depths={depths}"

        self.stages = nn.ModuleList()
        prev_ch = input_dim
        for i in range(len(dims)):
            stage = CoreEncoderBlock(
                depth=depths[i],
                in_channels=prev_ch,
                out_channels=dims[i],
                norm=norm,
                activation=activation,
            )
            self.stages.append(stage)
            prev_ch = dims[i]

    def forward(self, x):
        """
        Returns:
          bottom_feats: the final downsampled feature map,
          skip_list: list of feature maps *before* downsampling 
                     for each stage, used by the decoder.
        """
        skip_list = []
        for stage in self.stages:
            x, before_downsample = stage(x)
            skip_list.append(before_downsample)
        return x, skip_list


# -----------------------------------------
# DECODER
# -----------------------------------------

class CoreDecoderBlock(nn.Module):
    def __init__(self, depth, in_channels, out_channels, *, norm="batch", activation="relu"):
        super(CoreDecoderBlock, self).__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_blocks = activation
        self.activation = get_activation(activation)
        self.norm = norm

        self.upsample = make_bilinear_upsample(self.in_channels)
        self.match_channels = CoreCNNBlock(self.in_channels * 2, self.out_channels, norm=self.norm, activation=self.activation_blocks)
        self.attention = CoreAttentionBlock(self.in_channels, self.in_channels, norm=self.norm, activation=self.activation_blocks)

        self.blocks = []
        for _ in range(self.depth):
            block = CoreCNNBlock(self.out_channels, self.out_channels, norm=self.norm, activation=self.activation_blocks)
            self.blocks.append(block)

        self.blocks = nn.Sequential(*self.blocks)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        attn_s, attn_c = self.attention(x, skip)
        x = torch.cat([x, (skip * attn_s) + (skip + attn_c)], dim=1)
        x = self.match_channels(x)

        for i in range(self.depth):
            x = self.blocks[i](x)

        return x




class FoundationDecoder(nn.Module):
    """
    A wrapper around multiple CoreDecoderBlocks.
    Each stage will upsample and merge skip features with attention.
    """
    def __init__(
        self,
        depths: List[int],
        dims: List[int],
        norm="batch",
        activation="relu",
    ):
        """
        Args:
            depths: same length as dims (reversed order).
            dims: e.g. [64, 128, 256, 512] if used in reverse for building blocks.
        """
        super().__init__()
        assert len(depths) == len(dims), f"depths and dims must have the same length. dims={dims}, depths={depths}"

        self.stages = nn.ModuleList()
        self.num_stages = len(dims)
        # We'll build them in reverse to match the skip connections
        for i in reversed(range(len(dims))):
            in_ch = dims[i]
            out_ch = dims[i - 1] if i - 1 >= 0 else dims[0]
            stage = CoreDecoderBlock(
                depth=depths[i],
                in_channels=in_ch,
                out_channels=out_ch,
                norm=norm,
                activation=activation,
            )
            self.stages.append(stage)

    def forward(self, x, skip_list):
        """
        Args:
            x: bottom_feats from the encoder/bridge
            skip_list: list of skip features in the same order 
                       they were collected by the encoder
        Returns:
            x: the final upsampled feature map
        """
        skip_list = skip_list[::-1]
        for i in range(self.num_stages):
            skip = skip_list[i]
            x = self.stages[i](x, skip)
        
        return x












class phisat2net_geoaware_downstream(nn.Module):
    def __init__(
        self,
        *,
        input_dim=3,
        output_dim=None,
        depths=None,
        dims=None,
        img_size=128,
        dropout=True,      # optional, not really used by the Core blocks
        activation="gelu",
        task='classification'
    ):
        super().__init__()

        # Basic model parameters
        self.input_dim = input_dim
        self.output_dim = input_dim if output_dim is None else output_dim
        self.depths = depths
        self.dims = dims
        self.img_size = img_size
        self.task = task

        self.dropout = dropout

        if dropout:
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

        # ---------------------
        # 1) Stem
        # ---------------------
        # We will go from input_dim -> dims[0]
        self.stem = CoreCNNBlock(
            in_channels=self.input_dim,
            out_channels=self.dims[0],
            norm="batch",
            activation=activation,
            residual=False
        )

        # ---------------------
        # 2) Encoder
        # ---------------------
        # We'll create an encoder from dims[0] -> dims[1] -> dims[2] -> ...
        # with the specified depths
        self.encoder = FoundationEncoder(
            input_dim=self.dims[0],
            depths=self.depths,
            dims=self.dims,  # we already consumed dims[0] in the stem
            norm="batch",
            activation=activation,
        )

        # ---------------------
        # Bridge
        # ---------------------
        # Typically we do a small bridging block at the bottom
        # to let the model "breathe" in the bottleneck:
        self.bridge = CoreCNNBlock(
            in_channels=self.dims[-1],
            out_channels=self.dims[-1],
            norm="batch",
            activation=activation,
            residual=True
        )

        # ---------------------
        # 3) Decoder
        # ---------------------
        self.decoder = FoundationDecoder(
            depths=self.depths,
            dims=self.dims,
            norm="batch",
            activation=activation,
        )

        # -------------------------------
        # Heads (choose by task)
        # -------------------------------
        if self.task == 'classification':
            self.head_clas = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.dims[-1], self.output_dim),
            )

        elif self.task == 'segmentation':
            self.head_segm = nn.Sequential(
                CoreCNNBlock(self.dims[0], self.dims[0], norm="batch", activation=activation),
                nn.Conv2d(self.dims[0], self.output_dim, kernel_size=1, padding=0),
            )

        else:
            raise ValueError(f"Invalid task: {self.task}")
    
    def forward(self, x):
        x = self.stem(x)
        x, skips = self.encoder(x)
        if self.task == 'classification':
            x = self.head_clas(x)
        elif self.task == 'segmentation':
            x = self.decoder(x, skips)
            x = self.head_segm(x)
        else:
            raise ValueError(f"Invalid task: {self.task}")
        return x
