import torch
import torch.nn as nn

from typing import List # need because using python 3.8

from pretrain.models.geoaware_blocks import CoreCNNBlock, CoreAttentionBlock
from pretrain.models.util_tools import make_bilinear_upsample, get_activation

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
        activation="gelu",
        fixed_task=None,
    ):
        """
        A U-Net-like model with heads for:
          - climate zone classification
          - geolocation
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
        self.activation = activation

        # ---------------------
        # 1) Stem
        # ---------------------
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
        self.encoder = FoundationEncoder(
            input_dim=self.dims[0],
            depths=self.depths,
            dims=self.dims,
            norm="group",
            activation=activation,
        )

        # ---------------------
        # Bridge
        # ---------------------
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
            self.decoder = FoundationDecoder(
                depths=self.depths,
                dims=self.dims,
                norm="group",
                activation=activation,
            )
        else:
            self.decoder = nn.Identity()


        # ---------------------
        # 4) Heads
        # ---------------------
        # 4.1 Reconstruction head (output_dim channels)
        if self.fixed_task is None or self.fixed_task == "reconstruction":
            self.head_recon = nn.Conv2d(in_channels=self.dims[0], out_channels=self.output_dim, kernel_size=1)

        else:
            self.head_recon = nn.Identity()

        # 4.2 Climate zone segmentation head (31 classes)
        if self.fixed_task is None or self.fixed_task == "climate":
            self.head_seg = nn.Linear(self.dims[-1], 31)
        else:
            self.head_seg = nn.Identity()

        # 4.3 Geolocation head (4 values)
        if self.fixed_task is None or self.fixed_task == "coords":
            self.head_geo = nn.Linear(self.dims[-1], 4)
        else:
            self.head_geo = nn.Identity()

    def pool_feats(self, x):
        """
        Pools a 4D feature map (B, C, H, W) into a 1D vector (B, C)
        using global average pooling.
        """
        return torch.flatten(nn.AdaptiveAvgPool2d(1)(x), 1)

    def forward(self, x):
        """
        Args:
            x: (B, input_dim, H, W) e.g. (B, 8, 224, 224)

        Returns:
            dict with keys: "coords", "climate", "reconstruction"
        """
        
        # ---------------------
        # 1) Stem
        # ---------------------
        x_stem = self.stem(x)  # (B, dims[0], H, W)


        # ---------------------
        # 2) Encoder
        # ---------------------
        bottom, skips = self.encoder(x_stem) # (B, dims[-1], H//(2^num_stages), W//(2^num_stages))

        # ---------------------
        # 3) Bridge
        # ---------------------
        bottom_feats = self.bridge(bottom) # (B, dims[-1], H//(2^num_stages), W//(2^num_stages))

        # ---------------------
        # 4) Decoder
        # ---------------------
        if self.fixed_task is None or self.fixed_task == "reconstruction":
            decoded_feats = self.decoder(bottom_feats, skips)  # (B, dims[0], H, W)
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
            "climate": climate_logits,         # (B, 31)
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
        super().__init__()
        assert len(depths) == len(dims), f"depths and dims must have the same length. dims={dims}, depths={depths}"

        self.stages = nn.ModuleList()
        self.num_stages = len(dims)
        # Build them in reverse to match the skip connections
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
        skip_list = skip_list[::-1]
        for i in range(self.num_stages):
            skip = skip_list[i]
            x = self.stages[i](x, skip)
        
        return x

