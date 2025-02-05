import torch
import torch.nn as nn

from tabulate import tabulate

from .uniphi_blocks import CNNBlock, ChannelGLU, ScaleSkip2D
from .util_tools import make_bilinear_upsample

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
