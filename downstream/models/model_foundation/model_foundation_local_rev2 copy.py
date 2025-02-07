import torch
import torch.nn as nn
from .blocks import CNNBlock, ScaleSkip2D, ChannelGLU
from torchvision import models
import torch.nn.functional as F

# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from model_CoreCNN import CoreCNNBlock, CoreEncoderBlock, CoreDecoderBlock


class FoundationEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim=3,
        depths=None,
        dims=None,
        img_size=64,
        latent_dim=512,
        activation=nn.ReLU6(),
        norm='batch',
    ):
        super().__init__()

        self.depths = depths
        self.dims = dims
        self.input_dim = input_dim
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.steps = 1
        self.sizes = [img_size]
        self.activation = activation

        for i in range(len(self.depths) - 1):
            half = self.sizes[-1] // 2
            self.sizes.append(half)
            self.steps += 1

        self.linear_dim = int(((img_size // (2 ** (self.steps - 1))) ** 2) * self.dims[-1])
        print(f'linear_dim: {self.linear_dim}, img_size: {img_size}, steps: {self.steps}, self.dims[-1]: {self.dims[-1]}')
        assert len(self.depths) == self.steps, f"Invalid depths, steps: {self.steps} -- depths: {len(self.depths)}"
        assert len(self.dims) == self.steps, f"Invalid dims, steps: {self.steps} -- dims: {len(self.dims)}"
        assert self.depths is not None, "Invalid depths"
        assert self.dims is not None, "Invalid dims"
        assert self.steps == len(self.dims), "Invalid dims"

        self.downsample = nn.ModuleList()
        for i in range(self.steps - 1):
            self.downsample.append(nn.Sequential(
                nn.Conv2d(self.dims[i], self.dims[i + 1], 1, padding=0),
                nn.MaxPool2d(2, stride=2),
            ))

        self.block_scalers = nn.ModuleList()
        for i in range(self.steps):
            self.block_scalers.append(ScaleSkip2D(self.dims[i]))

        self.blocks_down = nn.ModuleList()

        if norm == 'batch':
            for i in range(self.steps):
                self.blocks_down.append(nn.ModuleList())
                for _ in range(self.depths[i]):
                    self.blocks_down[i].append(
                        CNNBlock(self.dims[i], chw=None, activation=self.activation)
                    )
        else:
            for i in range(self.steps):
                self.blocks_down.append(nn.ModuleList())
                for _ in range(self.depths[i]):
                    self.blocks_down[i].append(
                        CNNBlock(self.dims[i], chw=[self.dims[i], self.sizes[i], self.sizes[i]], activation=self.activation)
                    )

        self.bridge = nn.Sequential(
            nn.Conv2d(current_channels, bridge_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(bridge_channels),
            nn.ReLU(inplace=True),
            # Add more layers as needed
        )

    def forward(self, x):
        skips = []

        for i in range(self.steps):
            pre_block = x
            for j in range(self.depths[i]):
                block = self.blocks_down[i][j]
                x = block(x)

            if len(self.blocks_down[i]) > 1:
                x = self.block_scalers[i](x, pre_block)

            skips.append(x)

            if i < self.steps - 1:
                x = self.downsample[i](x)

        # Remove linear encoding
        # embeddings_cnn = self.prelinear_norm(x)
        # flat = embeddings_cnn.reshape(-1, self.linear_dim)
        # embeddings = self.linear_encode(flat)
        
        # Optionally, apply a bridge (additional convolutional layers)
        if hasattr(self, 'bridge'):
            x = self.bridge(x)

        # Return the 4D tensor instead of a 2D embedding
        return x, skips



class FoundationDecoder(nn.Module):
    def __init__(
        self,
        *,
        depths=None,
        dims=None,
        img_size=64,
        latent_dim=512,
        dropout=None,
        activation=nn.ReLU6(),
    ):
        super().__init__()
        self.depths = depths
        self.dims = dims
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.steps = 1
        self.sizes = [img_size]
        self.dropout = dropout
        self.activation = activation

        for i in range(len(self.depths) - 1):
            half = self.sizes[-1] // 2
            self.sizes.append(half)
            self.steps += 1

        self.sizes = self.sizes[::-1]
        self.linear_dim = int(((img_size // (2 ** (self.steps - 1))) ** 2) * self.dims[0])

        if self.dropout is None:
            self.dropout = [None] * self.steps
        elif isinstance(self.dropout, (int, float)):
            self.dropout = [self.dropout] * self.steps

        assert len(self.depths) == self.steps, "Invalid depths"
        assert len(self.dims) == self.steps, "Invalid dims"
        assert len(self.dropout) == self.steps, "Invalid dropout"
        assert self.depths is not None, "Invalid depths"
        assert self.dims is not None, "Invalid dims"
        assert self.dropout is not None, "Invalid dropout"

        self.linear_decode = nn.Linear(self.latent_dim, self.linear_dim)

        self.latent_norm = nn.LayerNorm([self.dims[0], self.img_size // (2 ** (self.steps - 1)), self.img_size // (2 ** (self.steps - 1))])
        self.prehead_norm = nn.LayerNorm([self.dims[-1], self.sizes[-1], self.sizes[-1]])

        self.skip_scalers = nn.ModuleList()
        self.block_scalers = nn.ModuleList()
        for i in range(self.steps):
            self.skip_scalers.append(ScaleSkip2D(self.dims[i], drop_y=self.dropout[i], signal_to_noise=(0.1, 0.9)))
            self.block_scalers.append(ScaleSkip2D(self.dims[i]))

        self.blocks_up = nn.ModuleList()
        for i in range(self.steps):
            self.blocks_up.append(nn.ModuleList())
            for _ in range(self.depths[i]):
                self.blocks_up[i].append(
                    CNNBlock(self.dims[i], chw=[self.dims[i], self.sizes[i], self.sizes[i]], activation=self.activation)
                )

        self.upsamplers = nn.ModuleList()
        for i in range(self.steps - 1):
            self.upsamplers.append(nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(self.dims[i], self.dims[i + 1], 3, padding=1, bias=False, padding_mode='replicate'),
                nn.LayerNorm([self.dims[i + 1], self.sizes[i + 1], self.sizes[i + 1]]),
                self.activation,
            ))

    def forward(self, x, skips):
        x = self.linear_decode(x)
        x = x.reshape(-1, self.dims[0], self.img_size // (2 ** (self.steps - 1)), self.img_size // (2 ** (self.steps - 1)))
        x = self.latent_norm(x)

        for i in range(self.steps):
            skip_x = skips[-(i + 1)]
            x = self.skip_scalers[i](x, skip_x)

            pre_block = x
            for block in self.blocks_up[i]:
                x = block(x)

            if len(self.blocks_up[i]) > 1:
                x = self.block_scalers[i](x, pre_block)

            if i < self.steps - 1:
                x = self.upsamplers[i](x)

        x = self.prehead_norm(x)

        return x


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
                 use_rgb=True):
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
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = requires_grad
        self.vgg = vgg

        # Save normalization parameters (ImageNet)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

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



class FoundationModel4Task(nn.Module):
    def __init__(self, *,
        input_dim=10,
        output_dim=1,
        depths=None,
        dims=None,
        activation="relu",
        norm="batch",
        padding="same",
    ):
        super(FoundationModel4Task, self).__init__()

        self.depths = [3, 3, 9, 3] if depths is None else depths
        self.dims = [96, 192, 384, 768] if dims is None else dims
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.activation = activation
        self.norm = norm
        self.padding = padding

        self.dims = [v // 2 for v in self.dims]

        assert len(self.depths) == len(self.dims), "depths and dims must have the same length."

        # Stem to initial feature representation
        self.stem = nn.Sequential(
            CoreCNNBlock(self.input_dim, self.dims[0], norm=self.norm, activation=self.activation, padding=self.padding),
        )

        # Encoder - outputs embeddings, skips
        self.encoder_blocks = []
        for i in range(len(self.depths)):
            encoder_block = CoreEncoderBlock(
                self.depths[i],
                self.dims[i - 1] if i > 0 else self.dims[0],
                self.dims[i],
                norm=self.norm,
                activation=self.activation,
                padding=self.padding,
            )
            self.encoder_blocks.append(encoder_block)

        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)

        # Bridge - bottleneck
        self.bridge = nn.Sequential(
            CoreCNNBlock(self.dims[-1], self.dims[-1], norm=self.norm, activation=self.activation, padding=self.padding),
        )

        # Decoder - reconstruct spatial features
        self.decoder_blocks = []
        for i in reversed(range(len(self.encoder_blocks))):
            decoder_block = CoreDecoderBlock(
                self.depths[i],
                self.dims[i],
                self.dims[i - 1] if i > 0 else self.dims[0],
                norm=self.norm,
                activation=self.activation,
                padding=self.padding,
            )
            self.decoder_blocks.append(decoder_block)

        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

        # self.head = nn.Sequential(
        #     CoreCNNBlock(self.dims[0], self.dims[0], norm=self.norm, activation=self.activation, padding=self.padding),
        #     nn.Conv2d(self.dims[0], self.output_dim, kernel_size=1, padding=0),
        # )

        # Heads:
        # 1) Reconstruction head (already present)
        self.head_recon = CoreCNNBlock(self.dims[0], self.output_dim, norm=self.norm, activation='sigmoid', padding=self.padding)

        # 2) Climate zone segmentation head
        #    We want 31 classes over the spatial dimension.
        #    We'll produce (31, 256, 256) logits. The user can do argmax.
        self.head_seg = nn.Sequential(
            nn.Conv2d(self.dims[0], 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 31, kernel_size=3, padding=1)
        )

        # 3) Geolocation head: from global embeddings
        #    Assume `embeddings` is of shape (B, latent_dim).
        #    We predict (3,): latitude, sin(lon), cos(lon)
        self.head_geo = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

        # 4) Zoom level head: a single scalar
        self.head_zoom = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward_body(self, x):
        skip_connections = []

        x = self.stem(x)
        for block in self.encoder_blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        x = self.bridge(x)

        for block in self.decoder_blocks:
            skip = skip_connections.pop()
            x = block(x, skip)
        return x

    def forward(self, x):

        x = self.forward_body(x)
        x = self.head(x)

        return x




class FoundationModel4Task(nn.Module):
    def __init__(
        self,
        *,
        input_dim=3,
        output_dim=None,    # For reconstruction output channels, typically 3 for RGB
        depths=None,
        dims=None,
        img_size=256,       # Adjusted for your input size
        latent_dim=512,
        dropout=None,
        activation=nn.GELU(),
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim if output_dim is None else output_dim
        self.depths = depths
        self.dims = dims
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.activation = activation

        # Stem to initial feature representation
        self.stem = CNNBlock(
            input_dim,
            dims[0],
            # chw=[input_dim, img_size, img_size],
            chw=None,
            activation=self.activation,
        )
        
        # Encoder - outputs embeddings, embeddings_cnn, skips, and possibly intermediate predictions
        self.encoder = FoundationEncoder(
            input_dim=dims[0],
            depths=depths,
            dims=dims,
            img_size=img_size,
            latent_dim=latent_dim,
            activation=ChannelGLU(),
        )

        # Decoder - reconstruct spatial features
        self.decoder = FoundationDecoder(
            depths=depths[::-1],
            dims=dims[::-1],
            img_size=img_size,
            latent_dim=latent_dim,
            dropout=dropout,
            activation=ChannelGLU(),
        )

        # Heads:
        # 1) Reconstruction head (already present)
        self.head_recon = CNNBlock(
            self.dims[0],
            self.output_dim,  # 3 channels for RGB reconstruction
            chw=[self.output_dim, self.img_size, self.img_size],
            activation=self.activation,
            activation_out=nn.Sigmoid(),
        )

        # 2) Climate zone segmentation head
        #    We want 31 classes over the spatial dimension.
        #    We'll produce (31, 256, 256) logits. The user can do argmax.
        self.head_seg = nn.Sequential(
            nn.Conv2d(self.dims[0], 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 31, kernel_size=3, padding=1)
        )

        # 3) Geolocation head: from global embeddings
        #    Assume `embeddings` is of shape (B, latent_dim).
        #    We predict (3,): latitude, sin(lon), cos(lon)
        self.head_geo = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

        # 4) Zoom level head: a single scalar
        self.head_zoom = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Optional: A global pooling if needed.
        # If embeddings are already (B, latent_dim), no pooling needed.
        # If embeddings_cnn is a spatial map, we might do global average pooling.

    def forward(self, x):
        # x: (B, 3, 256, 256)
        x = self.stem(x)  # (B, dims[0], H, W)
        embeddings, skips = self.encoder(x)
        
        # embeddings is expected to be (B, latent_dim)
        # embeddings_cnn might be a spatial map (depends on encoder design)
        
        decoded = self.decoder(embeddings, skips)  # (B, dims[0], 256, 256)

        # Reconstruction
        reconstruction = self.head_recon(decoded)  # (B, 3, 256, 256)

        # Climate zone segmentation
        climate_logits = self.head_seg(decoded)     # (B, 31, 256, 256)

        # Geolocation and Zoom level predictions from embeddings
        geo_pred = self.head_geo(embeddings)        # (B, 3)
        zoom_pred = self.head_zoom(embeddings)      # (B, 1)

        # Return all tasks
        return {
            "coords": geo_pred,                     # (B, 4)
            "climate": climate_logits,              # (B, 31, 256, 256)
            "zoom_factor": zoom_pred,               # (B, 1)
            "reconstruction": reconstruction,       # (B, 3, 256, 256)
            # "embeddings": embeddings,
            # "embeddings_cnn": embeddings_cnn,
            # "decoded_features": decoded,
            # "intermediate_predictions": predictions
        }







class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        # Initialize log variances as learnable parameters
        self.log_sigma_recon = nn.Parameter(torch.zeros(1)) # For reconstruction
        self.log_sigma_perc = nn.Parameter(torch.zeros(1)) # For perceptual loss
        self.log_sigma_seg = nn.Parameter(torch.zeros(1)) # For climate segmentation
        self.log_sigma_geo = nn.Parameter(torch.zeros(1)) # For geolocation
        self.log_sigma_zoom = nn.Parameter(torch.zeros(1)) # For zoom level
        self.log_sigma_tv = nn.Parameter(torch.zeros(1))  # For TV loss
        
        # Define loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.perceptual_loss = PerceptualLoss()

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
        
        # Reconstruction Loss (Pixel-wise and Perceptual)
        loss_recon = F.smooth_l1_loss(output["reconstruction"], labels["reconstruction"])
        loss_perceptual = self.perceptual_loss(output["reconstruction"], labels["reconstruction"])
        
        # Climate Segmentation Loss (Cross-entropy and Total Variation)
        loss_climate = self.ce_loss(output["climate"], labels["climate"])
        loss_tv = self.total_variation_loss(output["climate"])
        
        # Geolocation Loss
        loss_geo = self.mse_loss(output["coords"], labels["coords"])
        
        # Zoom Level Loss
        loss_zoom = self.mse_loss(output["zoom_factor"], labels["zoom_factor"])
        
        # Combine all losses with uncertainty-based weighting
        # Using the formulation: (1/(2*sigma^2)) * loss + log(sigma)
        loss = (
            (torch.exp(-self.log_sigma_recon) * loss_recon) + self.log_sigma_recon
            + (torch.exp(-self.log_sigma_perc) * loss_perceptual) + self.log_sigma_perc
            + (torch.exp(-self.log_sigma_seg) * loss_climate) + self.log_sigma_seg
            + (torch.exp(-self.log_sigma_tv) * loss_tv) + self.log_sigma_tv
            + (torch.exp(-self.log_sigma_geo) * loss_geo) + self.log_sigma_geo
            + (torch.exp(-self.log_sigma_zoom) * loss_zoom) + self.log_sigma_zoom
        )
        
        log_loss = {
            'total_loss': loss.item(),
            'loss_components': {
                'reconstruction': loss_recon.item(),
                'perceptual': loss_perceptual.item(),
                'climate_segmentation': loss_climate.item(),
                'total_variation': loss_tv.item(),
                'geolocation': loss_geo.item(),
                'zoom_level': loss_zoom.item(),
            },
            'log_sigmas': {
                'log_sigma_recon': self.log_sigma_recon.item(),
                'log_sigma_perc': self.log_sigma_perc.item(),
                'log_sigma_seg': self.log_sigma_seg.item(),
                'log_sigma_tv': self.log_sigma_tv.item(),
                'log_sigma_geo': self.log_sigma_geo.item(),
                'log_sigma_zoom': self.log_sigma_zoom.item(),
            }
        }

        
        return loss, log_loss


def FoundationModel4Task_CoreUnet_nano(**kwargs):
    """
    Total params: 16,400,685
    Trainable params: 16,400,685
    Non-trainable params: 0
    Total mult-adds (G): 50.95
    =========================================================================================================
    Input size (MB): 5.24
    Forward/backward pass size (MB): 3388.57
    Params size (MB): 65.60
    Estimated Total Size (MB): 3459.42
    
    *2 because full unet
    """
    # model = FoundationModel4Task(depths=[2, 2, 8, 2], dims=[80*2, 160*2, 320*2, 640*2], **kwargs)
    model = FoundationModel4Task(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def FoundationModel4Task_CoreUnet_nano_GeoAware(**kwargs):
    """
    Total params: 16,400,685
    Trainable params: 16,400,685
    Non-trainable params: 0
    Total mult-adds (G): 50.95
    =========================================================================================================
    Input size (MB): 5.24
    Forward/backward pass size (MB): 3388.57
    Params size (MB): 65.60
    Estimated Total Size (MB): 3459.42
    
    *2 because full unet
    """
    # model = FoundationModel4Task(depths=[2, 2, 8, 2], dims=[80*2, 160*2, 320*2, 640*2], **kwargs)
    model = FoundationModel4TaskGeoAware(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model


