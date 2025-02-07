import torch
import torch.nn as nn
from .blocks import CNNBlock, ScaleSkip2D, ChannelGLU
from torchvision import models
import torch.nn.functional as F


class FoundationEncoderFlat(nn.Module):
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

            self.prelinear_norm = nn.BatchNorm2d(self.dims[-1])
            self.linear_encode = nn.Sequential(
                self.activation,
                nn.Linear(self.linear_dim, self.latent_dim),
                nn.BatchNorm1d(self.latent_dim),
            )

        else:
            for i in range(self.steps):
                self.blocks_down.append(nn.ModuleList())
                for _ in range(self.depths[i]):
                    self.blocks_down[i].append(
                        CNNBlock(self.dims[i], chw=[self.dims[i], self.sizes[i], self.sizes[i]], activation=self.activation)
                    )

            self.prelinear_norm = nn.LayerNorm([self.dims[-1], self.sizes[-1], self.sizes[-1]])
            self.linear_encode = nn.Sequential(
                self.activation,
                nn.Linear(self.linear_dim, self.latent_dim),
                nn.LayerNorm(self.latent_dim),
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

        embeddings_cnn = self.prelinear_norm(x)
        flat = embeddings_cnn.reshape(-1, self.linear_dim)
        embeddings = self.linear_encode(flat)
        
        return (
            embeddings,
            skips,
        )

class FoundationDecoderFlat(nn.Module):
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
