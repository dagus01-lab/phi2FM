"""
UNet_Myriad2 Downstream Model
Adapted from enhanced_distillation_production.py for downstream burned area classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block using BatchNorm instead of LayerNorm for better performance."""
    
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = input + x
        return x


class ConvBlock(nn.Module):
    """ConvNeXt-style block with BatchNorm for improved performance."""
    
    def __init__(self, in_channels: int, out_channels: int, drop_path: float = 0.0):
        super().__init__()
        
        if in_channels != out_channels:
            self.channel_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.channel_proj = nn.Identity()
        
        self.convnext_block = ConvNeXtBlock(out_channels, drop_path=drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_proj(x)
        x = self.convnext_block(x)
        return x


class UNet_Myriad2_Downstream(nn.Module):
    """
    UNet_Myriad2 adapted for downstream classification tasks.
    Supports loading pretrained reconstruction weights and adding a classification head.
    """
    
    def __init__(self, 
                 pretrained_path: str = None,
                 task: str = 'classification',
                 input_dim: int = 8, 
                 output_dim: int = 4, 
                 base_filters: int = 16,
                 depth: int = 3,
                 channel_multipliers: list = None,
                 freeze_body: bool = False,
                 img_size: int = 256):
        """
        Initialize UNet_Myriad2 for downstream tasks.
        
        Args:
            pretrained_path (str): Path to pretrained model checkpoint
            task (str): Either 'classification' or 'segmentation'
            input_dim (int): Number of input channels (8 for PhiSat)
            output_dim (int): Number of output classes
            base_filters (int): Base number of filters
            depth (int): Number of encoder/decoder levels
            channel_multipliers (list): Channel multipliers for each level
            freeze_body (bool): If True, freeze encoder/decoder weights
            img_size (int): Input image size
        """
        super().__init__()
        
        self.task = task.lower()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.img_size = img_size
        
        # Set default channel multipliers if not provided
        if channel_multipliers is None:
            channel_multipliers = [2**i for i in range(depth + 1)]
        
        if len(channel_multipliers) != depth + 1:
            raise ValueError(f'channel_multipliers length ({len(channel_multipliers)}) must equal depth + 1 ({depth + 1})')
        
        self.channel_multipliers = channel_multipliers
        self.channels = [base_filters * mult for mult in channel_multipliers]
        
        # Build encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        # First encoder block
        self.encoders.append(ConvBlock(input_dim, self.channels[0]))
        
        # Remaining encoder blocks
        for i in range(depth - 1):
            self.pools.append(nn.MaxPool2d(2))
            self.encoders.append(ConvBlock(self.channels[i], self.channels[i + 1]))
        
        self.pools.append(nn.MaxPool2d(2))
        
        # Bottleneck
        self.bottleneck = ConvBlock(self.channels[depth - 1], self.channels[depth])
        
        # Build decoder
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i in range(depth):
            up_in_channels = self.channels[depth - i]
            up_out_channels = self.channels[depth - i]
            self.upsamplers.append(
                nn.ConvTranspose2d(up_in_channels, up_out_channels, kernel_size=2, stride=2)
            )
            
            dec_in_channels = self.channels[depth - i] + self.channels[depth - i - 1]
            dec_out_channels = self.channels[depth - i - 1]
            self.decoders.append(ConvBlock(dec_in_channels, dec_out_channels))
        
        # Task-specific head
        if self.task == 'classification':
            # Classification: Global pooling + MLP classifier
            # Use features from the bottleneck (deepest layer) instead of decoder output
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.channels[-1], self.channels[-1] // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(self.channels[-1] // 2, output_dim)
            )
        else:
            # Segmentation: Pixel-wise classification
            self.classifier = nn.Conv2d(self.channels[0], output_dim, kernel_size=1)
        
        # Load pretrained weights if provided
        if pretrained_path is not None:
            self._load_pretrained_weights(pretrained_path, freeze_body)
    
    def _load_pretrained_weights(self, pretrained_path: str, freeze_body: bool):
        """Load pretrained weights from reconstruction model."""
        try:
            print(f"Loading pretrained weights from: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load encoder, decoder, and bottleneck weights (excluding final_conv)
            model_dict = self.state_dict()
            pretrained_dict = {}
            
            for k, v in state_dict.items():
                # Skip final_conv layer (reconstruction head)
                if 'final_conv' in k:
                    continue
                # Skip classifier if present
                if 'classifier' in k:
                    continue
                    
                # Load encoder, decoder, bottleneck weights
                if k in model_dict and model_dict[k].shape == v.shape:
                    pretrained_dict[k] = v
                    print(f"  Loaded: {k}")
                else:
                    print(f"  Skipped: {k} (shape mismatch or not found)")
            
            # Update model with pretrained weights
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            
            print(f"Loaded {len(pretrained_dict)} pretrained layers")
            
            # Freeze encoder/decoder if requested
            if freeze_body:
                print("Freezing encoder and decoder weights")
                for name, param in self.named_parameters():
                    if 'classifier' not in name:
                        param.requires_grad = False
                        
        except Exception as e:
            warnings.warn(f"Failed to load pretrained weights: {e}")
            print("Continuing with random initialization")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, num_classes] for classification or [B, num_classes, H, W] for segmentation
        """
        # Encoder path
        encoder_outputs = []
        current = x
        
        for i in range(self.depth):
            current = self.encoders[i](current)
            encoder_outputs.append(current)
            if i < self.depth - 1:
                current = self.pools[i](current)
        
        # Bottleneck
        current = self.pools[-1](current)
        bottleneck_features = self.bottleneck(current)
        
        # Task-specific processing
        if self.task == 'classification':
            # Classification: Use bottleneck features directly
            pooled = self.global_pool(bottleneck_features)  # [B, C, 1, 1]
            out = self.classifier(pooled)  # [B, num_classes]
        else:
            # Segmentation: Decoder path
            current = bottleneck_features
            for i in range(self.depth):
                current = self.upsamplers[i](current)
                skip_connection = encoder_outputs[self.depth - 1 - i]
                current = torch.cat([current, skip_connection], dim=1)
                current = self.decoders[i](current)
            
            # Pixel-wise classification
            out = self.classifier(current)  # [B, num_classes, H, W]
        
        return out
    
    def get_model_info(self) -> dict:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Get classifier parameters
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        return {
            'task': self.task,
            'depth': self.depth,
            'channels_per_level': self.channels,
            'channel_multipliers': self.channel_multipliers,
            'uses_decoder': self.task == 'segmentation',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'classifier_parameters': classifier_params,
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.parameters()) / 1024**2
        }
