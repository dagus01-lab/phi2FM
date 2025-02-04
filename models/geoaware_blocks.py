import torch
import torch.nn as nn
from utils.training_utils import get_activation, get_normalization, SE_Block


class CoreCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, norm="batch", activation="relu", residual=True):
        super(CoreCNNBlock, self).__init__()
        
        self.activation = get_activation(activation)
        self.residual = residual
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.squeeze = SE_Block(self.out_channels)

        self.match_channels = nn.Identity()
        if in_channels != out_channels:
            self.match_channels = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),  # padding=0 for 1x1 conv
                get_normalization(norm, out_channels),
            )

        # Conv1: 1x1 convolution never needs padding
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0)
        self.norm1 = get_normalization(norm, self.out_channels)

        # Conv2: Depthwise 3x3 conv needs padding=1 to maintain size
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, groups=self.out_channels)
        self.norm2 = get_normalization(norm, self.out_channels)
        
        # Conv3: Regular 3x3 conv needs padding=1 to maintain size
        self.conv3 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1,  groups=1)
        self.norm3 = get_normalization(norm, self.out_channels)

    def forward(self, x):
        # Forward remains unchanged
        identity = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        x = self.norm3(self.conv3(x))

        x = x * self.squeeze(x)

        if self.residual:
            x = x + self.match_channels(identity)

        x = self.activation(x)
        return x


class CoreAttentionBlock(nn.Module):
    def __init__(self,
        lower_channels,
        higher_channels, *,
        norm="batch",
        activation="relu",
    ):
        super(CoreAttentionBlock, self).__init__()

        self.lower_channels = lower_channels
        self.higher_channels = higher_channels
        self.activation = get_activation(activation)
        self.norm = norm
        self.expansion = 4
        self.reduction = 4

        if self.lower_channels != self.higher_channels:
            self.match = nn.Sequential(
                nn.Conv2d(self.higher_channels, self.lower_channels, kernel_size=1, padding=0, bias=False),
                get_normalization(self.norm, self.lower_channels),
            )

        self.compress = nn.Conv2d(self.lower_channels, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.attn_c_pool = nn.AdaptiveAvgPool2d(self.reduction)
        self.attn_c_reduction = nn.Linear(self.lower_channels * (self.reduction ** 2), self.lower_channels * self.expansion)
        self.attn_c_extention = nn.Linear(self.lower_channels * self.expansion, self.lower_channels)

    def forward(self, x, skip):
        if x.size(1) != skip.size(1):
            x = self.match(x)
        x = x + skip
        x = self.activation(x)

        attn_spatial = self.compress(x)
        attn_spatial = self.sigmoid(attn_spatial)

        attn_channel = self.attn_c_pool(x)
        attn_channel = attn_channel.reshape(attn_channel.size(0), -1)
        attn_channel = self.attn_c_reduction(attn_channel)
        attn_channel = self.activation(attn_channel)
        attn_channel = self.attn_c_extention(attn_channel)
        attn_channel = attn_channel.reshape(x.size(0), x.size(1), 1, 1)
        attn_channel = self.sigmoid(attn_channel)

        return attn_spatial, attn_channel



