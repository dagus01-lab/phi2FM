# flake8: noqa: E501
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChannelGLU(nn.Module):
    def __init__(self):
        super(ChannelGLU, self).__init__()
    
    def forward(self, x):
        if x.dim() != 4:
            return F.gelu(x)

        split = x.size(1) // 2
        split_left = x[:, split:, :, :]
        split_right = x[:, :split, :, :]

        return torch.cat([
            torch.sigmoid(split_right) * split_left,
            F.gelu(split_right),
        ], dim=1)




class ScaleSkip2D(nn.Module):
    """
    Learnable channel-wise scale and bias for skip connections.
    
    Parameters
    ----------
    channels : int
        Number of channels in the input

    drop_y : float
        Probability of dropping a channel in the skip connection.
        Drops are replaces with Gaussian noise.

    signal_to_noise : tuple or None
        Range of signal to noise ratios to use for the dropped channels. 0.0 is pure noise, 1.0 is pure signal.
        The amount of signal is randomly sampled from this range for each channel.
        If None, no signal is added to the dropped channels.
        default: (0.1, 0.9)

    size : float
        Standard deviation of the normal distribution to sample inital values from
        default: 0.01
    """
    def __init__(self, channels, drop_y=None, signal_to_noise=(0.1, 0.9), size=0.01):
        super(ScaleSkip2D, self).__init__()
        self.channels = channels
        self.drop_y = drop_y
        self.size = size

        # Learnable scale and bias
        self.x_skipscale = nn.Parameter(torch.ones(1, self.channels, 1, 1))
        self.y_skipscale = nn.Parameter(torch.ones(1, self.channels, 1, 1))
        self.y_skipbias = nn.Parameter(torch.zeros(1, self.channels, 1, 1))
        self.x_skipbias = nn.Parameter(torch.zeros(1, self.channels, 1, 1))

        if self.drop_y is not None and self.drop_y > 0.0:
            # self.drop_y = GaussianDropout2d(self.drop_y, signal_to_noise=signal_to_noise)
            self.drop_y = nn.Dropout2d(p=self.drop_y)
        else:
            self.drop_y = None

        self.set_weights()
        # while torch.any(self.x_skipscale == 0) or torch.any(self.y_skipscale == 0) or torch.any(self.y_skipbias == 0) or torch.any(self.x_skipbias == 0):
        #     self.set_weights()

    def set_weights(self):
        nn.init.trunc_normal_(self.x_skipscale, 1.0, self.size)
        nn.init.trunc_normal_(self.y_skipscale, 1.0, self.size)
        nn.init.trunc_normal_(self.y_skipbias, 0.0, self.size)
        nn.init.trunc_normal_(self.x_skipbias, 0.0, self.size)

    def forward(self, x, y):
        x = (x * self.x_skipscale) + self.x_skipbias
        y = (y * self.y_skipscale) + self.y_skipbias

        if self.drop_y is not None:
            y = self.drop_y(y)

        return x + y


class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.reduction = reduction
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, max(1, channels // self.reduction), bias=False),
            nn.GELU(),
            nn.Linear(max(1, channels // self.reduction), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)

        if not torch.isfinite(y).all():
            print("Found NaNs or Infs in squeeze output")

        y = self.excitation(y).view(bs, c, 1, 1)

        return x * y.expand_as(x)






class CNNBlock(nn.Module):
    """
    This is a standard CNN block with a 1x1 convolutional matcher for the skip connection.
    It adds a learnable scale and bias to the skip connection.

    Parameters
    ----------
    channels_in : int
        Number of channels in the input

    channels_out : int or None
        Number of channels in the output. If None, the number of channels is unchanged.
        default: None

    group_size : int
        Number of groups for the 3x3 convolution.
        default: 1

    activation : torch.nn.Module
        Activation function to use after the first convolution.
        default: torch.nn.GELU()

    activation_out : torch.nn.Module or None
        Activation function to use after the last convolution. If None, the same activation as the first convolution is used.
        default: None

    chw : tuple or None
        Height and width of the input. If None, batch norm is used instead of layer norm.
        default: None
    """
    def __init__(
        self,
        channels_in,
        channels_out=None,
        chw=None,
        group_size=1,
        activation=nn.GELU(),
        activation_out=nn.Identity(),
        residual=True,
        reduction=1,
        drop_prob_main=0.0,
    ):
        super().__init__()

        # assert chw is not None, "chw must be specified"

        self.channels_in = channels_in
        self.channels_out = channels_in if channels_out is None else channels_out
        self.channels_internal = self.channels_out // reduction
        self.chw = chw
        self.group_size = group_size
        self.activation = activation
        self.activation_out = activation if activation_out is None else activation_out
        self.residual = residual
        self.reduction = reduction
        self.squeeze = SE_Block(self.channels_out, 16)

        self.matcher = nn.Conv2d(self.channels_in, self.channels_out, 1, padding=0, bias=False) if self.channels_in != self.channels_out else None

        if self.chw is None:
            self.norm1 = nn.BatchNorm2d(self.channels_internal)
            self.norm2 = nn.BatchNorm2d(self.channels_internal)
            # self.norm1 = nn.Identity()
            # self.norm2 = nn.Identity()

        elif self.chw == 'group_norm':
            self.norm1 = nn.GroupNorm(8, self.channels_internal)
            self.norm2 = nn.GroupNorm(8, self.channels_internal)

        else:
            self.norm1 = nn.LayerNorm([self.channels_internal, self.chw[1], self.chw[2]])
            self.norm2 = nn.LayerNorm([self.channels_internal, self.chw[1], self.chw[2]])
            # self.norm1 = ManualLayerNorm([self.channels_internal, self.chw[1], self.chw[2]])
            # self.norm2 = ManualLayerNorm([self.channels_internal, self.chw[1], self.chw[2]])

        self.conv1 = nn.Conv2d(self.channels_in, self.channels_internal, 1, padding=0, bias=False)
        # self.conv2 = nn.Conv2d(self.channels_internal, self.channels_internal, 3, padding=1, groups=self.group_size, bias=False, padding_mode="replicate")
        self.conv2 = nn.Conv2d(self.channels_internal, self.channels_internal, 3, padding=1, bias=False, padding_mode="replicate")
        self.conv3 = nn.Conv2d(self.channels_internal, self.channels_out, 1, padding=0, bias=True)

        self.scaler = ScaleSkip2D(self.channels_out) if self.residual else None

        if drop_prob_main > 0.0:
            # Using 2D dropout on features
            self.dropout_main = nn.Dropout2d(p=drop_prob_main)
        else:
            self.dropout_main = None

    def forward(self, x):
        identity = x if self.matcher is None else self.matcher(x)
        
        # print(" ")
        # init_x = x.mean().item()
        # print(f"Input: min={x.min().item()}, max={x.max().item()}, mean={x.mean().item()}, std={x.std().item()}")
        x = self.conv1(x)
        # print(f"{init_x:.16f} -- Before Norm: Min={x.min().item()} Max={x.max().item()} Mean={x.mean().item()} NaNs={torch.isnan(x).sum().item()}")
        x = self.norm1(x)
        # print(f"After norm1: min={x.min().item()}, max={x.max().item()}, mean={x.mean().item()}, std={x.std().item()}")
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        if self.dropout_main is not None:
            x = self.dropout_main(x)

        x = self.conv3(x)
        x = self.squeeze(x)

        if self.residual:
            x = self.scaler(x, identity)

        x = self.activation_out(x)
        
        # print(f"{init_x:.16f} -- After Conv: Min={x.min().item()} Max={x.max().item()} Mean={x.mean().item()} NaNs={torch.isnan(x).sum().item()}")

        return x