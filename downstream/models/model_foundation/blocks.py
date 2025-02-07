# flake8: noqa: E501
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def weights_init(m, size=0.001):
    """
    Initialise the weights of a module. Does not change the default initialisation
    method of linear, conv2d, or conv2dtranspose layers.                                               

    Parameters
    ----------
    m : torch.nn.Module
        Module to initialise
    
    size : float
        Standard deviation of the normal distribution to sample initial values from
        default: 0.001

    Returns
    -------
    None
    """
    if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        if m.bias is not None:
            nn.init.trunc_normal_(m.bias, 0.0, size)

            while torch.any(m.bias == 0.0):
                nn.init.trunc_normal_(m.bias, 0.0, size)

    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)) and m.bias is not None:
        nn.init.trunc_normal_(m.bias, 0.0, size)

        while torch.any(m.bias == 0.0):
            nn.init.trunc_normal_(m.bias, 0.0, size)


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


class GaussianDropout2d(nn.Module):
    """
    Drop out channels of a 2D input with Gaussian noise.

    Parameters
    ----------
    p : float
        Probability of dropping a channel
        default: 0.5

    signal_to_noise : tuple
        Range of signal to noise ratios to use for the dropped channels. 0.0 is pure noise, 1.0 is pure signal.
        The amount of signal is randomly sampled from this range for each channel.
        If None, no signal is added to the dropped channels.
        default: (0.1, 0.9)
    """
    def __init__(self, p=0.5, signal_to_noise=(0.1, 0.9)):
        super(GaussianDropout2d, self).__init__()
        self.p = p
        self.signal_to_noise = signal_to_noise

    def forward(self, x):
        if self.training:
            batch_size, num_channels, height, width = x.size()

            # Create a mask of channels to drop
            mask = torch.rand(batch_size, num_channels, 1, 1, device=x.device) > self.p

            # If all channels are dropped, redraw the mask
            while torch.all(mask):
                mask = torch.rand(batch_size, num_channels, 1, 1, device=x.device) > self.p

            mean = x.mean([2, 3], keepdim=True).repeat(1, 1, height, width)
            std = x.std([2, 3], keepdim=True).repeat(1, 1, height, width)

            # Create the noise (Same mean and std as the input)
            noise = torch.normal(mean, torch.clamp(std, min=1e-6))

            if self.signal_to_noise is not None:
                signal_level = torch.rand(batch_size, num_channels, 1, 1, device=x.device) * (self.signal_to_noise[1] - self.signal_to_noise[0]) + self.signal_to_noise[0]
                adjusted_noise = noise * (1 - signal_level)
                adjusted_signal = x * signal_level

            # Combine the adjusted noise and signal
            return (adjusted_signal * mask) + (adjusted_noise * (~mask))
        
        return x
    

class GaussianDropout1d(nn.Module):
    def __init__(self, p=0.5):
        super(GaussianDropout1d, self).__init__()
        self.p = p
    
    def forward(self, x):
        if self.training:
            batch_size, size = x.size()

            # Create a mask of channels to drop
            mask = torch.rand(batch_size, size, device=x.device) > self.p

            # If all channels are dropped, redraw the mask
            while torch.all(mask):
                mask = torch.rand(batch_size, size, device=x.device) > self.p

            mean = x.mean([1], keepdim=True).repeat(1, size)
            std = x.std([1], keepdim=True).repeat(1, size)

            # Create the noise (Same mean and std as the input)
            noise = torch.normal(mean, torch.clamp(std, min=1e-6))

            # Combine the adjusted noise and signal
            return (x * mask) + (noise * (~mask))
        
        return x


class RandomMask2D(nn.Module):
    """
    Randomly masks pixels of an image with zeros across all channels

    Parameters
    ----------
    p : float
        Probability of masking a pixel
        default: 0.5
    """
    def __init__(self, p=0.5):
        super(RandomMask2D, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = torch.rand(x.size(0), 1, x.size(2), x.size(3), device=x.device) > self.p

            return x * mask

        return x


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


class ScaleSkip1D(nn.Module):
    """
    Learnable weight and bias for 1D skip connections.
    """
    def __init__(self, drop_y=None, size=0.01):
        super(ScaleSkip1D, self).__init__()

        self.size = size
        self.drop_y = drop_y

        # Learnable scale and bias
        self.x_skipscale = nn.Parameter(torch.ones(1, 1))
        self.y_skipscale = nn.Parameter(torch.ones(1, 1))
        self.y_skipbias = nn.Parameter(torch.zeros(1, 1))
        self.x_skipbias = nn.Parameter(torch.zeros(1, 1))

        self.set_weights()
        while torch.any(self.x_skipscale == 0) or torch.any(self.y_skipscale == 0) or torch.any(self.y_skipbias == 0) or torch.any(self.x_skipbias == 0):
            self.set_weights()

        if self.drop_y is not None and self.drop_y > 0.0:
            self.drop_y = GaussianDropout1d(self.drop_y)
            # self.drop_y = nn.Dropout1d(self.drop_y)
        else:
            self.drop_y = None

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
        y = self.excitation(y).view(bs, c, 1, 1)

        return x * y.expand_as(x)



class ManualLayerNorm(nn.Module):
    """
    Manual LayerNorm that normalizes across the *last 3* dims: (C,H,W).
    Typically used for inputs of shape (N, C, H, W).
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            # Initialize weights and biases already in the shape (1, C, H, W) for direct broadcasting
            self.weight = nn.Parameter(torch.ones((1, *self.normalized_shape)))
            self.bias   = nn.Parameter(torch.zeros((1, *self.normalized_shape)))
            # self.weight = nn.Parameter(torch.ones((1, 8, 128, 128)))
            # self.bias   = nn.Parameter(torch.zeros((1, 8, 128, 128)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (N, C, H, W)
        N = x.shape[0]
        # x = x.view(1, 8, 128, 128)
        # Compute mean and var over dimensions (1, 2, 3) which correspond to (C, H, W)
        mean = x.mean(dim=(1,2,3), keepdim=True)
        var  = x.var(dim=(1,2,3), keepdim=True, unbiased=False)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Apply elementwise affine transformation directly using broadcasting
        if self.elementwise_affine:
            x_norm = x_norm * self.weight + self.bias  # Broadcasting happens here

        return x_norm



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
        activation_out=None,
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

        x = self.conv1(x)
        x = self.norm1(x)
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

        return x


# -------------------------------------------------------------------
# Approximate Activations for OV Compatibility (NOT USED)
# -------------------------------------------------------------------
class ApproxGELU(nn.Module):
    def forward(self, x):
        # 0.7978845608 is sqrt(2/pi)
        return 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

import math
class ErfGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2)))

class ApproxChannelGLU(nn.Module):
    def __init__(self):
        super(ApproxChannelGLU, self).__init__()
        self.gelu = ErfGELU()
    
    def forward(self, x):
        if x.dim() != 4:
            return self.gelu(x)

        split = x.size(1) // 2
        split_left = x[:, split:, :, :]
        split_right = x[:, :split, :, :]

        return torch.cat([
            torch.sigmoid(split_right) * split_left,
            self.gelu(split_right),
        ], dim=1)






# USED

def make_bilinear_upsample(in_channels: int):
    """
    Returns a ConvTranspose2d layer that does a fixed 2× bilinear upsampling
    for 'in_channels' channels without cross-channel mixing.
    """
    layer = nn.ConvTranspose2d(
        in_channels  = in_channels,
        out_channels = in_channels,
        kernel_size  = 4,
        stride       = 2,
        padding      = 1,
        groups       = in_channels,  # ensures each channel is processed separately
        bias         = False
    )

    def bilinear_filter_2d(kernel_size: int) -> np.ndarray:
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5

        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center)/factor) * (1 - abs(og[1] - center)/factor)
        return filt

    kernel_size = layer.kernel_size[0]
    filt_2d = bilinear_filter_2d(kernel_size)

    w = np.zeros((in_channels, 1, kernel_size, kernel_size), dtype=np.float32)
    for c in range(in_channels):
        w[c, 0, :, :] = filt_2d

    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(w))

    return layer
