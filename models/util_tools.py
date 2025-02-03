import numpy as np
import torch
import torch.nn as nn

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
