# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np

from timm.models.vision_transformer import Block
from timm.models.layers import to_2tuple
from functools import partial
from terratorch.registry import BACKBONE_REGISTRY
from terratorch.models.backbones.terramind.model.terramind_register import v1_pretraining_mean, v1_pretraining_std

from models.model_DecoderUtils import CoreDecoder


class TerraMindSegmenter(nn.Module):
    """
    TerraMind with pre-defined decoder head.
    """

    def __init__(self, model, embed_dim=768, output_dim=1,
                 decoder_norm='batch', decoder_padding='same',
                 decoder_activation='relu', decoder_depths=[2, 2, 8, 2], decoder_dims=[160, 320, 640, 1280]
                 ):
        super().__init__()

        self.depths = decoder_depths
        self.dims = decoder_dims
        self.output_dim = output_dim
        self.model = model

        self.decoder_head = CoreDecoder(embedding_dim=embed_dim,
                                        output_dim=output_dim,
                                        depths=decoder_depths,
                                        dims=decoder_dims,
                                        activation=decoder_activation,
                                        padding=decoder_padding,
                                        norm=decoder_norm)

        self.decoder_downsample_block = nn.Identity()

    def reshape(self, x):
        # Separate channel axis
        N, L, D = x.shape
        x = x.permute(0, 2, 1)
        x = x.view(N, D, int(L ** 0.5), int(L ** 0.5))
        return x

    def forward(self, x):
        return self.model(x)


class TerraMindClassifier(nn.Module):
    """
    TerraMind with pre-defined classification head.
    """
    def __init__(self, model, embed_dim=768, output_dim=1):
        super().__init__()
        self.model = model
        self.classification_head = nn.Sequential(nn.Linear(in_features=embed_dim, out_features=int(embed_dim/2)),
                                                 nn.LayerNorm(int(embed_dim/2)),
                                                 nn.ReLU(),
                                                 nn.Linear(in_features=int(embed_dim/2), out_features=output_dim)
                                                 )

    def forward(self, x):
        return self.model(x)


def terramind(output_dim=1, decoder_norm='batch', decoder_padding='same',
            decoder_activation='relu', decoder_depths=[2, 2, 8, 2], decoder_dims=[160, 320, 640, 1280], freeze_body=True,
            classifier=False, inference=False):

    model = BACKBONE_REGISTRY.build(
        "terramind_v1_base",
        modalities=["S2L1C"],
        pretrained=True,
    )

    if classifier:
        model = TerraMindClassifier(model=model,
                                    output_dim=output_dim)

    else:
        model = TerraMindSegmenter(model=model,
                                   output_dim=output_dim,
                                   decoder_norm=decoder_norm,
                                   decoder_padding=decoder_padding,
                                   decoder_activation=decoder_activation,
                                   decoder_depths=decoder_depths,
                                   decoder_dims=decoder_dims
                                   )

    if freeze_body:
        for _, param in model.model.encoder.named_parameters():
            param.requires_grad = False

    model.float()
    return model
