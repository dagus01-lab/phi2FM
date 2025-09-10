# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import torch.nn as nn

from terratorch.registry import BACKBONE_REGISTRY
from terratorch.models import EncoderDecoderFactory

# from models.model_DecoderUtils import CoreDecoder


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
        x = self.model(x)

        # select cls token
        x = x[:, 0, :]

        y = self.classification_head(x)
        return y


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
        model = EncoderDecoderFactory().build_model(
            task="segmentation",
            backbone="terramind_v1_base",
            backbone_modalities=["S2L1C"],
            decoder="UNetDecoder",
            decoder_channels=[512, 256, 128, 64],
            backbone_pretrained=True,
            num_classes=4,
            necks=[{
                "name": "SelectIndices",
                "indices": [2, 5, 8, 11]
            },
                {"name": "ReshapeTokensToImage",
                 "remove_cls_token": False},
                {"name": "LearnedInterpolateToPyramidal"}]
        )

    if freeze_body:
        for _, param in model.encoder.named_parameters():
            param.requires_grad = False

    model.float()
    return model
