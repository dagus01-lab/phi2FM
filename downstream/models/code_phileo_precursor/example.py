import torch
import torch.nn as nn

from model_foundation_local_rev2 import Foundation as Foundation_local
from blocks import CNNBlock


class FloodPredictorHSL(nn.Module):
    def __init__(
        self,
        *,
        input_dim=5,
        output_dim=None,
        path_weights="",
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.path_weights = path_weights

        self.foundation = Foundation_local(
            input_dim=10, # B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
            depths=[3, 3, 4, 4, 5],  # 128, 64, 32, 16, 8
            dims=[32, 32, 32, 64, 64],
            img_size=128,
            latent_dim=1024,
            dropout=[0.85, 0.90, 0.90, 0.95, 0.95],
            activation=nn.LeakyReLU(),
        )

        if self.path_weights != "":
            weights = torch.load(self.path_weights)
            self.foundation.load_state_dict(weights, strict=False)

        self.encoder = self.foundation.encoder # expects dim 32 input
        self.decoder = self.foundation.decoder

        self.stem = CNNBlock(self.input_dim, 32, activation=nn.LeakyReLU(), residual=False, chw=[self.input_dim, 128, 128])
        self.head = CNNBlock(64, self.output_dim, activation=nn.LeakyReLU(), activation_out=nn.Sigmoid(), chw=[self.output_dim, 128, 128])

    def forward(self, x):
        x = self.stem(x)
        embeddings, embeddings_cnn, skips, predictions = self.encoder(x)
        decoded = self.decoder(embeddings, skips)
        reconstruction = self.head(decoded)

        return reconstruction, embeddings, embeddings_cnn, decoded, predictions
