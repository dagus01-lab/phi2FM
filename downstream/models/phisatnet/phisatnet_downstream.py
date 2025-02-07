import torch
import torch.nn as nn

from geoaware_blocks import CoreCNNBlock
from geoaware_foundation import FoundationEncoder, FoundationDecoder

# Assuming CoreCNNBlock, FoundationEncoder, FoundationDecoder are defined elsewhere.

class phisatnet_downstream(nn.Module):
    def __init__(
        self,
        *,
        pretrained_path,  # path to the .pt file with pretrained weights
        task,           # either "segmentation" or "classification"
        input_dim=3,
        output_dim=None,  # number of segmentation classes or classification outputs
        depths=None,      # list of encoder depths (e.g., [2,2,6,2])
        dims=None,        # list of channel dimensions (e.g., [64, 128, 256, 512])
        img_size=128,     # image resolution (if needed)
        norm_foundation="group",
        norm_downstream="batch",
        activation="gelu",
        freeze_body=False  # when True, freezes the pretrained stem and encoder parameters
    ):
        """
        Downstream model that reuses the pretrained stem and encoder from the foundation model.
        For segmentation, a new FoundationDecoder, bridge, and head are added.
        For classification, a new classification head is added.
        
        Args:
            pretrained_path (str): Path to the .pt file with pretrained weights.
            task (str): Either "segmentation" or "classification".
            input_dim (int): Number of input channels.
            output_dim (int): Number of segmentation classes or classification outputs.
            depths (list): List of encoder depths (e.g., [2,2,6,2]).
            dims (list): List of channel dimensions (e.g., [64, 128, 256, 512]).
            img_size (int): Image resolution.
            norm_foundation (str): Normalization type for the foundation (e.g., "group").
            norm_downstream (str): Normalization type for the downstream (e.g., "batch").
            activation (str): Activation function to use (e.g., "gelu").
            freeze_body (bool): If True, freezes the parameters of the pretrained stem and encoder.
        """
        super().__init__()

        if depths is None or dims is None:
            raise ValueError("Both 'depths' and 'dims' must be specified and match the pretrained model.")
        if output_dim is None:
            raise ValueError("An output_dim (number of classes or output features) must be provided.")

        self.task = task.lower()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depths = depths
        self.dims = dims
        self.norm_foundation = norm_foundation
        self.norm_downstream = norm_downstream
        self.activation = activation
        self.img_size = img_size

        # ----------------------------------
        # 1) Reuse the pretrained stem and encoder
        # ----------------------------------
        # These are created with the same hyper-parameters as in the foundation model.
        self.stem = CoreCNNBlock(
            in_channels=self.input_dim,
            out_channels=self.dims[0],
            norm=self.norm_foundation,
            activation=self.activation,
            residual=True
        )
        self.encoder = FoundationEncoder(
            input_dim=self.dims[0],
            depths=self.depths,
            dims=self.dims,
            norm=self.norm_foundation,
            activation=self.activation,
        )

        # Load the pretrained weights for the stem and encoder.
        self._load_pretrained(pretrained_path)

        # Freeze stem and encoder if requested.
        if freeze_body:
            for param in self.stem.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = False

        # ----------------------------------
        # 2) Build the downstream branch
        # ----------------------------------
        if self.task == "segmentation":
            # For segmentation we add a new FoundationDecoder (newly initialized)
            self.decoder = FoundationDecoder(
                depths=self.depths,
                dims=self.dims,
                norm=self.norm_downstream,
                activation=self.activation,
            )
            # New bridge that mimics the original bridge but freshly initialized
            self.bridge = nn.Sequential(
                CoreCNNBlock(
                    in_channels=self.dims[-1],
                    out_channels=self.dims[-1],
                    norm=self.norm_downstream,
                    activation=self.activation,
                )
            )
            # New head that processes the decoder output into segmentation logits
            self.head = nn.Sequential(
                CoreCNNBlock(
                    in_channels=self.dims[0],
                    out_channels=self.dims[0],
                    norm=self.norm_downstream,
                    activation=self.activation,
                ),
                nn.Conv2d(self.dims[0], self.output_dim, kernel_size=1, padding=0)
            )
        elif self.task == "classification":
            # For classification, no decoder or bridge is used. Instead, we add a classification head.
            self.decoder = None
            self.bridge = None
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(self.dims[-1], self.output_dim)
            )
        else:
            raise ValueError("Unsupported task. Choose either 'segmentation' or 'classification'.")

    def _load_pretrained(self, pretrained_path):
        """
        Loads the pretrained weights (from a .pt file) into the stem and encoder.
        Assumes that the saved state dict has keys prefixed with "stem." and "encoder.".
        Also handles the case when the keys are prefixed with "module." due to DDP.
        """
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        # If the checkpoint is a dict with a 'state_dict' key, use that.
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        # Remove "module." prefix if it exists.
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith("module."):
                new_key = key[len("module."):]
            new_state_dict[new_key] = value

        # Load stem weights.
        stem_state = {
            key[len("stem."):]: value
            for key, value in new_state_dict.items() if key.startswith("stem.")
        }
        missing, unexpected = self.stem.load_state_dict(stem_state, strict=False)
        if missing:
            print("Warning: The following keys were not found in the pretrained stem:", missing)
        if unexpected:
            print("Note: The following unexpected keys in pretrained stem were ignored:", unexpected)

        # Load encoder weights.
        encoder_state = {
            key[len("encoder."):]: value
            for key, value in new_state_dict.items() if key.startswith("encoder.")
        }
        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        if missing:
            print("Warning: The following keys were not found in the pretrained encoder:", missing)
        if unexpected:
            print("Note: The following unexpected keys in pretrained encoder were ignored:", unexpected)

    def forward(self, x):
        """
        Forward pass.
          - For segmentation:
              x -> stem -> encoder -> bridge -> decoder -> head -> segmentation logits (B, output_dim, H, W)
          - For classification:
              x -> stem -> encoder -> head -> classification logits (B, output_dim)
        """
        # 1) Stem
        x_stem = self.stem(x)  # shape: (B, dims[0], H, W)

        # 2) Encoder
        bottom, skips = self.encoder(x_stem)  # bottom: (B, dims[-1], ...)

        # 3) Downstream branch
        if self.task == "segmentation":
            bottom_feats = self.bridge(bottom)
            decoded_feats = self.decoder(bottom_feats, skips)  # shape: (B, dims[0], H, W)
            seg_logits = self.head(decoded_feats)  # shape: (B, output_dim, H, W)
            return seg_logits

        elif self.task == "classification":
            class_logits = self.head(bottom)  # shape: (B, output_dim)
            return class_logits
