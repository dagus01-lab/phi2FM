import torch
import torch.nn as nn
import warnings

from pretrain.models.geoaware_blocks import CoreCNNBlock
from pretrain.models.geoaware_foundation import FoundationEncoder, FoundationDecoder


class PhiSatNetDownstream(nn.Module):
    def __init__(
        self,
        *,
        pretrained_path: str,
        task: str,
        input_dim: int = 3,
        output_dim: int = None,
        depths: list = None,
        dims: list = None,
        img_size: int = 224,
        norm_foundation: str = "group",
        norm_downstream: str = "batch",
        activation: str = "gelu",
        freeze_body: bool = False
    ):
        """
        Downstream model that reuses the pretrained stem and encoder from the foundation model.
        For segmentation, a new FoundationDecoder, bridge, and head are added.
        For classification, a new classification head is added.
        
        Args:
            pretrained_path (str): Path to the .pt file with pretrained model (of class phisat2net_geoaware).
            task (str): Either "segmentation" or "classification".
            input_dim (int): Number of input channels (10 for S2, 8 for phisat2).
            output_dim (int): Number of classification classes (either pixel or image level). For regression, set to 1.
            depths (list): Encoder depths (e.g., [2, 2, 6, 2]).
            dims (list): Channel dimensions (e.g., [64, 128, 256, 512]).
            img_size (int): Image resolution (model was pretrained with 224x224).
            norm_foundation (str): Normalization type for the stem and encoder.
            norm_downstream (str): Normalization type for the bridge, decoder, and head.
            activation (str): Activation function to use.
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

        # Load the pretrained weights.
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

        # (Optional) Freeze stem and encoder.
        if freeze_body:
            for param in self.stem.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = False

        # ----------------------------------
        # 2) Build the downstream branch
        # ----------------------------------
        if self.task == "segmentation":
            # Bridge + Decoder + Head (no pretrained weights used here).
            self.bridge = nn.Sequential(
                CoreCNNBlock(
                    in_channels=self.dims[-1],
                    out_channels=self.dims[-1],
                    norm=self.norm_downstream,
                    activation=self.activation,
                )
            )
            self.decoder = FoundationDecoder(
                depths=self.depths,
                dims=self.dims,
                norm=self.norm_downstream,
                activation=self.activation,
            )
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
            # Head (no pretrained weights used here).
            self.decoder = None
            self.bridge = None
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(self.dims[-1], self.output_dim)
            )
        else:
            raise ValueError(f"Task {self.task} not recognized. Must be either 'segmentation' or 'classification'.")

    def _load_pretrained(self, pretrained_path):
        """
        Loads the pretrained weights (from a .pt file) into the stem, encoder,
        and (if applicable) bridge and decoder. Handles the case when the keys are
        prefixed with "module." (due to training with DP/DDP).
        """
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        
        # If the checkpoint is a dict with a 'state_dict' key, use that.
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        # Remove "module." prefix if it exists.
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[len("module."): ] if key.startswith("module.") else key
            new_state_dict[new_key] = value

        # Load stem weights.
        stem_state = {
            key[len("stem."):]: value
            for key, value in new_state_dict.items() if key.startswith("stem.")
        }
        missing, unexpected = self.stem.load_state_dict(stem_state, strict=False)
        if missing:
            warnings.warn(f"The following keys were not found in the pretrained stem: {missing}")
        if unexpected:
            warnings.warn(f"The following unexpected keys in pretrained stem were ignored: {unexpected}")

        # Load encoder weights.
        encoder_state = {
            key[len("encoder."):]: value
            for key, value in new_state_dict.items() if key.startswith("encoder.")
        }
        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        if missing:
            warnings.warn(f"The following keys were not found in the pretrained encoder: {missing}")
        if unexpected:
            warnings.warn(f"The following unexpected keys in pretrained encoder were ignored: {unexpected}")

        # If task is segmentation, also load bridge and decoder weights.
        if self.task == "segmentation":
            # Load bridge weights.
            if self.bridge is not None:
                bridge_state = {
                    key[len("bridge."):]: value
                    for key, value in new_state_dict.items() if key.startswith("bridge.")
                }
                missing, unexpected = self.bridge.load_state_dict(bridge_state, strict=False)
                if missing:
                    warnings.warn(f"The following keys were not found in the pretrained bridge: {missing}")
                if unexpected:
                    warnings.warn(f"The following unexpected keys in pretrained bridge were ignored: {unexpected}")

            # Load decoder weights.
            if self.decoder is not None:
                decoder_state = {
                    key[len("decoder."):]: value
                    for key, value in new_state_dict.items() if key.startswith("decoder.")
                }
                missing, unexpected = self.decoder.load_state_dict(decoder_state, strict=False)
                if missing:
                    warnings.warn(f"The following keys were not found in the pretrained decoder: {missing}")
                if unexpected:
                    warnings.warn(f"The following unexpected keys in pretrained decoder were ignored: {unexpected}")

    def forward(self, x):
        """
        Forward pass.
          - For segmentation:
              x -> stem -> encoder -> bridge -> decoder -> head -> segmentation logits (B, output_dim, H, W)
          - For classification:
              x -> stem -> encoder -> head -> classification logits (B, output_dim)
        """
        # 1) Stem
        x_stem = self.stem(x)                                   # shape: (B, dims[0], H, W)

        # 2) Encoder
        bottom, skips = self.encoder(x_stem)                    # bottom: (B, dims[-1], H//(2^num_stages), W//(2^num_stages))

        # 3) Downstream branch
        if self.task == "segmentation":
            bottom_feats = self.bridge(bottom)                  # shape: (B, dims[-1], H, W)
            decoded_feats = self.decoder(bottom_feats, skips)   # shape: (B, dims[0], H, W)
            seg_logits = self.head(decoded_feats)               # shape: (B, output_dim, H, W)
            return seg_logits

        elif self.task == "classification":
            class_logits = self.head(bottom)                    # shape: (B, output_dim)
            return class_logits
