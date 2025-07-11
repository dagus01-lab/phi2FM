import os
import torch
import torch.nn as nn
from torchvision.models import resnet50
from models.model_DecoderUtils import CoreDecoder, DecoderBlock

def load_dino_weights_to_base(ckpt_path, in_channels=13):
    """
    Load a DINO pretrained checkpoint into a base ResNet50 model.
    
    This function:
      - Loads the checkpoint.
      - Checks if the checkpoint has a 'student' state dict; if not, it looks for 'teacher'.
      - Removes prefixes ("module.", "backbone.", and "teacher." if present).
      - Replaces the first conv layer to accept in_channels.
      - Loads the state dict into a ResNet50 instance.
    """
    if not os.path.isfile(ckpt_path):
        print("DINO checkpoint not found at {}".format(ckpt_path))
        return resnet50()
    
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Prefer the student weights; if not available, try teacher.
    if "student" in state_dict:
        state_dict = state_dict["student"]
    elif "teacher" in state_dict:
        print(f'WARNING: Using teacher weights for DINO model because available keys are: {state_dict.keys()}')
        state_dict = state_dict["teacher"]
        
        

    # Remove prefixes: "module.", "backbone.", and "teacher." if still present.
    state_dict = {k.replace("module.", "")
                     .replace("backbone.", "")
                     .replace("teacher.", ""): v 
                  for k, v in state_dict.items()}

    # Create a base ResNet50 model.
    base_model = resnet50()
    # Modify the first convolution layer to accept in_channels (e.g., 13).
    base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Load the state dict into the base model.
    msg = base_model.load_state_dict(state_dict, strict=False)
    print("Base DINO model load message:", msg)
    return base_model

class DINOResNet50(nn.Module):
    def __init__(self, output_dim=1, path_model_weights=None, in_channels=13,
                 decoder_norm='batch', decoder_padding='same',
                 decoder_activation='relu', decoder_depths=[2, 2, 8, 2],
                 decoder_dims=[160, 320, 640, 1280]):
        super(DINOResNet50, self).__init__()
        if path_model_weights:
            self.encoder = load_dino_weights_to_base(path_model_weights, in_channels=in_channels)
        else:
            self.encoder = resnet50()
            if in_channels != 3:
                self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Extract encoder layers (drop avgpool and fc).
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        
        # Save decoder parameters.
        self.depths = decoder_depths
        self.dims = decoder_dims
        self.output_dim = output_dim

        self.decoder_head = CoreDecoder(embedding_dim=2048,
                                        output_dim=output_dim,
                                        depths=decoder_depths,
                                        dims=decoder_dims,
                                        norm=decoder_norm,
                                        activation=decoder_activation,
                                        padding=decoder_padding)

        self.decoder_upsample_block = nn.Sequential(
            DecoderBlock(depth=1, in_channels=2048,
                         out_channels=2048,
                         norm=decoder_norm,
                         activation=decoder_activation,
                         padding=decoder_padding)
        )

    def forward(self, x):
        # Optionally re-order channels if needed (e.g., select RGB channels)
        # x = x[:, (2, 1, 0), :, :]
        x = self.encoder(x)
        x = self.decoder_upsample_block(x)
        x = self.decoder_head(x)
        return x

class DINOResNet50_Classifier(nn.Module):
    def __init__(self, output_dim=1, path_model_weights=None, in_channels=13):
        super(DINOResNet50_Classifier, self).__init__()
        if path_model_weights:
            self.encoder = load_dino_weights_to_base(path_model_weights, in_channels=in_channels)
        else:
            self.encoder = resnet50()
            if in_channels != 3:
                self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Extract encoder layers (drop only the fc layer).
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        self.classification_head = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(2048, output_dim)
        )

    def forward(self, x):
        # Reorder channels to RGB if desired:
        # x = x[:, (2, 1, 0), :, :]
        x = self.encoder(x)
        x = self.classification_head(x)
        return x

def dino_resnet(path_model_weights, output_dim=1, freeze_body=True, classifier=False, in_channels=13, **kwargs):
    if classifier:
        model = DINOResNet50_Classifier(output_dim=output_dim,
                                        path_model_weights=path_model_weights,
                                        in_channels=in_channels)
    else:
        model = DINOResNet50(output_dim=output_dim,
                             path_model_weights=path_model_weights,
                             in_channels=in_channels,
                             **kwargs)
    if freeze_body:
        for _, param in model.encoder.named_parameters():
            param.requires_grad = False
    return model
