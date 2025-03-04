import torch
import torch.nn as nn
from torchvision.models import resnet50
from models.model_DecoderUtils import CoreDecoder, DecoderBlock

def load_moco_weights_to_base(ckpt_path):
    # Load the MoCo checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    # Create a new dictionary with keys stripped of the MoCo prefix.
    new_state_dict = {}
    for k, v in state_dict.items():
        # Only consider the query encoder and ignore its fc (projection head) layers.
        if k.startswith("module.encoder_q") and not k.startswith("module.encoder_q.fc"):
            new_key = k.replace("module.encoder_q.", "")
            new_state_dict[new_key] = v

    # Create a base resnet50 model without any pretrained weights.
    base_model = resnet50()
    base_model.conv1 = torch.nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # Load the new state dict into the base model.
    msg = base_model.load_state_dict(new_state_dict, strict=False)
    print("Base model load message:", msg)
    return base_model


class MocoRN50(nn.Module):
    def __init__(self, output_dim=1, path_model_weights=True, decoder_norm='batch', decoder_padding='same',
                 decoder_activation='relu', decoder_depths=[2, 2, 8, 2], decoder_dims=[160, 320, 640, 1280]):
        super(MocoRN50, self).__init__()
        if path_model_weights:
            self.encoder = load_moco_weights_to_base(path_model_weights)

        else:
            self.encoder = resnet50()

        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])

        # CNN Decoder Blocks:
        self.depths = decoder_depths
        self.dims = decoder_dims
        self.output_dim = output_dim

        self.decoder_head = CoreDecoder(embedding_dim=2048,
                                        output_dim=output_dim,
                                        depths=decoder_depths, 
                                        dims= decoder_dims,
                                        norm=decoder_norm,
                                        activation=decoder_activation,
                                        padding=decoder_padding,)

        self.decoder_upsample_block = nn.Sequential(DecoderBlock(depth=1, in_channels=2048,
                                                                 out_channels=2048,                 
                                                                 norm=decoder_norm,
                                                                 activation=decoder_activation,
                                                                 padding=decoder_padding,))



    def forward(self, x):
        # order S2 bands: 0-B02, 1-B03, 2-B04, 3-B08, 4-B05, 5-B06, 6-B07, 7-B8A, 8-B11, 9-B12
        # x = x[:, (2, 1, 0), :, :] # select RGB bands
        x = self.encoder(x)
        x = self.decoder_upsample_block(x)
        x = self.decoder_head(x)
        return x


class MocoRN50_Classifier(nn.Module):
    def __init__(self, output_dim=1, path_model_weights=True):
        super(MocoRN50_Classifier, self).__init__()
        if path_model_weights:
            self.encoder = load_moco_weights_to_base(path_model_weights)

        else:
            self.encoder = resnet50()

        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        self.classification_head = nn.Sequential(nn.Flatten(start_dim=1, end_dim=-1),
                                  nn.Linear(2048, output_dim))

    def forward(self, x):
        # order S2 bands: 0-B02, 1-B03, 2-B04, 3-B08, 4-B05, 5-B06, 6-B07, 7-B8A, 8-B11, 9-B12
        # x = x[:, (2, 1, 0), :, :] # select RGB bands
        x = self.encoder(x)
        x = self.classification_head(x)
        return x

def moco_resnet(path_model_weights, output_dim=1, freeze_body=True, classifier=False, **kwargs):

    if classifier:
        model = MocoRN50_Classifier(output_dim=output_dim, path_model_weights=path_model_weights)
        
    else:
        model = MocoRN50(output_dim=output_dim, path_model_weights=path_model_weights, **kwargs)

    if freeze_body:
        for _, param in model.encoder.named_parameters():
            param.requires_grad = False

    return model


