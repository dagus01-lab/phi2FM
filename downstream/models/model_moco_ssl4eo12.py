import torch
import torch.nn as nn
from torchvision.models import resnet50
from models.model_DecoderUtils import CoreDecoder, DecoderBlock
import math
from collections import OrderedDict

def load_moco_weights_to_base(ckpt_path, bands=13):
    # Load the MoCo checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint  # It's already the raw state_dict

    # Create a new dictionary with keys stripped of the MoCo prefix.
    new_state_dict = {}
    for k, v in state_dict.items():
        # Only consider the query encoder and ignore its fc (projection head) layers.
        if k.startswith("module.encoder_q") and not k.startswith("module.encoder_q.fc"):
            new_key = k.replace("module.encoder_q.", "")
            new_state_dict[new_key] = v

    
    # Create a base resnet50 model without any pretrained weights.
    base_model = resnet50()
    base_model.conv1 = torch.nn.Conv2d(bands, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # Load the new state dict into the base model.
    msg = base_model.load_state_dict(new_state_dict, strict=False)
    print("Base model load message:", msg)
    return base_model




def _rename_ckpt_keys(old_key: str):
    """
    Map the top‑level integer prefixes coming from an nn.Sequential back to the
    canonical ResNet names. Return None for keys we intentionally drop
    (e.g. avgpool, fc).
    """
    idx_map = {
        "0": "conv1",
        "1": "bn1",
        "4": "layer1",
        "5": "layer2",
        "6": "layer3",
        "7": "layer4",
    }

    parts = old_key.split(".")
    if parts[0] not in idx_map:     # skip avg‑pool, fc, etc.
        return None
    parts[0] = idx_map[parts[0]]
    return ".".join(parts)


def _adapt_first_conv(weight, in_channels):
    """
    Project a (64, 3, 7, 7) kernel to any number of input channels.
    The simplest—yet surprisingly effective—strategy is to **repeat** the RGB
    filters and average them so that the weight norm stays similar.
    """
    if weight.shape[1] == in_channels:          # 3‑channel case → nothing to do
        return weight

    if in_channels < 3:                         # drop the excess channels
        return weight[:, :in_channels]

    # repeat until we have enough channels, then crop
    repeat = math.ceil(in_channels / 3)
    weight = weight.repeat(1, repeat, 1, 1)     # (64, 3*repeat, 7, 7)
    weight = weight[:, :in_channels]            # crop
    weight /= repeat                            # keep variance roughly constant
    return weight


def load_moco_weights_to_base_caco(ckpt_path: str, bands: int = 13):
    # ---------- 1. read the checkpoint ----------
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # ---------- 2. rename the keys ----------
    new_state = OrderedDict()
    for k, v in raw_state.items():
        new_key = _rename_ckpt_keys(k)
        if new_key is not None:
            new_state[new_key] = v

    # ---------- 3. build the target model ----------
    base_model = resnet50()
    base_model.conv1 = torch.nn.Conv2d(
        bands, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    # ---------- 4. adapt / insert the first‑layer weights ----------
    if "conv1.weight" in new_state:
        new_state["conv1.weight"] = _adapt_first_conv(
            new_state["conv1.weight"], bands
        )

    # ---------- 5. load ----------
    msg = base_model.load_state_dict(new_state, strict=False)
    print("base_model.load_state_dict →", msg)
    return base_model





class MocoRN50(nn.Module):
    def __init__(self, output_dim=1, path_model_weights=True, decoder_norm='batch', decoder_padding='same',
                 decoder_activation='relu', decoder_depths=[2, 2, 8, 2], decoder_dims=[160, 320, 640, 1280],
                 bands=13, is_caco=False):
        super(MocoRN50, self).__init__()
        if path_model_weights:
            if is_caco:
                self.encoder = load_moco_weights_to_base_caco(path_model_weights, bands=bands)
            else:
                self.encoder = load_moco_weights_to_base(path_model_weights, bands=bands)

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
    def __init__(self, output_dim=1, path_model_weights=True, bands=13, is_caco=False):
        super(MocoRN50_Classifier, self).__init__()

        if path_model_weights:
            if is_caco:
                self.encoder = load_moco_weights_to_base_caco(path_model_weights, bands=bands)
            else:
                self.encoder = load_moco_weights_to_base(path_model_weights, bands=bands)

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

def moco_resnet(path_model_weights, output_dim=1, freeze_body=True, classifier=False, bands=13, is_caco=False, **kwargs):

    if classifier:
        model = MocoRN50_Classifier(output_dim=output_dim, path_model_weights=path_model_weights, bands=bands, is_caco=is_caco)
        
    else:
        model = MocoRN50(output_dim=output_dim, path_model_weights=path_model_weights, bands=bands, is_caco=is_caco, **kwargs)

    if freeze_body:
        for _, param in model.encoder.named_parameters():
            param.requires_grad = False

    return model


