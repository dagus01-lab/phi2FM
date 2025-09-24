import torch
import torch.nn as nn
from copy import deepcopy
from models.seco_utils import moco2_module, segmentation
from models.model_DecoderUtils import CoreDecoder, DecoderBlock
from terratorch.models.encoder_decoder_factory import EncoderDecoderFactory, _get_decoder_and_head_kwargs

class UNetDecoderNoSkipWrapper(nn.Module):
    """
    Wrapper for UNetDecoder to mimic CoreDecoder behavior:
    - Uses only the deepest encoder feature (no skip connections)
    - Decoder dimensions match CoreDecoder
    """
    def __init__(
        self,
        embedding_dim=2048,
        output_dim=1,
        depths=None,
        dims=None,
        activation="relu",
        norm="batch",
        padding="same"
    ):
        super().__init__()
        self.depths = [3, 3, 9, 3] if depths is None else depths
        self.dims = [96, 192, 384, 768] if dims is None else dims
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim

        # UNetDecoder expects encoder_channels and decoder_channels
        # We fake encoder_channels as if all skip connections are the same as the deepest feature
        encoder_channels = [embedding_dim] * (len(self.dims) + 1)
        decoder_channels = list(self.dims[::-1])  # reverse dims for UNetDecoder order

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=len(decoder_channels),
            use_batchnorm=(norm == "batch"),
            attention_type=None,
            center=False
        )

        # Final head to match CoreDecoder output
        self.head = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(decoder_channels[-1], output_dim, kernel_size=1)
        )

    def forward(self, x):
        # x: [B, embedding_dim, H, W]
        # Create fake skip connections (all same as x)
        features = [x] * (len(self.dims) + 1)
        x = self.decoder(*features)
        #x = self.head(x)
        return x
class Seco(nn.Module):
    def __init__(self, ckpt_path, output_dim=1, decoder_norm='batch', decoder_padding='same',
                 decoder_activation='relu', decoder_depths=[2, 2, 8, 2], decoder_dims=[256, 512, 1024, 2048], decoder_name="UNetDecoder"):
        super(Seco, self).__init__()
        # [160, 320, 640, 1280]
        model = moco2_module.MocoV2.load_from_checkpoint(ckpt_path, map_location='cpu')
        self.encoder = deepcopy(model.encoder_q)

        # CNN Decoder Blocks:
        self.depths = decoder_depths
        self.dims = decoder_dims
        self.output_dim = output_dim

        self.decoder_head = CoreDecoder(embedding_dim=2048,
                                output_dim=output_dim,
                                depths=decoder_depths, 
                                dims= decoder_dims,
                                activation=decoder_activation,
                                padding=decoder_padding, 
                                norm=decoder_norm)

        # # Fix UNetDecoder configuration: encoder_channels must match encoder output (2048), decoder_channels is reversed decoder_dims
        # from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
        # encoder_channels = [0] * (len(decoder_dims))+ [2048]
        # decoder_channels = decoder_dims[::-1]

        # self.decoder_head = UnetDecoder(
        #     encoder_channels=encoder_channels,
        #     decoder_channels=decoder_channels,
        #     n_blocks=len(decoder_channels),
        #     use_batchnorm=(decoder_norm == "batch"),
        #     attention_type=None,
        #     center=False
        # )
        # print(f"Decoder head: {self.decoder_head}")
        self.decoder_upsample_block = nn.Sequential(DecoderBlock(depth=1, in_channels=2048,
                                                                 out_channels=2048,
                                                                 norm=decoder_norm,
                                                                 activation=decoder_activation,
                                                                 padding=decoder_padding,))
        # print(f"Decoder upsampling: {self.decoder_upsample_block}")



    def forward(self, x):
        # order S2 bands: 0-B02, 1-B03, 2-B04, 3-B08, 4-B05, 5-B06, 6-B07, 7-B8A, 8-B11, 9-B12
        x = x[:, (2, 1, 0), :, :] # select RGB bands
        x = self.encoder(x)
        # print(f"Intermediate shape before upsampling: {x.shape}")
        x = self.decoder_upsample_block(x)
        # print(f"Intermediate shape before decoder head: {x.shape}")
        x = self.decoder_head(x)
        #x = self.final_head(x)
        return x


class Seco_Classifier(nn.Module):
    def __init__(self, ckpt_path, output_dim=1):
        super(Seco_Classifier, self).__init__()
        model = moco2_module.MocoV2.load_from_checkpoint(ckpt_path, map_location='cpu')
        self.encoder = deepcopy(model.encoder_q)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1,1)),
                                  nn.Flatten(start_dim=1, end_dim=-1),
                                  nn.Linear(2048, 2048),
                                  nn.ReLU(),
                                  nn.Linear(2048, output_dim))

    def forward(self, x):
        # order S2 bands: 0-B02, 1-B03, 2-B04, 3-B08, 4-B05, 5-B06, 6-B07, 7-B8A, 8-B11, 9-B12
        x = x[:, (2, 1, 0), :, :] # select RGB bands
        x = self.encoder(x)
        x = self.head(x)

        return x


def seasonal_contrast(checkpoint, output_dim=1, freeze_body=True, classifier=False, **kwargs):

    if classifier:
        model = Seco_Classifier(ckpt_path=checkpoint, output_dim=output_dim)
    else:
        model = Seco(ckpt_path=checkpoint, output_dim=output_dim, **kwargs)

    if freeze_body:
        for _, param in model.encoder.named_parameters():
            param.requires_grad = False

    return model


if __name__ == "__main__":
    from torchinfo import summary

    BATCH_SIZE = 4
    CHANNELS = 3
    HEIGHT = 128
    WIDTH = 128

    model = Seco_Classifier(ckpt_path='/phileo_data/pretrained_models/seco_resnet50_1m.ckpt')
    model.cpu()

    x = model(torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)))

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )

    sd = model.state_dict()
    torch.save(sd, 'test.pt')