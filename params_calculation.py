import torch
import torch.nn as nn
from downstream.models.phisatnet_downstream import PhiSatNetDownstream
from downstream.training_script import get_models_pretrained
from terratorch.models import EncoderDecoderFactory

def get_module_size(module: nn.Module) -> float:
    """
    Returns the memory size of all parameters in a given module
    """
    total_params = 0
    for param in module.parameters():
        total_params += param.numel() #* param.element_size()
    return total_params / (1000 ** 2)  # Milions

if __name__ == "__main__":
    # Fill these according to your setup
    pretrained_paths = {
        "phisatnet":"/home/gdaga/pretrained_weights/phisat2net_geoaware_best.pt",
        # phisatnet has input_dim=8, output_dim=4, img_size=224, depths=[2, 2, 8, 2], dims=[80, 160, 320, 640] 
        "seasonal_contrast": "/home/gdaga/pretrained_weights/seco_resnet50_1m.ckpt",

    }
    task = "segmentation"  # decoder only exists for segmentation

    input_dim = 8          # or 3, depending on your model
    output_dim = 3         # e.g. number of classes
    input_size = 224      # image size used during training
    for model_name, pretrained_path in pretrained_paths.items():
        model = get_models_pretrained(model_name, input_dim, output_dim, input_size, path_model_weights=pretrained_path, freeze=True)
        print("#"*20)
        print(f"Model: {model_name}")
        print("#"*20)
        print(f"Model depths: {model.depths}")

        # model = EncoderDecoderFactory().build_model(
        #     task="segmentation",
        #     backbone="terramind_v1_tiny",
        #     backbone_modalities=["S2L1C"],
        #     decoder="UNetDecoder",
        #     decoder_channels=[512, 256, 128, 64],
        #     backbone_pretrained=True,
        #     num_classes=4,
        #     necks=[{
        #         "name": "SelectIndices",
        #         "indices": [2, 5, 8, 11]
        #     },
        #         {"name": "ReshapeTokensToImage",
        #             "remove_cls_token": False},
        #         {"name": "LearnedInterpolateToPyramidal"}]
        # )

        # Get size of decoder
        if model_name == "phisatnet":
            bridge = model.bridge
            head = model.head
            if bridge is not None:
                bridge_size = get_module_size(bridge)
                print(f"Bridge params count: {bridge_size:.2f} M")
            if head is not None:
                head_size = get_module_size(head)
                print(f"Head params count: {head_size:.2f} M")
            decoder = model.decoder
        else:
            decoder = model.decoder_head
        decoder_size = get_module_size(decoder)
        print(f"Decoder params count: {decoder_size:.2f} M")

        # (Optional) Full model size
        full_size = get_module_size(model)
        print(f"Full model params count: {full_size:.2f} M")

        # Feed a random tensor and print output shape
        x = torch.randn((1, input_dim, input_size, input_size))  # batch size 1, 8 channels, 244x244
        # try:
        output = model(x)
        print(f"Output shape: {output.shape}")
        # except Exception as e:
        #     print(f"Error during forward pass: {e}")
    
