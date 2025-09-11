import torch
import torch.nn as nn
from downstream.models.phisatnet_downstream import PhiSatNetDownstream
from downstream.training_script import get_models_pretrained

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
    pretrained_path = "/home/gdaga/pretrained_weights/phisat2net_geoaware_best.pt"
    pretrained_path = "/home/gdaga/pretrained_weights/seco_resnet50_1m.ckpt"
    task = "segmentation"  # decoder only exists for segmentation
    model_name = "phisatnet"
    model_name = "seasonal_contrast"
    input_dim = 8          # or 3, depending on your model
    output_dim = 9         # e.g. number of classes
    depths = [2, 2, 6, 2]  # example, set as needed
    input_size = 224      # image size used during training
    model = get_models_pretrained(model_name, input_dim, output_dim, input_size, path_model_weights=pretrained_path, freeze=True)
    print(f"Model depths: {model.depths}")


    # Get size of decoder
    decoder_size = get_module_size(model.decoder_head)#decoder)
    print(f"Decoder params count: {decoder_size:.2f} M")

    # (Optional) Full model size
    full_size = get_module_size(model)
    print(f"Full model params count: {full_size:.2f} M")
    
