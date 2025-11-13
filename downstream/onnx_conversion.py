from collections import OrderedDict
import torch
import torch.nn as nn
from downstream.models.phisatnet_downstream import PhiSatNetDownstream

def get_module_size(module: nn.Module) -> float:
    """
    Returns the memory size of all parameters in a given module
    """
    total_params = 0
    for param in module.parameters():
        total_params += param.numel() #* param.element_size()
    return total_params / (1000 ** 2)  # Milions

class DecoderOnly(nn.Module):
    """Wrapper to export only the decoder part (bridge + decoder + head)"""
    def __init__(self, decoder, head):
        super().__init__()
        self.decoder = decoder
        self.head = head
    
    def forward(self, bottom_feats, *skips):
        # Pass through decoder with skip connections
        decoded_feats = self.decoder(bottom_feats, skips)
        # Apply final head
        seg_logits = self.head(decoded_feats)
        return seg_logits

class EncoderOnly(nn.Module):
    """Wrapper to export only the encoder part"""
    def __init__(self, encoder, stem):
        super().__init__()
        self.encoder = encoder
        self.stem = stem
    def forward(self, x):
        x = self.stem(x)
        bottom, skips = self.encoder(x)
        return bottom, skips


# Set these according to your model/checkpoint
pretrained_model_path = "/home/gdaga/pretrained_weights/phisat2net_geoaware_best.pt"
downstream_model_path = "/Data/phi2FM_n_shot/lp/phisatnet_downstream/worldfloods/worldfloods/20250927_PhiSatNetDownstream_unfrozen_5000/PhiSatNetDownstream_unfrozen_best.pt" #"/Data/phisatnet_new_decoder_last/lp/phisatnet_downstream/burned_area/burned_area/20250918_PhiSatNetDownstream_unfrozen_5000/PhiSatNetDownstream_unfrozen_best.pt"
onnx_path_encoder = "/home/gdaga/phi2FM/downstream/onnx/phisat2net_geoaware_best_encoder.onnx"
onnx_path_decoder = "/home/gdaga/phi2FM/downstream/onnx/phisat2net_geoaware_best_decoder.onnx"
onnx_path_model = "/home/gdaga/phi2FM/downstream/onnx/phisat2net_geoaware_best_model.onnx"
state_dict = torch.load(downstream_model_path)

new_state_dict = OrderedDict()

# Instantiate the model
input_dim = 8
output_dim = 3
img_size = 256
depths = [2, 2, 2, 2] # RICAMBIA QUESTO A SECONDA DEL MODELLO SCELTO ALLA FINE!!! O [2, 2, 2, 2], [2, 2, 8, 2] oppure [1, 1, 1, 1]
activation = "relu" # occhio perchè nelle ultime versioni ho provato relu
dims = [80, 160, 320, 640]
task = "segmentation"
model = PhiSatNetDownstream(
    pretrained_path=pretrained_model_path,
    task=task,
    input_dim=input_dim,
    output_dim=output_dim,
    depths=depths,
    dims=dims,
    img_size=img_size,
    freeze_body=False, 
    activation=activation
)
print(f"Loaded pretrained model from {pretrained_model_path}")
for key, value in state_dict.items():
    # Remove 'module.' prefix if it exists
    new_key = key.replace("module.", "")
    new_state_dict[new_key] = value

# Load the modified state dictionary into the model
model.load_state_dict(new_state_dict, strict=True)
# Model config (must match the checkpoint)

model.eval()

# Dummy input (batch size 1)
dummy_input_encoder = torch.randn(1, input_dim, img_size, img_size)
torch.onnx.export(
    model,
    dummy_input_encoder,
    onnx_path_model,
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    do_constant_folding=True,
    keep_initializers_as_inputs=False,
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX
    # dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)
print(f"ONNX model exported to {onnx_path_model}")

encoder_only = EncoderOnly(model.encoder, model.stem)
print(f"Parameters in encoder: {get_module_size(encoder_only):.2f} M")
# Export to ONNX
torch.onnx.export(
    encoder_only,
    dummy_input_encoder,
    onnx_path_encoder,
    input_names=["input"],
    output_names=["bottom_feats", "skip1", "skip2", "skip3", "skip4"],
    opset_version=11,
    do_constant_folding=True,
    keep_initializers_as_inputs=False,
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX
    # dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)
print(f"ONNX model encoder exported to {onnx_path_encoder}")

encoder_output, encoder_skips = encoder_only(dummy_input_encoder)
shapes = [out.shape for out in (encoder_output, *encoder_skips)]
print(f"Encoder output shapes: {shapes}")

dummy_input_decoder = torch.randn(80, 8, 1, 1)

decoder_only = DecoderOnly(model.decoder, model.head)
#([1, 640, 14, 14]), torch.Size([1, 80, 224, 224]), torch.Size([1, 160, 112, 112]), torch.Size([1, 320, 56, 56]), torch.Size([1, 640, 28, 28])]
# Dummy inputs for decoder
# bottom_feats: output from encoder (before bridge)
dummy_bottom_feats = torch.randn(1, 640, 14, 14)  # Adjust size based on your encoder output
# skips: skip connections from encoder at different resolutions
dummy_skip4 = torch.randn(1, 640, 28, 28)
dummy_skip3 = torch.randn(1, 320, 56, 56)
dummy_skip2 = torch.randn(1, 160, 112, 112) 
dummy_skip1 = torch.randn(1, 80, 224, 224)
print(f"Parameters in decoder: {get_module_size(decoder_only):.2f} M")
# Export decoder to ONNX
torch.onnx.export(
    decoder_only,
    (dummy_bottom_feats, dummy_skip1, dummy_skip2, dummy_skip3, dummy_skip4),
    onnx_path_decoder,
    input_names=["bottom_feats", "skip4", "skip3", "skip2", "skip1"],
    output_names=["seg_logits"],
    opset_version=11,
    dynamic_axes={
        "bottom_feats": {0: "batch"}, 
        "skip1": {0: "batch"}, 
        "skip2": {0: "batch"}, 
        "skip3": {0: "batch"}, 
        "seg_logits": {0: "batch"}
    }
)
print(f"ONNX decoder model exported to {onnx_path_decoder}")

