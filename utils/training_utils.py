import os
import yaml
import multiprocessing

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist




class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, channels, reduction=16, activation="relu"):
        super().__init__()
        reduction = max(1, channels // reduction)
        self.reduction = reduction
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // self.reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)

        return x * y.expand_as(x)


def get_activation(activation_name):
    if activation_name == "relu":
        return nn.ReLU6(inplace=False)
    elif isinstance(activation_name, torch.nn.modules.activation.ReLU6):
        return activation_name

    elif activation_name == "gelu":
        return nn.GELU()
    elif isinstance(activation_name, torch.nn.modules.activation.GELU):
        return activation_name

    elif activation_name == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    elif isinstance(activation_name, torch.nn.modules.activation.LeakyReLU):
        return activation_name

    elif activation_name == "prelu":
        return nn.PReLU()
    elif isinstance(activation_name, torch.nn.modules.activation.PReLU):
        return activation_name

    elif activation_name == "selu":
        return nn.SELU(inplace=True)
    elif isinstance(activation_name, torch.nn.modules.activation.SELU):
        return activation_name

    elif activation_name == "sigmoid":
        return nn.Sigmoid()
    elif isinstance(activation_name, torch.nn.modules.activation.Sigmoid):
        return activation_name

    elif activation_name == "tanh":
        return nn.Tanh()
    elif isinstance(activation_name, torch.nn.modules.activation.Tanh):
        return activation_name

    elif activation_name == "mish":
        return nn.Mish()
    elif isinstance(activation_name, torch.nn.modules.activation.Mish):
        return activation_name
    else:
        raise ValueError(f"activation must be one of leaky_relu, prelu, selu, gelu, sigmoid, tanh, relu. Got: {activation_name}")


def get_normalization(normalization_name, num_channels, num_groups=32, dims=2):
    if normalization_name == "batch":
        if dims == 1:
            return nn.BatchNorm1d(num_channels)
        elif dims == 2:
            return nn.BatchNorm2d(num_channels)
        elif dims == 3:
            return nn.BatchNorm3d(num_channels)
    elif normalization_name == "instance":
        if dims == 1:
            return nn.InstanceNorm1d(num_channels)
        elif dims == 2:
            return nn.InstanceNorm2d(num_channels)
        elif dims == 3:
            return nn.InstanceNorm3d(num_channels)
    elif normalization_name == "layer":
        # return LayerNorm(num_channels)
        return nn.LayerNorm(num_channels)
    elif normalization_name == "group":
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif normalization_name == "bcn":
        if dims == 1:
            return nn.Sequential(
                nn.BatchNorm1d(num_channels),
                nn.GroupNorm(1, num_channels)
            )
        elif dims == 2:
            return nn.Sequential(
                nn.BatchNorm2d(num_channels),
                nn.GroupNorm(1, num_channels)
            )
        elif dims == 3:
            return nn.Sequential(
                nn.BatchNorm3d(num_channels),
                nn.GroupNorm(1, num_channels)
            )    
    elif normalization_name == "none":
        return nn.Identity()
    else:
        raise ValueError(f"normalization must be one of batch, instance, layer, group, none. Got: {normalization_name}")


class AttrDict(dict):

    def __init__(self, *args, **kwargs):

        super(AttrDict, self).__init__(*args, **kwargs)

        self.__dict__ = self


def read_yaml(path):

    f = open(path)

    params = yaml.load(f, Loader=yaml.Loader)

    return AttrDict(params)






def module_memory_usage(model, part=None, min_size=0):
    """
    Prints the memory usage of each module in the model or a specified part of the model.
    Additionally, when a part is specified, it prints the total memory usage of that part.

    Args:
        model (nn.Module): The PyTorch model to inspect.
        part (str, optional): The name of the module to inspect. 
                            Supports nested modules using dot notation (e.g., 'encoder.linear_encode').
                            If None, all top-level modules are inspected.
        min_size (int, optional): Minimum memory size (in MB) to display a module.
    """
    def bytes_to_mb(bytes_size):
        return bytes_size / 1_048_576  # Convert bytes to megabytes

    def get_module_by_name(model, name):
        """
        Retrieves a submodule from the model based on a dot-separated module name.

        Args:
            model (nn.Module): The PyTorch model.
            name (str): Dot-separated module name.

        Returns:
            nn.Module: The retrieved submodule.

        Raises:
            AttributeError: If the module name is invalid.
        """
        if not name:
            return model
        return model.get_submodule(name)

    def print_module_memory(module, module_name, indent=0):
        prefix = "    " * indent
        total_params = sum(p.numel() for p in module.parameters())
        total_size_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
        total_size_mb = bytes_to_mb(total_size_bytes)
        if total_size_mb < min_size:
            return
        print(f"{prefix}{module_name}:")
        print(f"{prefix}  Total parameters: {total_params}")
        print(f"{prefix}  Memory usage: {total_size_mb:.4f} MB")

    if part:
        try:
            target_module = get_module_by_name(model, part)
        except AttributeError:
            print(f"Error: '{part}' is not a valid module name in the model.")
            return

        print(f"Memory usage for module '{part}':")
        submodules = list(target_module.named_children())

        total_size_bytes = 0
        for sub_name, sub_module in submodules:
            print_module_memory(sub_module, sub_name, indent=1)
            submodule_size = sum(p.numel() * p.element_size() for p in sub_module.parameters())
            total_size_bytes += submodule_size

        # Calculate exclusive parameters (parameters not part of any submodules)
        submodule_param_ids = set()
        for _, sub_module in submodules:
            for p in sub_module.parameters():
                submodule_param_ids.add(id(p))

        exclusive_params = [p for p in target_module.parameters() if id(p) not in submodule_param_ids]

        if exclusive_params:
            exclusive_params_num = sum(p.numel() for p in exclusive_params)
            exclusive_size_bytes = sum(p.numel() * p.element_size() for p in exclusive_params)
            exclusive_size_mb = bytes_to_mb(exclusive_size_bytes)
            print(f"    {part} exclusive parameters:")
            print(f"      Total parameters: {exclusive_params_num}")
            print(f"      Memory usage: {exclusive_size_mb:.4f} MB")
            total_size_bytes += exclusive_size_bytes

        total_size_mb = bytes_to_mb(total_size_bytes)
        print(f"Total memory usage of module '{part}': {total_size_mb:.4f} MB")
    else:
        print("Memory usage per top-level module:")
        total_model_memory = 0
        for name, module in model.named_children():
            total_params = sum(p.numel() for p in module.parameters())
            total_size_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
            total_size_mb = bytes_to_mb(total_size_bytes)
            if total_size_mb < min_size:
                continue
            print(f"{name}:")
            print(f"  Total parameters: {total_params}")
            print(f"  Memory usage: {total_size_mb:.4f} MB")
            total_model_memory += total_size_bytes

        total_model_memory_mb = bytes_to_mb(total_model_memory)
        print(f"Total memory usage of the model: {total_model_memory_mb:.4f} MB")



def dataloader_to_arrays(dataloader, device='cpu'):
    all_x = []
    all_y = {}

    # Iterate over all batches in the DataLoader
    for batch_idx, (x, y) in enumerate(dataloader):
        # Move tensors to the specified device
        x = x.to(device)
        
        # Detach and convert x to a NumPy array
        all_x.append(x.detach().cpu().numpy())
        
        # Process y if it's a dictionary
        if isinstance(y, dict):
            for key, value in y.items():
                value = value.to(device).detach().cpu().numpy()
                if key not in all_y:
                    all_y[key] = []
                all_y[key].append(value)
        else:
            y = y.to(device).detach().cpu().numpy()
            if 'default' not in all_y:
                all_y['default'] = []
            all_y['default'].append(y)
        
        print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")
    
    # Concatenate all x batches along the first dimension (batch size)
    all_x = np.concatenate(all_x, axis=0)

    # Concatenate all y batches for each key in the dictionary
    for key in all_y:
        all_y[key] = np.concatenate(all_y[key], axis=0)
    
    return all_x, all_y


def dataloader_to_tensors(dataloader, device='cpu'):
    all_x = []
    all_y = {}

    # Iterate over all batches in the DataLoader
    for batch_idx, (x, y) in enumerate(dataloader):
        # Move tensors to the specified device
        x = x.to(device)
        all_x.append(x)

        # Process y if it's a dictionary
        if isinstance(y, dict):
            for key, value in y.items():
                value = value.to(device)
                if key not in all_y:
                    all_y[key] = []
                all_y[key].append(value)
        else:
            y = y.to(device)
            if 'default' not in all_y:
                all_y['default'] = []
            all_y['default'].append(y)

        print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")
    
    # Concatenate all x tensors along the first dimension (batch size)
    all_x = torch.cat(all_x, dim=0)

    # Concatenate all y tensors for each key in the dictionary
    for key in all_y:
        all_y[key] = torch.cat(all_y[key], dim=0)
    
    return all_x, all_y

def convert_to_onnx(trainer, batch_size=32, input_size=128, num_channels=10, onnx_path = "geoaware_n50.onnx"):
    dummy_input = torch.randn(batch_size, num_channels, input_size, input_size)
    trainer.model.eval()
    torch.onnx.export(trainer.model, dummy_input, onnx_path, opset_version=10)

    # dummy_input = torch.randn(batch_size, 8, input_size, input_size)
    # torch.onnx.export(model, dummy_input, "model.onnx", opset_version=10)


def ddp_setup():
    """
    Set up DistributedDataParallel training using environment variables.
    """
    world_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    os.environ["OMP_NUM_THREADS"] = str(max(1, multiprocessing.cpu_count() // world_size))
    
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=world_rank
    )
    torch.cuda.set_device(local_rank)  # Bind the current process to the local GPU
    return world_rank, local_rank, world_size



def ddp_cleanup():
    """
    Cleanup the distributed process group.
    """
    dist.destroy_process_group()



