import os

import random
import warnings
import argparse
from datetime import date
import inspect
from collections import OrderedDict

import yaml
import numpy as np

# PyTorch imports
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torchinfo import summary

# Import project-specific utilities
from pretrain.models.utils_fm import get_phisat2_model
from utils import load_data, training_loops
from utils.training_utils import (
    read_yaml,
    module_memory_usage,
    dataloader_to_arrays,
    dataloader_to_tensors,
    convert_to_onnx,
    ddp_setup,
    ddp_cleanup,
)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def get_models(model_name, downstream_task, input_channels, input_size, fixed_task=None):
    if model_name.startswith('phisatnet_geoaware_'):
        model_size = model_name.split('_')[-1]
        return get_phisat2_model(model_size=model_size, downstream_task=downstream_task, fixed_task=fixed_task,
                                 unet_type='geoaware',
                                 input_dim=input_channels, output_dim=input_channels, img_size=input_size) # this line are kwargs

    if model_name.startswith('phisatnet_uniphi_'):
        model_size = model_name.split('_')[-1]
        return get_phisat2_model(model_size=model_size, downstream_task=downstream_task, fixed_task=fixed_task,
                                 unet_type='uniphi',
                                 input_dim=input_channels, output_dim=input_channels, img_size=input_size) # this line are kwargs

    else:
        raise ValueError(f"Model {model_name} not found")


def get_args_fn():
    # YAML Configuration
    parser_yaml = argparse.ArgumentParser(description='Experiment TestBed for Phi-Leo Foundation Model Project')
    parser_yaml.add_argument('-r', '--read_yaml', type=str, default='default_args.yml',
                             help='Path to YAML file with parameters')

    # Main Argument Parser
    parser = argparse.ArgumentParser(description='Experiment TestBed for Phi-Leo Foundation Model Project')

    # Experiment Settings
    parser.add_argument('--experiment_name', type=str, 
                        default=f'{date.today().strftime("%d%m%Y")}_experiment',
                        help='Experiment folder name')
    parser.add_argument('--downstream_task', type=str, default='pretrain',)
    parser.add_argument('--model_name', type=str, required=True, help='Select appropriate model')
    parser.add_argument('--output_path', type=str, default='results', help='Path to save experiment results')
    parser.add_argument('--fixed_task', type=str, default=None, choices=[None, 'reconstruction', 'coords', 'climate'], 
                        help='Fixed task for model training. Choose from None, reconstruction, coords, climate')

    # Training Parameters
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=250, help='Number of training epochs')
    parser.add_argument('--early_stop', type=int, default=50, help='Patience for early stopping')
    parser.add_argument('--train_mode', type=str, default='no', help='Run training or inference mode')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default=None, choices=[None, 'reduce_on_plateau', 'cosine_annealing'], 
                        help='Learning rate scheduler')
    parser.add_argument('--warmup', action="store_true", help='Enable 5-epoch linear warmup scheduler')
    parser.add_argument('--warmup_steps', type=int, default=5, help='Number of warmup steps')
    parser.add_argument('--warmup_gamma', type=int, default=10, help='Gamma factor for warmup scheduler')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')

    # Hardware & Parallelism
    parser.add_argument('--model_device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Device for model training')
    parser.add_argument('--generator_device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Device for data generation')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker threads for data loading')
    parser.add_argument('--data_parallel', type=str, default=None, help='Enable data parallelism')
    parser.add_argument('--device_ids', type=list, default=[0, 1, 2, 3], help='List of GPU device IDs')

    # Data Parameters
    parser.add_argument('--data_path', type=str, 
                        default='/home/phimultigpu/phileo_NFS/phileo_data/downstream/downstream_dataset_patches_np/',
                        help='Path to dataset')
    parser.add_argument('--input_channels', type=int, default=10, help='Number of input channels')
    parser.add_argument('--input_size', type=int, required=True, default=128, help='Input image size')
    parser.add_argument('--n_shot', type=int, default=None, 
                        help='Number of samples from specific geographic regions')
    parser.add_argument('--split_ratio', type=float, default=None, 
                        help='Percentage of data used from specified geographic regions')
    parser.add_argument('--augmentations', action="store_true", help='Enable data augmentations')
    parser.add_argument('--only_get_datasets', action="store_true", help='Only load datasets without training')

    # Visualization
    parser.add_argument('--vis_val', action="store_true", help='Enable saving of intermediate visualization plots')

    return parser, parser_yaml


def get_args():
    parser, parser_yaml = get_args_fn()
    args_yaml, remainder = parser_yaml.parse_known_args()
    
    if args_yaml.read_yaml is not None:
        # print(f"WARNING: overwriting all parameters with defaults stored in {args_yaml.read_yaml}")
        args = read_yaml(args_yaml.read_yaml)
    else:
        args = parser.parse_args()

    return args






def main(
    experiment_name: str,
    downstream_task: str,
    model_name: str,
    augmentations: bool,
    batch_size: int,
    model_device: str,
    generator_device: str,
    num_workers: int,
    early_stop: int,
    epochs: int,
    input_channels: int,
    input_size: int,
    lr: float,
    lr_scheduler: str,
    n_shot: int,
    split_ratio: float,
    vis_val: bool,
    warmup: bool,
    warmup_steps: int,
    warmup_gamma: int,
    data_path: str,
    train_mode: str,
    output_path: str,
    data_parallel: str,
    device_ids: list,
    only_get_datasets: bool,
    min_lr: float,
    fixed_task: str,
    train_precision: str,
    val_precision: str,
):
    """
    Main function to train, test, and perform inference on a model.
    """
    
    # -----------------------------------------------------------------------
    # 1. Handle Distributed Data Parallel (DDP) or DataParallel (DP) setup
    # -----------------------------------------------------------------------
    if data_parallel == 'DDP':
        world_rank, local_rank, world_size = ddp_setup()
        device = torch.device(f'cuda:{device_ids[local_rank]}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device)
        model_device, generator_device = device, 'cpu'
        print(f'Using DDP: rank {world_rank}/{world_size}, device {device}')
    else:
        world_rank, world_size = 0, 1
        if model_device == 'cuda':
            model_device = f'cuda:{device_ids[0]}' if device_ids else 'cuda'
        torch.set_default_device(model_device)
        generator_device = model_device
        print(f'Device (not using DDP): {model_device}')

    if torch.cuda.device_count() > 1 and world_rank == 0:
        num_gpus = torch.cuda.device_count() if device_ids is None else len(device_ids)
        print(f"Let's use {num_gpus} GPUs!")



    # -----------------------------------------------------------------------
    # 2. DEFINE THE MODEL
    # -----------------------------------------------------------------------
    model = get_models(model_name, downstream_task, input_channels, input_size, fixed_task)
    NAME = model.__class__.__name__

    # Parallelize model (DP or DDP) and print model summary
    if data_parallel == 'DP':
        model = nn.DataParallel(model, device_ids=device_ids).to(model_device)
    elif data_parallel == 'DDP':
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(model_device)
        model = DDP(model, device_ids=[model_device], output_device=model_device)

    # Print model summary and module sizes
    if world_rank == 0:
        model_summary = summary(model, input_size=(batch_size, input_channels, input_size, input_size))
        if model_device == 'cpu':
            model.to(model_device)
            print('Model moved back to CPU after summary') # sometimes summary moves model to GPU if available

        valid_modules = ["module", "model", "encoder", "decoder", "module.encoder", "module.decoder"]

        # Filter out invalid ones
        existing_modules = [
            name for name in valid_modules 
            if hasattr(model, name) or (lambda n: hasattr(model, "get_submodule") and hasattr(model, n) and model.get_submodule(n))(name)
        ]
        
        # Print memory usage only for valid modules
        print(module_memory_usage(model))
        print('-------------------')
        for module_name in existing_modules:
            print(module_memory_usage(model, module_name))
            print('-------------------')

    else:
        model_summary = None


    # -----------------------------------------------------------------------
    # 3. Construct output folder
    # -----------------------------------------------------------------------
    current_date = date.today().strftime('%Y%m%d')
    split_string = str(split_ratio) if split_ratio is not None else str(n_shot)
    OUTPUT_FOLDER = os.path.join(
        output_path,
        experiment_name,
        f"{current_date}_{NAME}_{split_string}"
    )
    if fixed_task is not None:
        OUTPUT_FOLDER += f"_{fixed_task}"

    if world_rank == 0:
        print(f"Output folder: {OUTPUT_FOLDER}")


    # -----------------------------------------------------------------------
    # 4. Load datasets
    # -----------------------------------------------------------------------
    
    # Ensure data partition protocol is valid
    assert n_shot is not None or split_ratio is not None, "Please define data partition protocol!"
    # XOR check: exactly one of (n_shot, split_ratio) must be used
    assert isinstance(n_shot, int) ^ isinstance(split_ratio, float), "n_shot cannot be used with split_ratio!"

    if world_rank == 0:
        print(f"Fixed task: {fixed_task}")
        print('--------------------------------')
        partition_type = 'n_shot' if n_shot is not None else 'split_ratio'
        partition_value = n_shot if n_shot is not None else split_ratio
        print(f"{partition_type}: {partition_value}")
        print('--------------------------------')

    # Load datasets
    dl_train, dl_val, dl_test, dl_inference = load_data.load_foundation_data(
        lmdb_path_train=f'{data_path}/train.lmdb',
        lmdb_path_val=f'{data_path}/val.lmdb',
        lmdb_path_test=f'{data_path}/test.lmdb',
        lmdb_path_inference=f'{data_path}/test.lmdb',
        with_augmentations=augmentations,
        num_workers=num_workers,
        batch_size=batch_size,
        device_dataset='cpu',
        device_dataloader=generator_device,
        input_size=input_size,
        fixed_task=fixed_task,
        use_ddp=(data_parallel == 'DDP'),
        rank=world_rank,
        world_size=world_size,
        split_ratio=split_ratio,
    )

    # Print dataset sizes
    if world_rank == 0:
        print(f"Dataset sizes: train={len(dl_train)}, val={len(dl_val)}, test={len(dl_test)}, inference={len(dl_inference)}")
        print(f'Augmentations: {augmentations}, Data Parallel: {data_parallel}, generator_device: {generator_device}')
        print(f'Training on: {model_name}')
        print('--'*30)

    if only_get_datasets:
        return dl_train, dl_val, dl_test, dl_inference

    # -----------------------------------------------------------------------
    # 6. Initialize the trainer
    # -----------------------------------------------------------------------
    
    # Adjust LR for warmup
    init_lr = lr
    assert (min_lr is None) != (warmup_gamma is None), "min_lr and warmup_gamma cannot be used together"
    if warmup and warmup_gamma is not None:
        lr /= 10 ** warmup_steps  # Decrease LR by 10^(warmup_steps)

    trainer = training_loops.TrainFoundation(
        epochs=epochs,
        lr=lr,
        model=model,
        device=model_device,
        lr_scheduler=lr_scheduler,
        warmup=warmup,
        early_stop=early_stop,
        train_loader=dl_train,
        val_loader=dl_val,
        test_loader=dl_test,
        inference_loader=dl_inference,
        name=NAME,
        out_folder=OUTPUT_FOLDER,
        visualise_validation=vis_val,
        warmup_steps=warmup_steps,
        warmup_gamma=warmup_gamma,
        save_info_vars=(model_summary, n_shot, split_ratio, warmup, init_lr),
        apply_zoom=False,
        fixed_task=fixed_task,
        rank=world_rank,
        min_lr=min_lr,
        train_precision=train_precision,
        val_precision=val_precision,
    )
    



    # -----------------------------------------------------------------------
    # 7. Training / testing / inference workflow
    # -----------------------------------------------------------------------

    # import pdb; pdb.set_trace()

    if train_mode == 'train_test_inference':
        trainer.train()
        trainer.test()
        trainer.save_info(
            model_summary=model_summary,
            n_shot=n_shot,
            p_split=split_ratio,
            warmup=warmup,
            lr=init_lr
        )
        trainer.inference()

    elif train_mode == 'train_test':
        trainer.train()
        trainer.test()
        print('Saving model summary')
        trainer.save_info(
            model_summary=model_summary,
            n_shot=n_shot,
            p_split=split_ratio,
            warmup=warmup,
            lr=init_lr
        )

    elif train_mode == 'test':
        trainer.test()

    elif train_mode == 'inference':
        trainer.inference()

    elif train_mode == 'train_inference':
        trainer.train()
        trainer.inference()

    else:
        raise ValueError(
            "train_mode should be one of [train_test_inference, train_test, inference, train_inference], "
            f"got {train_mode}"
        )



    # -----------------------------------------------------------------------
    # 8. Finish script. Save parameters to YAML and cleanup if DDP
    # -----------------------------------------------------------------------

    # Save parameters to YAML
    sig = inspect.signature(main)
    input_params = {key: value for key, value in locals().items() if key in sig.parameters}

    yaml_file = f'{experiment_name}_{date.today().strftime("%d%m%Y")}_{NAME}.yaml'
    yaml_file = yaml_file.replace('/', '_')
    params_path = os.path.join(OUTPUT_FOLDER, yaml_file)

    with open(params_path, 'w') as file:
        yaml.dump(input_params, file)

    # Cleanup DDP
    if data_parallel == 'DDP':
        ddp_cleanup()

    if world_rank == 0:
        print(f"See results of experiment in {OUTPUT_FOLDER}")




if __name__ == "__main__":
    
    print('Starting training_script.py')
    pid = os.getpid()
    print(f"Script started with PID: {pid}")

    args = get_args()

    main(**vars(args))

    print('Finished')