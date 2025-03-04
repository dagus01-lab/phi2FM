import os
import yaml

import torch
# torch.autograd.detect_anomaly(check_nan=True)

from torchinfo import summary

import numpy as np
import random
import inspect
from collections import OrderedDict


import torch.nn as nn
from datetime import date
import argparse

from torch.nn.parallel import DistributedDataParallel as DDP



import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from models.model_Baseline import BaselineNet
from models.model_CoreCNN_versions import CoreUnet_nano, CoreUnet_tiny, CoreUnet_base, CoreUnet_large, CoreUnet_huge, Core_nano, CoreUnetGeolocation_nano
from models.model_Mixer_versions import Mixer_nano, Mixer_tiny, Mixer_base, Mixer_large, Mixer_huge
from models.model_LinearViT_versions import LinearViT_base, LinearViT_large, LinearViT_huge
from models.model_AutoEncoderViT_versions import AutoencoderViT_base, AutoencoderViT_large, AutoencoderViT_huge
from models.model_GeoAwarePretrained import MixerGeoPretrained, get_mixer_kwargs, get_core_encoder_kwargs, CoreEncoderGeoPretrained, CoreEncoderGeoPretrained_combined, CoreEncoderGeoAutoEncoder
from models.model_GeoAwarePretrained_classifier import CoreEncoderGeoPretrained_Classifier
from models.model_AutoEncoderViTPretrained import vit_cnn, vit_cnn_gc, vit_large, get_core_decoder_kwargs
from models.model_AutoEncoderViTPretrained_wSkip import vit_cnn_wSkip, vit_cnn_gc_wSkip, vit_large_wSkip
from models.model_AutoEncoderViTPretrained_classifier import vit_cnn_classifier, vit_cnn_gc_classifier
from models.model_CoreVAE import CoreVAE_nano
from models.model_SatMAE import satmae_vit_cnn
from models.models_Prithvi import prithvi
from models.model_Seco import seasonal_contrast
from models.model_Resnet50 import resnet
from models.code_phileo_precursor.model_foundation_local_rev2 import PhileoPrecursor

from pretrain.models.utils_fm import get_phisat2_model
from downstream.models.phisatnet_downstream import PhiSatNetDownstream


from utils import data_protocol
from utils import load_data
from utils import training_loops
from utils.training_utils import read_yaml
from utils.utils import module_memory_usage, dataloader_to_arrays, dataloader_to_tensors, convert_to_onnx, ddp_setup, ddp_cleanup

torch.manual_seed(123456)
CNN_LIST = ['baseline_cnn', 'core_unet_nano','core_unet_tiny','core_unet_base', 'core_unet_large', 'core_unet_huge',
            'core_vae_nano', 'resnet_imagenet', 'resnet', 'core_encoder_nano', 'resnet_imagenet_classifier',
            'core_unet_geolocation_nano']

VIT_CNN_LIST = ['vit_cnn_base', 'vit_cnn_base_wSkip']

MIXER_LIST = ['mixer_nano', 'mixer_tiny', 'mixer_base', 'mixer_large', 'mixer_huge']

VIT_LIST = ['linear_vit_base', 'linear_vit_larger', 'linear_vit_huge',
            'autoencoder_vit_base', 'autoencoder_vit_large', 'autoencoder_vit_huge']

CNN_PRETRAINED_LIST = ['GeoAware_core_nano', 'GeoAware_core_tiny', 'GeoAware_mixer_nano', 'GeoAware_mixer_tiny',
                       'GeoAware_contrastive_core_nano', 'GeoAware_mh_pred_core_nano', 'GeoAware_combined_core_nano',
                       'GeoAware_core_autoencoder_nano', 'seasonal_contrast',
                       'GeoAware_core_nano_classifier', 'GeoAware_contrastive_core_nano_classifier',
                       'GeoAware_mh_pred_core_nano_classifier', 'seasonal_contrast_classifier',
                       'phileo_precursor', 'phisatnet', 'phisatnet_classifier'
                       ]

VIT_CNN_PRETRAINED_LIST = ['prithvi', 'vit_cnn', 'vit_cnn_gc', 'SatMAE', 'SatMAE_classifier', 'vit_cnn_gc_classifier',
                           'vit_cnn_classifier', 'prithvi_classifier', 'vit_cnn_wSkip', 'vit_cnn_gc_wSkip']

MODELS_224 = ['seasonal_contrast', 'resnet_imagenet', 'resnet', 'seasonal_contrast_classifier', 'resnet_imagenet_classifier', 'phisatnet', 'phisatnet_classifier']
MODELS_224_r30 = ['prithvi', 'prithvi_classifier']

MODEL_LIST = CNN_LIST + MIXER_LIST + VIT_LIST + CNN_PRETRAINED_LIST + VIT_CNN_LIST + VIT_CNN_PRETRAINED_LIST
DOWNSTREAM_LIST = ['lc', 'building', 'roads', 'lc_classification', 'building_classification', 'roads_classification']


def get_trainer(model_name, downstream_task, epochs, lr, model, device, lr_scheduler, warmup, early_stop, dl_train,
                dl_val, dl_test, dl_inference, NAME, OUTPUT_FOLDER, vis_val, warmup_steps, warmup_gamma, pos_weight, 
                weights, save_info_vars, rank=None, min_lr=None):
    
    if model_name in (CNN_LIST + MIXER_LIST + VIT_CNN_LIST + CNN_PRETRAINED_LIST + VIT_CNN_PRETRAINED_LIST):
        if downstream_task == 'roads' or downstream_task == 'building':
            trainer = training_loops.TrainBase(epochs=epochs, lr=lr, model=model, device=device,
                                               lr_scheduler=lr_scheduler, warmup=warmup, early_stop=early_stop,
                                               train_loader=dl_train,
                                               val_loader=dl_val, test_loader=dl_test, inference_loader=dl_inference, name=NAME,
                                               out_folder=OUTPUT_FOLDER, visualise_validation=vis_val,
                                               warmup_steps=warmup_steps, warmup_gamma=warmup_gamma, save_info_vars=save_info_vars)
        elif downstream_task == 'coords':
            trainer = training_loops.TrainGeoLocate(epochs=epochs, lr=lr, model=model, device=device,
                                                    lr_scheduler=lr_scheduler, warmup=warmup, early_stop=early_stop,
                                                    train_loader=dl_train,
                                                    val_loader=dl_val, test_loader=dl_test, inference_loader=dl_inference, name=NAME,
                                                    out_folder=OUTPUT_FOLDER, visualise_validation=vis_val,
                                                    warmup_steps=warmup_steps, warmup_gamma=warmup_gamma, save_info_vars=save_info_vars)
            
        elif downstream_task == 'lc':
            trainer = training_loops.TrainLandCover(epochs=epochs, lr=lr, model=model, device=device,
                                                    lr_scheduler=lr_scheduler, warmup=warmup, early_stop=early_stop,
                                                    train_loader=dl_train,
                                                    val_loader=dl_val, test_loader=dl_test, inference_loader=dl_inference, name=NAME,
                                                    out_folder=OUTPUT_FOLDER, visualise_validation=vis_val,
                                                    warmup_steps=warmup_steps, warmup_gamma=warmup_gamma, save_info_vars=save_info_vars)
        elif downstream_task == 'building_classification':
            trainer = training_loops.TrainClassificationBuildings(epochs=epochs, lr=lr, model=model, device=device,
                                                                  lr_scheduler=lr_scheduler, warmup=warmup, early_stop=early_stop,
                                                                  train_loader=dl_train,
                                                                  val_loader=dl_val, test_loader=dl_test, inference_loader=dl_inference, name=NAME,
                                                                  out_folder=OUTPUT_FOLDER, visualise_validation=vis_val,
                                                                  warmup_steps=warmup_steps, warmup_gamma=warmup_gamma, weights=weights,
                                                                  save_info_vars=save_info_vars)

        elif downstream_task == 'lc_classification':
            trainer = training_loops.TrainClassificationLC(epochs=epochs, lr=lr, model=model, device=device,
                                                           lr_scheduler=lr_scheduler, warmup=warmup, early_stop=early_stop,
                                                           train_loader=dl_train,
                                                           val_loader=dl_val, test_loader=dl_test, inference_loader=dl_inference, name=NAME,
                                                           out_folder=OUTPUT_FOLDER, visualise_validation=vis_val,
                                                           warmup_steps=warmup_steps, warmup_gamma=warmup_gamma, pos_weight=pos_weight,
                                                           save_info_vars=save_info_vars)

        elif downstream_task == 'roads_classification':
            trainer = training_loops.TrainClassificationRoads(epochs=epochs, lr=lr, model=model, device=device,
                                                           lr_scheduler=lr_scheduler, warmup=warmup, early_stop=early_stop,
                                                           train_loader=dl_train,
                                                           val_loader=dl_val, test_loader=dl_test, inference_loader=dl_inference, name=NAME,
                                                           out_folder=OUTPUT_FOLDER, visualise_validation=vis_val,
                                                           warmup_steps=warmup_steps, warmup_gamma=warmup_gamma,
                                                           save_info_vars=save_info_vars)

    elif model_name in (VIT_LIST):
        if downstream_task == 'roads' or downstream_task == 'building':
            trainer = training_loops.TrainViT(epochs=epochs, lr=lr, model=model, device=device,
                                              lr_scheduler=lr_scheduler, warmup=warmup, early_stop=early_stop, train_loader=dl_train,
                                              val_loader=dl_val, test_loader=dl_test, inference_loader=dl_inference, name=NAME,
                                              out_folder=OUTPUT_FOLDER, visualise_validation=vis_val,
                                              warmup_steps=warmup_steps, warmup_gamma=warmup_gamma)

        elif downstream_task == 'lc':
            trainer = training_loops.TrainViTLandCover(epochs=epochs, lr=lr, model=model, device=device,
                                                       lr_scheduler=lr_scheduler, warmup=warmup, early_stop=early_stop,
                                                       train_loader=dl_train,
                                                       val_loader=dl_val, test_loader=dl_test, inference_loader=dl_inference, name=NAME,
                                                       out_folder=OUTPUT_FOLDER, visualise_validation=vis_val,
                                                       warmup_steps=warmup_steps, warmup_gamma=warmup_gamma)

    if model_name == 'core_vae_nano':
        trainer = training_loops.TrainVAE(epochs=epochs, lr=lr, model=model, device=device,
                                          lr_scheduler=lr_scheduler, warmup=warmup, early_stop=early_stop,
                                          train_loader=dl_train,
                                          val_loader=dl_val, test_loader=dl_test, inference_loader=dl_inference, name=NAME,
                                          out_folder=OUTPUT_FOLDER, visualise_validation=vis_val,
                                          warmup_steps=warmup_steps, warmup_gamma=warmup_gamma)

    return trainer


def get_models(model_name, input_channels, output_channels, input_size):
    if model_name == 'baseline_cnn':
        return BaselineNet(input_dim=input_channels, output_dim=output_channels)
    elif model_name == 'core_unet_nano':
        return CoreUnet_nano(input_dim=input_channels, output_dim=output_channels)
    elif model_name == 'core_encoder_nano':
        return Core_nano(input_dim=input_channels, output_dim=output_channels)
    elif model_name == 'core_unet_tiny':
        return CoreUnet_tiny(input_dim=input_channels, output_dim=output_channels)
    elif model_name == 'core_unet_base':
        return CoreUnet_base(input_dim=input_channels, output_dim=output_channels)
    elif model_name == 'core_unet_large':
        return CoreUnet_large(input_dim=input_channels, output_dim=output_channels)
    elif model_name == 'core_unet_huge':
        return CoreUnet_huge(input_dim=input_channels, output_dim=output_channels)
    elif model_name == 'mixer_nano':
        return Mixer_nano(chw=(input_channels, input_size, input_size),
                          output_dim=output_channels)
    elif model_name == 'mixer_tiny':
        return Mixer_tiny(chw=(input_channels, input_size, input_size),
                          output_dim=output_channels)
    elif model_name == 'mixer_base':
        return Mixer_base(chw=(input_channels, input_size, input_size),
                          output_dim=output_channels)
    elif model_name == 'mixer_large':
        return Mixer_large(chw=(input_channels, input_size, input_size),
                           output_dim=output_channels)
    elif model_name == 'mixer_huge':
        return Mixer_huge(chw=(input_channels, input_size, input_size),
                          output_dim=output_channels)
    elif model_name == 'linear_vit_base':
        return LinearViT_base(chw=(input_channels, input_size, input_size),
                              output_dim=output_channels)
    elif model_name == 'linear_vit_large':
        return LinearViT_large(chw=(input_channels, input_size, input_size),
                               output_dim=output_channels)
    elif model_name == 'linear_vit_huge':
        return LinearViT_huge(chw=(input_channels, input_size, input_size),
                              output_dim=output_channels)
    elif model_name == 'autoencoder_vit_base':
        return AutoencoderViT_base(chw=(input_channels, input_size, input_size),
                                   output_dim=output_channels)
    elif model_name == 'autoencoder_vit_large':
        return AutoencoderViT_large(chw=(input_channels, input_size, input_size),
                                    output_dim=output_channels)
    elif model_name == 'autoencoder_vit_huge':
        return AutoencoderViT_huge(chw=(input_channels, input_size, input_size),
                                   output_dim=output_channels)
    elif model_name == 'core_vae_nano':
        return CoreVAE_nano(input_dim=input_channels, output_dim=10)

    elif model_name == 'vit_cnn_base':
        return vit_large(chw=(input_channels, input_size, input_size),
                         output_dim=output_channels)
    elif model_name == 'vit_cnn_base_wSkip':
        return vit_large_wSkip(chw=(input_channels, input_size, input_size),
                         output_dim=output_channels)
    elif model_name == 'resnet_imagenet':
        resnet_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return resnet(imagenet_weights=True, **resnet_kwargs)
    elif model_name == 'resnet_imagenet_classifier':
        resnet_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return resnet(imagenet_weights=True, classifier=True, **resnet_kwargs)
    elif model_name == 'resnet':
        resnet_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return resnet(imagenet_weights=False, **resnet_kwargs)
    elif model_name == 'core_unet_geolocation_nano':
        return CoreUnetGeolocation_nano(input_dim=input_channels, output_dim=output_channels)


def get_models_pretrained(model_name, input_channels, output_channels, input_size, path_model_weights=None, freeze=False, device='cuda'):
    test_input = torch.rand((2,input_channels,input_size,input_size))
    
    if model_name == 'phisatnet' or model_name == 'phisatnet_classifier':
        core_kwargs = get_phisat2_model(model_size='xsmall', unet_type='geoaware')
        print(f'core_kwargs: {core_kwargs}')
        model = PhiSatNetDownstream(pretrained_path=path_model_weights, 
                                     task='segmentation' if model_name == 'phisatnet' else 'classification',
                                     input_dim=input_channels,
                                     output_dim=output_channels,
                                     freeze_body=freeze,
                                     img_size=input_size,
                                     **core_kwargs
                                    )
        model(test_input)
        return model

    if (model_name == 'GeoAware_core_nano' or model_name == 'GeoAware_contrastive_core_nano' or
            model_name == 'GeoAware_mh_pred_core_nano'):

        sd = torch.load(path_model_weights)
        core_kwargs = get_core_encoder_kwargs(output_dim=output_channels, input_dim=input_channels, core_size='core_nano', full_unet=True)
        model = CoreEncoderGeoPretrained(output_channels, checkpoint=sd, core_encoder_kwargs=core_kwargs, freeze_body=freeze)
        model(test_input)
        return model

    if (model_name == 'GeoAware_core_nano_classifier' or model_name == 'GeoAware_contrastive_core_nano_classifier' or
            model_name == 'GeoAware_mh_pred_core_nano_classifier'):

        sd = torch.load(path_model_weights)
        core_kwargs = get_core_encoder_kwargs(output_dim=output_channels, input_dim=input_channels, core_size='core_nano', full_unet=False)
        model = CoreEncoderGeoPretrained_Classifier(checkpoint=sd, core_encoder_kwargs=core_kwargs, freeze_body=freeze)
        model(test_input)
        return model

    if model_name == 'GeoAware_core_autoencoder_nano':
        sd = torch.load(path_model_weights)
        core_kwargs = get_core_encoder_kwargs(output_dim=output_channels, input_dim=input_channels, core_size='core_nano', full_unet=True)
        model = CoreEncoderGeoAutoEncoder(output_channels, checkpoint=sd, core_encoder_kwargs=core_kwargs, freeze_body=freeze)
        model(test_input)
        return model

    if model_name == 'GeoAware_combined_core_nano':
        sd_1 = torch.load(path_model_weights[0])
        sd_2 = torch.load(path_model_weights[1])
        core_kwargs = get_core_encoder_kwargs(output_dim=output_channels, input_dim=input_channels, core_size='core_nano')
        model = CoreEncoderGeoPretrained_combined(output_channels, checkpoint_1=sd_1, checkpoint_2=sd_2,
                                                  core_encoder_kwargs=core_kwargs)

        model(test_input)
        return model
    
    if model_name == 'GeoAware_core_tiny':
        sd = torch.load(path_model_weights)
        core_kwargs = get_core_encoder_kwargs(output_dim=output_channels, input_dim=input_channels, core_size='core_tiny', full_unet=True)
        model = CoreEncoderGeoPretrained(output_channels, checkpoint=sd, core_encoder_kwargs=core_kwargs, freeze_body=freeze)
        model(test_input)
        return model

    elif model_name == 'phileo_precursor':
        sd = torch.load(path_model_weights)
        core_kwargs = get_core_encoder_kwargs(output_dim=output_channels, input_dim=input_channels, core_size='core_nano')
        if 'norm' in core_kwargs.keys():
            core_kwargs.pop('norm')
        if 'padding' in core_kwargs.keys():
            core_kwargs.pop('padding')
        model = PhileoPrecursor(output_dim=output_channels, checkpoint=sd, core_encoder_kwargs=core_kwargs, freeze_body=freeze)
        model(test_input)
        return model

    if model_name == 'GeoAware_mixer_nano':
        sd = torch.load(path_model_weights)
        mixer_kwargs = get_mixer_kwargs(chw=(input_channels,input_size,input_size),output_dim=output_channels, mixer_size='mixer_nano')
        model =  MixerGeoPretrained(output_dim=output_channels, checkpoint=sd, mixer_kwargs=mixer_kwargs, freeze_body=freeze)
        model(test_input)
        return model 
    
    if model_name == 'GeoAware_mixer_tiny':
        sd = torch.load(path_model_weights)
        mixer_kwargs = get_mixer_kwargs(chw=(input_channels,input_size,input_size),output_dim=output_channels, mixer_size='mixer_tiny')
        model = MixerGeoPretrained(output_dim=output_channels, checkpoint=sd, mixer_kwargs=mixer_kwargs, freeze_body=freeze)
        model(test_input)
        return model 

    elif model_name == 'SatMAE':
        sd = torch.load(path_model_weights)
        satmae_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return satmae_vit_cnn(img_size=96, patch_size=8, in_chans=input_channels,
                              checkpoint=sd, freeze_body=freeze, classifier=False, **satmae_kwargs)

    elif model_name == 'SatMAE_classifier':
        sd = torch.load(path_model_weights)
        satmae_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return satmae_vit_cnn(img_size=96, patch_size=8, in_chans=input_channels,
                              checkpoint=sd, freeze_body=freeze, classifier=True, **satmae_kwargs)

    elif model_name == 'prithvi':
        sd = torch.load(path_model_weights, map_location=device)
        prithvi_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return prithvi(checkpoint=sd, freeze_body=freeze, **prithvi_kwargs)

    elif model_name == 'prithvi_classifier':
        sd = torch.load(path_model_weights, map_location=device)
        prithvi_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return prithvi(checkpoint=sd, freeze_body=freeze, classifier=True, **prithvi_kwargs)

    elif model_name == 'vit_cnn':
        sd = torch.load(path_model_weights, map_location=device)
        vit_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return vit_cnn(checkpoint=sd, freeze_body=freeze, **vit_kwargs)

    elif model_name == 'vit_cnn_wSkip':
        sd = torch.load(path_model_weights, map_location=device)
        vit_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return vit_cnn_wSkip(checkpoint=sd, freeze_body=freeze, **vit_kwargs)

    elif model_name == 'vit_cnn_classifier':
        sd = torch.load(path_model_weights, map_location=device)
        return vit_cnn_classifier(checkpoint=sd, freeze_body=freeze, output_dim=output_channels)

    elif model_name == 'vit_cnn_gc':
        sd = torch.load(path_model_weights, map_location=device)
        vit_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return vit_cnn_gc(checkpoint=sd, freeze_body=freeze, **vit_kwargs)

    elif model_name == 'vit_cnn_gc_wSkip':
        sd = torch.load(path_model_weights, map_location=device)
        vit_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return vit_cnn_gc_wSkip(checkpoint=sd, freeze_body=freeze, **vit_kwargs)

    elif model_name == 'vit_cnn_gc_classifier':
        sd = torch.load(path_model_weights, map_location=device)
        return vit_cnn_gc_classifier(checkpoint=sd, freeze_body=freeze, output_dim=output_channels)

    elif model_name == 'seasonal_contrast':
        seco_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return seasonal_contrast(checkpoint=path_model_weights, freeze_body=freeze,
                                 **seco_kwargs)

    elif model_name == 'seasonal_contrast_classifier':
        seco_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return seasonal_contrast(checkpoint=path_model_weights, freeze_body=freeze, classifier=True,
                                 **seco_kwargs)
    else:
        raise ValueError(f'Unknown model name: {model_name}')

def get_args():
    parser_yaml = argparse.ArgumentParser(description='Experiment TestBed for Phi-Leo Foundation Model Project')
    parser_yaml.add_argument('-r', '--read_yaml', dest='read_yaml', type=str, help='Path to YAML file with parameters', default='default_args.yml')

    parser = argparse.ArgumentParser(description='Experiment TestBed for Phi-Leo Foundation Model Project')
    parser.add_argument('--experiment_name', type=str, default=f'{date.today().strftime("%d%m%Y")}_experiment',
                        help='Experiment folder name')
    parser.add_argument('--model_name', type=str, choices=MODEL_LIST, required=True,
                        help='Select appropriate model')
    parser.add_argument('--lr', type=float, default=0.001, help='Set learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Set batch size')
    parser.add_argument('--epochs', type=int, default=250, help='Set training epochs')
    parser.add_argument('--early_stop', type=int, default=50, help='set training loop patience for early stopping')
    parser.add_argument('--lr_scheduler', type=str, default=None,
                        choices=[None, 'reduce_on_plateau', 'cosine_annealing'], help='select learning rate scheduler')
    parser.add_argument('--warmup', action="store_true", help='Enables linear 5 epoch warmup scheduler')
    parser.add_argument('--model_device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='select training device')
    parser.add_argument('--generator_device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='select training device')
    parser.add_argument('--num_workers', type=int, default=0, help='set number of workers')
    parser.add_argument('--vis_val', action="store_true", help='enable saving of intermediate visualization plots')
    parser.add_argument('--downstream_task', type=str, choices=DOWNSTREAM_LIST, required=True,
                        help='select downstream task')
    parser.add_argument('--input_channels', type=int, required=False, default=10, help='Define Number of input channels')
    parser.add_argument('--input_size', type=int, required=True, default=128, help='Define input size')
    parser.add_argument('--output_channels', type=int, required=True, default=1, help='Define Number of output channels')

    parser.add_argument('--regions', type=list, default=None, help='select regions to be included',
                        choices=[None, 'denmark-1', 'denmark-2', 'east-africa', 'egypt-1', 'eq-guinea', 'europe', 'ghana-1',
                                 'isreal-1', 'isreal-2', 'japan', 'nigeria', 'north-america', 'senegal', 'south-america',
                                 'tanzania-1', 'tanzania-2', 'tanzania-3', 'tanzania-4', 'tanzania-5', 'uganda-1'])
    parser.add_argument('--n_shot', type=int, default=None,
                        help='Loads n-samples of data from specified geographic regions')
    parser.add_argument('--split_ratio', type=float, default=None,
                        help='Loads a percentage of the data from specified geographic regions.')
    parser.add_argument('--augmentations', action="store_true", help='enables augmentations')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained weights')
    parser.add_argument('--freeze_pretrained', action="store_true", help='freeze pretrained model weights')
    parser.add_argument('--data_path_128_10m', type=str, default='/home/phimultigpu/phileo_NFS/phileo_data/downstream/downstream_dataset_patches_np/')
    parser.add_argument('--data_path_224_10m', type=str, default='/home/phimultigpu/phileo_NFS/phileo_data/downstream/downstream_dataset_patches_np_224/')
    parser.add_argument('--data_path_224_30m', type=str, default='/home/phimultigpu/phileo_NFS/phileo_data/downstream/downstream_dataset_patches_np_HLS/')
    parser.add_argument('--data_path_inference_128', type=str, default='/home/ccollado/2_phileo_fm/inference_folder/')
    parser.add_argument	('--data_path_inference_224', type=str, default='/home/ccollado/2_phileo_fm/inference_folder_224/')
    parser.add_argument('--additional_inference', type=str, default='no', help='run inference only')
    parser.add_argument('--downstream_model_path', type=str, default=None)
    parser.add_argument('--C', type=str, default='/home/phimultigpu/phileo_NFS/phileo_data/experiments')
    parser.add_argument('--data_parallel', type=str, default=None)
    parser.add_argument('--device_ids', type=list, default=[0, 1, 2, 3])
    parser.add_argument('--warmp_steps', type=int, default=5)
    parser.add_argument('--warmup_gamma', type=int, default=10)
    parser.add_argument('--pad_to_10_bands', type=bool, default=True)
    parser.add_argument('--min_lr', type=float, default=1e-6)



    return parser, parser_yaml




def main(experiment_name, downstream_task, model_name, augmentations, batch_size, model_device, generator_device, num_workers, early_stop, 
        epochs, input_channels, output_channels, input_size, lr, lr_scheduler, n_shot, split_ratio, regions, vis_val, warmup, warmp_steps, 
        warmup_gamma, pretrained_model_path, freeze_pretrained, data_path_128_10m, data_path_224_10m, data_path_224_30m, data_path_inference_128, 
        data_path_inference_224, train_mode, downstream_model_path, output_path, data_parallel, 
        device_ids, only_get_datasets, pad_to_10_bands, min_lr):
    """ 
    main script for PhilEO Bench. Used to run model training experiments with randomly initialized and pre-trained models on a number of downstream tasks. 
    The script handles dataset creation (based on data protocol options selected), data preprocessing (based on downstream task & model type) & model, training, validation and testing. 

    Parameters
    ----------
        experiment_name (str): Experiment name
        downstream_task (str): Select downstream task to test, validate and test on. Options: {DOWNSTREAM_LIST}
        model_name (str): Select model. Options:{MODEL_LIST}
        augmentations (bool, optional): Toggle on/off basic data augmentations (Rotation, Mirror, Noise). Defaults to False.
        batch_size (int, optional): Define training batch size. Defaults to 16.
        model_device (_type_, optional): Select model device. Defaults to torch.device('cuda' if torch.cuda.is_available() else 'cpu').
        generator_device (_type_, optional): Select dataloader device. Defaults to torch.device('cuda' if torch.cuda.is_available() else 'cpu').
        num_workers (int, optional): Select number of workers for dataloader. Defaults to 4.
        early_stop (int, optional):Define early stoping patience. Defaults to 25.
        epochs (int, optional): Define number of training epochs. Defaults to 250.
        input_channels (int, optional): Define number of data input channels. Defaults to 10.
        output_channels (int, optional): Define number of model output channels. Defaults to 1.
        input_size (int, optional): Define data input size. Defaults to 128.
        lr (float, optional): Define optimizer learning rate. Defaults to 0.001.
        lr_scheduler (str, optional): Define learning rate scheduler. Options: [None, 'reduce_on_plateau', 'cosine_annealing']. Defaults to None.
        n_shot (int, optional): Define dataset protocol - n samples per region. Defaults to None.
        split_ratio (float, optional): Define dataset protocol - percentage of full dataset. Defaults to 0.1.
        regions (list, optional): Select regions to include in training and test sets. If no regions are defined (None) all avalible regions will be included
                                  Options: [None, 'denmark-1', 'denmark-2', 'east-africa', 'egypt-1', 'eq-guinea', 'europe', 'ghana-1',
                                 'isreal-1', 'isreal-2', 'japan', 'nigeria', 'north-america', 'senegal', 'south-america',
                                 'tanzania-1', 'tanzania-2', 'tanzania-3', 'tanzania-4', 'tanzania-5', 'uganda-1'] Defaults to None.
        vis_val (bool, optional): If set to True data visulisations will be generated at each validation step. Defaults to True.
        warmup (bool, optional): If set to True a linear optimizer warmup phase will occour. Defaults to False.
        warmp_steps (int, optional): Define number of steps for linear warmup phase. Defaults to 5.
        warmup_gamma (int, optional): Define learning rate increase per step in linear warmup phase - new_lr = lr*gamma. Defaults to 10. N.B. initial lr is calulated as follows init_lr = lr/(gamma**warmup_steps)
        pretrained_model_path (str, optional): For pretrained models define the model weights path. Defaults to None.
        freeze_pretrained (bool, optional): If True pretrained encoder weights will be frozen during training. Defaults to None.
        data_path_128_10m (str, optional): Define data path for 128x128 10m resolution dataset. Defaults to None.
        data_path_224_10m (str, optional): Define data path for 224x224 10m resolution dataset. Defaults to None.
        data_path_224_30m (str, optional): Define data path for 224x224 30m resolution dataset. Defaults to None.
        data_path_inference_128 (str, optional): Define data path for inference data of size 128. Defaults to None.
        data_path_inference_224 (str, optional): Define data path for inference data of size 224. Defaults to None.
        train_mode (str, optional): Define if only inference should be run. Options: ['yes', 'no', 'only']. Defaults to None.
        downstream_model_path (str, optional): Define model path for inference. Defaults to None.
        output_path (str, optional): Define folder to save artifacts in. Defaults to None.
        data_parallel (str, optional): If set to True Model training will be parallized on multiple gpus. Defaults to None.
        device_ids (list, optional): Define GPU IDs to use for parallization. Defaults to None.
        only_get_datasets (bool, optional): If set to True only datasets will be created, but no training will occur. Defaults to False.
        pad_to_10_bands (bool, optional): If set to True data will be padded to 10 bands. Defaults to False.
        min_lr (float, optional): Define minimum learning rate for cosine annealing scheduler and warmup. Defaults to 1e-6.
    """

    # -----------------------------------------------------------------------
    # 1. Multi GPU Setup (DDP, DP, or False) -- DDP NOT FULLY IMPLEMENTED FOR DOWNSTREAM
    # -----------------------------------------------------------------------
    if data_parallel == 'DDP':
        world_rank, local_rank, world_size = ddp_setup()
        device = torch.device(f'cuda:{device_ids[local_rank]}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device)
        model_device, generator_device = device, 'cpu'
        print(f'Using DDP: rank {world_rank}/{world_size}, device {device}')
    else:
        world_rank, local_rank, world_size = 0, 0, 1
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
    
    # LOAD PRETRAINED MODEL
    if pretrained_model_path is not None:
        if world_rank == 0:
            print('model_name: ', model_name)
        assert model_name in (CNN_PRETRAINED_LIST + VIT_CNN_PRETRAINED_LIST), f"Pretrained weights were given but model {model_name} not found in list of pretrained models: {(CNN_PRETRAINED_LIST + VIT_CNN_PRETRAINED_LIST)}"
        assert freeze_pretrained is not None, f"When supplying a pretrained model 'freeze_pretrained' must be either True or False"
        model = get_models_pretrained(model_name, input_channels, output_channels, input_size, path_model_weights=pretrained_model_path, freeze=freeze_pretrained)
        if model_name == 'GeoAware_contrastive_core_nano' or model_name == 'GeoAware_contrastive_core_nano_classifier':
            NAME = model.__class__.__name__ +'_contrastive_frozen' if freeze_pretrained else model.__class__.__name__ +'_contrastive_unfrozen'
        elif model_name == 'GeoAware_mh_pred_core_nano' or model_name == 'GeoAware_mh_pred_core_nano_classifier':
            NAME = model.__class__.__name__ +'_mh_pred_frozen' if freeze_pretrained else model.__class__.__name__ +'_mh_pred_unfrozen'
        else:
            NAME = model.__class__.__name__ + '_frozen' if freeze_pretrained else model.__class__.__name__ + '_unfrozen'
        if world_rank == 0:
            print(f'Loaded pretrained model: {model_name} with {NAME} weights')

    # LOAD RANDOMLY INITIALIZED MODEL
    else:
        if freeze_pretrained:
            if world_rank == 0:
                print(f"Ignoring freeze_pretrained set to {freeze_pretrained} as no pretrained model was supplied")
        model = get_models(model_name, input_channels, output_channels, input_size)
        NAME = model.__class__.__name__
    
    # If want to load weights of full downstream model, not just a feature extractor
    if downstream_model_path:
        print('\n\n------------------------------------------------------------------------')
        print(f'WARNING: IGNORING pretrained_model_path. Inference model path given. Full downstream model will be loaded.')
        print('------------------------------------------------------------------------\n\n')
        
        assert model is not None, "This model implementation requires pretrained weights to be loaded first, even if they will be overwritten"
        
        state_dict = torch.load(downstream_model_path)
        if world_rank == 0:
            print(f'Loading inference model from {downstream_model_path}')

        new_state_dict = OrderedDict()
        
        for key, value in state_dict.items():
            # Remove 'module.' prefix if it exists
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value

        # Load the modified state dictionary into the model
        model.load_state_dict(new_state_dict, strict=True)

    # Parallelize model (DP or DDP) and print model summary
    if data_parallel == 'DP':
        model = nn.DataParallel(model, device_ids=device_ids).to(model_device)
    elif data_parallel == 'DDP':
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(model_device)
        model = DDP(model, device_ids=[model_device], output_device=model_device)
    
    # Print model summary and module sizes
    if world_rank == 0:
        input_sizes = {
            'SatMAE': (batch_size, input_channels, 96, 96),
            'SatMAE_classifier': (batch_size, input_channels, 96, 96),
            'prithvi': (batch_size, 6, 224, 224),
            'prithvi_classifier': (batch_size, 6, 224, 224),
            'seasonal_contrast': (batch_size, input_channels, 224, 224),
            'resnet_imagenet': (batch_size, input_channels, 224, 224),
            'resnet': (batch_size, input_channels, 224, 224),
            'seasonal_contrast_classifier': (batch_size, input_channels, 224, 224)
        }

        input_size = input_sizes.get(model_name, (batch_size, input_channels, input_size, input_size))
        model_summary = summary(model, input_size=input_size, dtypes=[torch.float32])

        if model_device == 'cpu':
            model.to(model_device)
            print('Model moved back to CPU after summary') # sometimes summary moves model to GPU if available

        valid_modules = ["module", "model", "encoder", "decoder", "module.encoder", "module.decoder", "coreunet"]

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

    if world_rank == 0:
        print(f"Output folder: {OUTPUT_FOLDER}")


    # -----------------------------------------------------------------------
    # 4. Load datasets
    # -----------------------------------------------------------------------

    # Validate configuration
    task_output_channels = {
        'lc': 11,
        'lc_classification': 11,
        'roads': 1,
        'building': 1,
        'building_classification': 5,
        'roads_classification': 2,
        'coords': 3
    }
    assert output_channels == task_output_channels[downstream_task], (
        f"{downstream_task} tasks should have {task_output_channels[downstream_task]} output channels"
    )
    assert n_shot is not None or split_ratio is not None, "Please define data partition protocol!"
    assert isinstance(n_shot, int) ^ isinstance(split_ratio, float), "n_shot cannot be used with split_ratio!"

    # Display partition info
    if world_rank == 0:
        print('--------------------------------')
        partition_type = 'n_shot' if n_shot is not None else 'split_ratio'
        partition_value = n_shot if n_shot is not None else split_ratio
        print(f"{partition_type}: {partition_value}")
        print('--------------------------------')

    # Choose dataset path based on model
    if model_name in MODELS_224_r30:
        dataset_name, dataset_folder, data_path_inference = '224_30m', data_path_224_30m, data_path_inference_224
    elif model_name in MODELS_224:
        dataset_name, dataset_folder, data_path_inference = '224_10m', data_path_224_10m, data_path_inference_224
    else:
        dataset_name, dataset_folder, data_path_inference = '128_10m', data_path_128_10m, data_path_inference_128
        
    # Determine if cropping (model requires smaller images), if use by region (prob only relevant to PhilEO-Bench), 
    # and set weights or pos_weight for loss function
    crop_images = True if model_name == 'phileo_precursor' else False
    by_region = False if downstream_task == 'coords' else True
    pos_weight, weights = None, None


    # Data partition
    if isinstance(n_shot, int):
        if n_shot == 0:
            n_shot = 1
            train_mode = 'inference'
        
        x_train, y_train, x_val, y_val, pos_weight, weights = data_protocol.protocol_fewshot_memmapped(
            folder=dataset_folder,
            dst=None,
            n=n_shot,
            regions=regions,
            y=downstream_task,
            data_selection='create',
            name=dataset_name,
            crop_images=crop_images
        )

    elif isinstance(split_ratio, float):
        x_train, y_train, x_val, y_val = data_protocol.protocol_split(
            dataset_folder,
            split_percentage=split_ratio,
            regions=regions,
            y=downstream_task,
            by_region=by_region
        )

    # Prepare testset and inference set
    x_test, y_test = data_protocol.get_testset(
        folder=dataset_folder,
        y=downstream_task,
        crop_images=crop_images,
        by_region=by_region,
    )
    x_inference, y_inference = data_protocol.get_testset(
        folder=data_path_inference,
        y=downstream_task,
        crop_images=crop_images,
        by_region=by_region
    )

    # Log shapes of first elements in each dataset
    if world_rank == 0:
        print("Dataset protocol:", dataset_name)
        if len(x_train.array_list) > 0:
            print("Training set datapoint shape: X -", x_train.array_list[0].shape)
        else:
            print("Training dataset is empty.")
        if len(x_val.array_list) > 0:
            print("Validation set datapoint shape: X -", x_val.array_list[0].shape)
        else:
            print("Validation dataset is empty.")
        if len(x_test.array_list) > 0:
            print("Test set datapoint shape: X -", x_test.array_list[0].shape)
        else:
            print("Test dataset is empty.")
        if len(x_inference.array_list) > 0:
            print("Inference set datapoint shape: X -", x_inference.array_list[0].shape)
        else:
            print("Inference dataset is empty.")

    # Create dataloaders
    dl_train, dl_test, dl_val, dl_inference = load_data.load_data(
        x_train, y_train,
        x_val, y_val,
        x_test, y_test,
        x_inference, y_inference,
        with_augmentations=augmentations,
        num_workers=num_workers,
        batch_size=batch_size,
        downstream_task=downstream_task,
        model_name=model_name.split('_')[0],
        device=generator_device,
        pad_to_10_bands=pad_to_10_bands,
    )
    
    # Log dataloader sizes and training model
    if world_rank == 0:
        print(f"Length of training dataloader: {len(dl_train)}")
        print(f"Length of validation dataloader: {len(dl_val)}")
        print(f"Length of test dataloader: {len(dl_test)}")
        print(f"Length of inference dataloader: {len(dl_inference)}")
        print(f"Training on: {model_name}")
        print('--' * 10)

    # Optionally return datasets instead of running training
    if only_get_datasets:
        return dl_train, dl_val, dl_test, dl_inference


    # -----------------------------------------------------------------------
    # 5. Initialize the trainer
    # -----------------------------------------------------------------------

    # Get learning rate
    init_lr = lr
    assert (min_lr is None) != (warmup_gamma is None), 'min_lr and warmup_gamma cannot be used together'
    if warmup and warmup_gamma is not None:
        lr = lr / int(( 10 )**(warmp_steps))  # for warmup start


    trainer = get_trainer(
        model_name=model_name,
        downstream_task=downstream_task,
        epochs=epochs,
        lr=lr,
        model=model,
        device=model_device,
        lr_scheduler=lr_scheduler,
        warmup=warmup,
        early_stop=early_stop,
        dl_train=dl_train,
        dl_val=dl_val,
        dl_test=dl_test,
        dl_inference=dl_inference,
        NAME=NAME,
        OUTPUT_FOLDER=OUTPUT_FOLDER,
        vis_val=vis_val,
        warmup_steps=warmp_steps,
        warmup_gamma=warmup_gamma,
        pos_weight=pos_weight,
        weights=weights,
        save_info_vars=(model_summary, n_shot, split_ratio, warmup, init_lr),
        rank=world_rank,
        min_lr=min_lr
    )


    # -----------------------------------------------------------------------
    # 6. Training / testing / inference workflow
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
    # 7. Finish script. Save parameters to YAML and cleanup if DDP
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

    # 1. Reading YAML file
    parser, parser_yaml = get_args()
    args_yaml, remainder = parser_yaml.parse_known_args()
    
    if args_yaml.read_yaml is not None:
        # print(f"WARNING: overwriting all parameters with defaults stored in {args_yaml.read_yaml}")
        args = read_yaml(args_yaml.read_yaml)
    else:
        args = parser.parse_args()

    # 2. Run main function
    if True:
        # n_shot_list = [0, 50, 100, 500, 1000, 5000]
        n_shot_list = [5000, 1000, 500, 100, 50, 0]
        for n_shot in n_shot_list:
            args.n_shot = n_shot
            for freeze_pretrained in [True, False]:
                args.freeze_pretrained = freeze_pretrained
                if n_shot == 0 and not freeze_pretrained:
                    continue
                # for downstream_task in ['']:
                for downstream_task in ['_classification']:
                # for downstream_task in ['', '_classification']:
                    args.downstream_task = args.downstream_task + downstream_task
                    args.model_name = args.model_name + '_classifier' if 'classification' in args.downstream_task else args.model_name
                
                    print(f"Running experiment with n_shot: {args.n_shot}, freeze_pretrained: {args.freeze_pretrained}, downstream_task: {args.downstream_task}, model_name: {args.model_name}")
                    main(**vars(args))
                    
                    # Remove classification if added
                    args.downstream_task = args.downstream_task.replace('_classification', '')
                    args.model_name = args.model_name.replace('_classifier', '')

    else:
        main(**vars(args))

    print('Finished')