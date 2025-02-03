# Standard Library
import os
from tqdm import tqdm

from matplotlib import pyplot as plt

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
# from torch.amp import GradScaler, autocast
import numpy as np
import json
from collections import defaultdict
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

# utils
from utils import visualize

import sys
# sys.path.append('./models/')
from models.utils_fm import MultiTaskLoss

from utils.custom_scheduler import MultiMetricScheduler

class TrainBase():

    def __init__(self, model: nn.Module, device: torch.device, train_loader: DataLoader, val_loader: DataLoader,
                 test_loader: DataLoader, inference_loader: DataLoader, epochs:int = 50, early_stop:int=25, lr: float = 0.001, lr_scheduler: str = None, warmup:bool=True,
                 metrics: list = None, name: str="model", out_folder :str ="trained_models/", visualise_validation:bool=True, 
                 warmup_steps:int=5, warmup_gamma:int=10, pos_weight:np.array=None, weights:np.array=None, save_info_vars:tuple = None, apply_zoom:bool=False, 
                 climate_segm:bool=False, fixed_task:str=None, rank:int=None, min_lr:float=1e-6, perceptual_loss:bool=False):
        
        self.train_mode = 'fp32' # choose between 'fp32', 'amp', 'fp16'
        self.val_mode = 'fp32' # choose between 'fp32', 'amp', 'fp16'
        
        if self.train_mode == 'fp16':
            self.model = model.half()
        else:
            self.model = model

        self.rank = rank
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self.use_ddp = dist.is_available() and dist.is_initialized()
        self.is_main_process = (not dist.is_available()) or (not dist.is_initialized()) or self.rank == 0

        # if rank != 0:
        #     builtins.print = lambda *args, **kwargs: None  # Disable print on non-master ranks

        self.visualise_validation = visualise_validation if self.is_main_process else False


        # print(f"Initializing weights. Model training with {self.train_mode}, and validating with {self.val_mode}")
        # self.model.apply(self.weight_init)

        self.test_loss = None
        self.last_epoch = None
        self.best_sd = None
        self.epochs = epochs
        self.early_stop = early_stop
        self.learning_rate = lr
        self.device = device
        self.apply_zoom = apply_zoom
        self.fixed_task = fixed_task
        self.climate_segm = climate_segm
        self.perceptual_loss = perceptual_loss
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.inference_loader = inference_loader
        self.metrics = metrics
        self.lr_scheduler = lr_scheduler
        self.warmup = warmup
        self.warmup_steps = warmup_steps
        self.name = name
        self.out_folder = out_folder
        if visualise_validation:
            os.makedirs(f'{self.out_folder}/val_images', exist_ok=True)
            os.makedirs(f'{self.out_folder}/train_images', exist_ok=True)
        if pos_weight is not None:
            self.pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(self.device)
        if weights is not None:
            self.weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        self.min_lr = min_lr
        self.criterion = self.set_criterion()
        self.scaler, self.optimizer = self.set_optimizer()
        self.scheduler = self.set_scheduler()

        if self.warmup and warmup_gamma is not None:
            multistep_milestone =  list(range(1, self.warmup_steps+1))
            self.scheduler_warmup = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=multistep_milestone, gamma=(warmup_gamma))
            
        elif self.warmup and self.min_lr is not None:
            def warmup_linear(epoch):
                if epoch < warmup_steps:
                    # Linear increase from `min_lr / max_lr` to `1.0` over `warmup_steps`
                    return (self.min_lr + (self.learning_rate - self.min_lr) * (epoch / warmup_steps)) / self.learning_rate
                return 1.0  # After warmup, maintain the learning rate (no scaling)

            self.scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup_linear)

        # Save Info vars
        self.model_summary, self.n_shot, self.split_ratio, self.warmup, self.init_lr = save_info_vars
        
        self.test_metrics = None
                
        # initialize torch device        
        if (not dist.is_available()) and (not dist.is_initialized()) and dist.get_rank() == 0:
            torch.set_default_device(self.device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            print("No CUDA device available.")

        # init useful variables
        self.best_epoch = 0
        self.best_loss = None
        self.best_model_state = model.state_dict().copy()
        self.epochs_no_improve = 0
        
        # used for plots
        self.tl = []
        self.vl = []
        self.e = []
        self.lr = []

    @staticmethod
    def weight_init(module: nn.Module):
        """
        Applies Kaiming (He) initialization to Conv2D and Linear layers,
        and sensible defaults for norm layers.
        """
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight, 
                a=0,  # assuming ReLU/GELU-like
                mode='fan_in', 
                nonlinearity='relu'
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(
                module.weight, 
                a=0,  # assuming ReLU/GELU-like
                mode='fan_in',
                nonlinearity='relu'
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            # A common default for norm layers
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def set_optimizer(self):
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.learning_rate, eps=1e-06)

        scaler = GradScaler()

        # Save the initial learning rate in optimizer's param_groups
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = self.learning_rate

        return scaler, optimizer

    def set_criterion(self):
        return nn.MSELoss()

    def set_scheduler(self):
        if self.lr_scheduler == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                20,
                2,
                eta_min=0.000001,
                last_epoch=self.epochs - 1,
            )
        elif self.lr_scheduler == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=6, min_lr=1e-6)
        else:
            scheduler = None
        return scheduler

    def get_loss(self, images, labels):
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        return loss
    
    def get_metrics(self, images=None, labels=None, running_metric=None, k=None):
        
        if (running_metric is not None) and (k is not None):
            metric_names = ['mse','mae','mave','acc','precision','recall','baseline_mse']
            # intermediary_values = ['mse','mae','mave','acc','tp','fp','fn','baseline_mse']

            final_metrics = {'mse':running_metric[0] / (k + 1), 'mae':running_metric[1] / (k + 1), 'mave':running_metric[2] / (k + 1), 'acc':running_metric[3]/ (k + 1), 'precision':running_metric[4]/(running_metric[4]+running_metric[5]), 'recall':running_metric[4]/(running_metric[4]+running_metric[6]), 'baseline_mse':running_metric[7] / (k + 1)}
            final_metrics['f1'] = 2 * final_metrics['precision'] * final_metrics['recall'] / (final_metrics['precision'] + final_metrics['recall'])

            return final_metrics

        elif (images == None) and (labels == None):
            intermediary_values = ['mse','mae','mave','acc','tp','fp','fn','baseline_mse']
            metric_init = np.zeros(len(intermediary_values)) # 
            return  metric_init
        
        
        else:
            
            outputs = self.model(images)
            # regression metrics
            error = outputs - labels
            squared_error = error**2
            test_mse = squared_error.mean().item()
            test_mae = error.abs().mean().item()
            test_mave = torch.mean(torch.abs(outputs.mean(dim=(1,2)) - labels.mean(dim=(1,2)) ) ).item()

            # regression metrics disguised as classification
            threshold = 0.5
            label_classification = (labels > threshold).type(torch.int8)
            output_classification = (outputs > threshold).type(torch.int8)

            diff = output_classification - label_classification
            fp = torch.count_nonzero(diff==1).item()
            fn = torch.count_nonzero(diff==-1).item()
            tp = label_classification.sum().item() - fn

            test_accuracy = (label_classification==output_classification).type(torch.float).mean().item()
            test_zero_model_mse = (labels**2).mean().item()

            return np.array([test_mse,test_mae,test_mave,test_accuracy,tp,fp,fn,test_zero_model_mse])

    def t_loop(self, epoch, s):
        # Initialize the running loss
        train_loss = 0.0
        # Initialize the progress bar for training
        train_pbar = tqdm(self.train_loader, total=len(self.train_loader),
                          desc=f"Epoch {epoch + 1}/{self.epochs}")

        # loop training through batches
        for i, (images, labels) in enumerate(train_pbar):
            # Move inputs and targets to the device (GPU)
            images, labels = images.to(self.device), labels.to(self.device)
            # images.requires_grad = True; labels.requires_grad = True

            # Zero the gradients
            self.optimizer.zero_grad()
            # get loss
            with autocast(dtype=torch.float16):
                loss = self.get_loss(images, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            train_loss += loss.item()

            # display progress on console
            train_pbar.set_postfix({
                "loss": f"{train_loss / (i + 1):.4f}",
                f"lr": self.optimizer.param_groups[0]['lr']})

            # # Update the scheduler
            if self.lr_scheduler == 'cosine_annealing':
                s.step()

        return i, train_loss

    def val_visualize(self, images, labels, outputs, name):
        visualize.visualize(x=images, y=labels, y_pred=outputs, images=5,
                            channel_first=True, vmin=0, vmax=1, save_path=f"{self.out_folder}/{name}.png")

    def v_loop(self, epoch):

        # Initialize the progress bar for training
        val_pbar = tqdm(self.val_loader, total=len(self.val_loader),
                          desc=f"Epoch {epoch + 1}/{self.epochs}")

        with torch.no_grad():
            self.model.eval()
            val_loss = 0
            for j, (images, labels) in enumerate(val_pbar):
                # Move inputs and targets to the device (GPU)
                images, labels = images.to(self.device), labels.to(self.device)

                # get loss
                loss = self.get_loss(images, labels)
                val_loss += loss.item()

                # display progress on console
                val_pbar.set_postfix({
                    "val_loss": f"{val_loss / (j + 1):.4f}",
                    f"lr": self.optimizer.param_groups[0]['lr']})

            if self.visualise_validation:
                outputs = self.model(images)

                if type(outputs) is tuple:
                    outputs = outputs[0]

                self.val_visualize(images.detach().cpu().numpy(), labels.detach().cpu().numpy(), outputs.detach().cpu().numpy(), name=f'/val_images/val_{epoch}')

            return j, val_loss

    def save_ckpt(self, epoch, val_loss):
        model_sd = self.model.state_dict().copy()

        if self.best_loss is None:
            self.best_epoch = epoch
            self.best_loss = val_loss
            torch.save(model_sd, os.path.join(self.out_folder, f"{self.name}_best.pt"))
            self.best_sd = model_sd

        elif self.best_loss > val_loss:
            self.best_epoch = epoch
            self.best_loss = val_loss
            self.epochs_no_improve = 0

            torch.save(model_sd, os.path.join(self.out_folder, f"{self.name}_best.pt"))
            self.best_sd = model_sd

        else:
            self.epochs_no_improve += 1

        torch.save(model_sd, os.path.join(self.out_folder, f"{self.name}_last.pt"))

    def plot_curves(self, epoch):
        # visualize loss & lr curves
        self.e.append(epoch)

        fig = plt.figure()
        plt.plot(self.e, self.tl, label='Training Loss', )
        plt.plot(self.e, self.vl, label='Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(self.out_folder, f"loss.png"))
        plt.close('all')
        fig = plt.figure()
        plt.plot(self.e, self.lr, label='Learning Rate')
        plt.legend()
        plt.savefig(os.path.join(self.out_folder, f"lr.png"))
        plt.close('all')

    def train(self):
        print("Starting training...")
        print("")

        # init model
        self.model.to(self.device)
        self.model.train()

        # create dst folder for generated files/artifacts
        if self.is_main_process:
            os.makedirs(self.out_folder, exist_ok=True)
        s = self.scheduler

        # Training loop
        for epoch in range(self.epochs):
            if epoch == 0 and self.warmup == True:
                s = self.scheduler_warmup
                print('Starting linear warmup phase')
            elif epoch == self.warmup_steps and self.warmup == True:
                s = self.scheduler
                self.warmup = False
                print('Warmup finished')

            i, train_loss = self.t_loop(epoch, s)
            j, val_loss = self.v_loop(epoch)

            self.tl.append(train_loss / (i + 1))
            self.vl.append(val_loss / (j + 1))
            self.lr.append(self.optimizer.param_groups[0]['lr'])

            # Update the scheduler
            if self.warmup:
                s.step()
            elif self.lr_scheduler == 'reduce_on_plateau':
                s.step(self.vl[-1])

            #save check point
            self.save_ckpt(epoch, val_loss / (j + 1))

            # visualize loss & lr curves
            self.plot_curves(epoch)
            self.model.train()

            # Early stopping
            if self.epochs_no_improve == self.early_stop:
                print(f'Early stopping triggered after {epoch + 1} epochs.')
                self.last_epoch = epoch + 1
                break

    def test(self):
        # Load the best weights
        self.model.load_state_dict(self.best_sd)

        print("Finished Training. Best epoch: ", self.best_epoch + 1)
        print("")
        print("Starting Testing...")
        self.model.eval()
        test_pbar = tqdm(self.test_loader, total=len(self.test_loader),
                          desc=f"Test Set")
        with torch.no_grad():

            running_metric = self.get_metrics()

            for k, (images, labels) in enumerate(test_pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                running_metric += self.get_metrics(images,labels)

            self.test_metrics = self.get_metrics(running_metric=running_metric, k=k)

            print(f"Test Loss: {self.test_metrics}")
            outputs = self.model(images)
            self.val_visualize(images.detach().cpu().numpy(), labels.detach().cpu().numpy(),
                               outputs.detach().cpu().numpy(), name='test')

        if isinstance(self.model, nn.DataParallel):
            model_sd = self.model.module.state_dict().copy()
        else:
            model_sd = self.model.state_dict().copy()

        torch.save(model_sd, os.path.join(self.out_folder, f"{self.name}_final.pt"))

    def inference(self):

        print("Starting Inference...")
        self.model.eval()
        inference_pbar = tqdm(self.inference_loader, total=len(self.inference_loader),
                          desc=f"Inference Set")
        with torch.no_grad():

            running_metric = self.get_metrics()

            for k, (images, labels) in enumerate(inference_pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                running_metric += self.get_metrics(images,labels)

            self.inference_metrics = self.get_metrics(running_metric=running_metric, k=k)

            print(f"Inference Loss: {self.inference_metrics}")
            outputs = self.model(images)
            self.val_visualize(images.detach().cpu().numpy(), labels.detach().cpu().numpy(),
                               outputs.detach().cpu().numpy(), name='inference')

        artifacts = {'inference_metrics': self.inference_metrics}

        with open(f"{self.out_folder}/artifacts_inference.json", "w") as outfile:
            json.dump(artifacts, outfile, indent=4)


    '''
    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))

    def test_samples(self, plot_name='test_samples', images=None, labels=None):
        self.model.eval()
        num_samples = 5  # Number of samples to visualize
        
        with torch.no_grad():
            images_sample = []
            labels_sample = []

            if images is None or labels is None:
                # Randomly choose 5 batches from the dataloader if no images are provided
                batch_list = list(self.test_loader)
                selected_batches = random.sample(batch_list, num_samples)
                
                for batch in selected_batches:
                    img, lbl = batch
                    img = img.to(self.device)
                    lbl = lbl.to(self.device)

                    # Select a random image from each batch
                    index = random.choice(range(img.size(0)))
                    images_sample.append(img[index])
                    labels_sample.append(lbl[index])
                
                images_sample = torch.stack(images_sample)
                labels_sample = torch.stack(labels_sample)

            else:
                # Use the provided images and labels
                images_sample = images.to(self.device)
                labels_sample = labels.to(self.device)
            
            # Forward pass on the selected images
            outputs = self.model(images_sample)
            
            images_sample = images_sample.detach().cpu().numpy()
            labels_sample = labels_sample.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            
            # Visualization
            self.val_visualize(images_sample, labels_sample, outputs, 
                            name=plot_name)
            
            return images_sample, labels_sample, outputs
    '''

    def save_info(self, model_summary=None, n_shot=None, p_split=None, warmup=None, lr=None):
        print("Saving artifacts...")
        artifacts = {'training_parameters': {'model': self.name,
                                             'lr': lr,
                                             'scheduler': self.lr_scheduler,
                                             'warm_up': warmup,
                                             'optimizer': str(self.optimizer).split(' (')[0],
                                             'device': str(self.device),
                                             'training_epochs': self.epochs,
                                             'early_stop': self.early_stop,
                                             'train_samples': len(self.train_loader) * model_summary.input_size[0][0],
                                             'val_samples': len(self.val_loader) * model_summary.input_size[0][0],
                                             'test_samples': len(self.test_loader) * model_summary.input_size[0][0],
                                             'n_shot': n_shot,
                                             'p_split': p_split
                                             },

                     'training_info': {'best_val_loss': self.best_loss,
                                       'best_epoch': self.best_epoch,
                                       'last_epoch': self.last_epoch},

                     'test_metrics': self.test_metrics,

                     'plot_info': {'epochs': self.e,
                                   'val_losses': self.vl,
                                   'train_losses': self.tl,
                                   'lr': self.lr},

                     'model_summary': {'batch_size': model_summary.input_size[0],
                                       'input_size': model_summary.total_input,
                                       'total_mult_adds': model_summary.total_mult_adds,
                                       'back_forward_pass_size': model_summary.total_output_bytes,
                                       'param_bytes': model_summary.total_param_bytes,
                                       'trainable_params': model_summary.trainable_params,
                                       'non-trainable_params': model_summary.total_params - model_summary.trainable_params,
                                       'total_params': model_summary.total_params}
                     }
        print('artifacts')
        with open(f"{self.out_folder}/artifacts.json", "w") as outfile:
            json.dump(artifacts, outfile, indent=4)
        print("Artifacts saved successfully.")






class TrainFoundation(TrainBase):
    
    def loop_data(self):
        total_loss = 0.0
        
        for epoch in range(self.epochs):

            if hasattr(self.train_loader, 'sampler') and isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)
            if hasattr(self.val_loader, 'sampler') and isinstance(self.val_loader.sampler, DistributedSampler):
                self.val_loader.sampler.set_epoch(epoch)

            if self.is_main_process:
                train_pbar = tqdm(self.train_loader, total=len(self.train_loader),
                                desc=f"Epoch {epoch + 1}/{self.epochs}")
            else:
                train_pbar = self.train_loader  # a normal iterator, no tqdm
            
            for i, (images, labels) in enumerate(train_pbar):
                images = images.to(self.device)
                labels = {k: v.to(self.device) for k, v in labels.items()}
                
                if i == 0 and epoch == 0:
                    print("Input shapes:", images.shape, {k: v.shape for k, v in labels.items()})
                    out = self.model(images)

                loss, log_loss = self.get_loss(out, labels)
                
                total_loss += loss.item()
                
            if self.is_main_process and isinstance(train_pbar, tqdm):
                if self.fixed_task is None:
                    train_pbar.set_postfix({
                        "loss": f"{total_loss / (i + 1):.4f}",
                        "lr": self.optimizer.param_groups[1]['lr']
                    })
                else:
                    train_pbar.set_postfix({
                        "loss": f"{total_loss / (i + 1):.4f}",
                        "lr": self.optimizer.param_groups[0]['lr']
                    })

            self.check_inf_nan(loss, log_loss, images, labels, out)

    def set_criterion(self):
        return MultiTaskLoss(apply_zoom=self.apply_zoom, fixed_task=self.fixed_task, device=self.device)

    def set_optimizer(self):
        # Get initial losses for each task
        # This sets each log(𝜎_𝑖) to log(sqrt(loss_𝑖)) ≈ 1/2 log(loss_𝑖)
        if self.fixed_task is None:
            initial_losses = self.estimate_initial_losses(self.model, self.criterion, self.train_loader, self.device)
            
            if self.is_main_process:
                print("Initial losses:", initial_losses)
                print("Initializing scales of individual losses")

            self.criterion.scale_recon = 1.0 / initial_losses['reconstruction']
            self.criterion.scale_seg =   1.0 / initial_losses['climate']
            self.criterion.scale_geo =   1.0 / initial_losses['geolocation']
            
            if self.climate_segm:
                self.criterion.scale_tv = 0.1 / initial_losses['total_variation']
            if self.apply_zoom:
                self.criterion.scale_zoom = 1.0 / initial_losses['zoom_level']
            if self.perceptual_loss:
                self.criterion.scale_perc =  1.0 / initial_losses['perceptual']

            if self.is_main_process:
                print("Initializing log variances of each loss")
            with torch.no_grad():
                self.criterion.log_sigma_recon.copy_(torch.tensor(-0.5))
                self.criterion.log_sigma_clim.copy_(torch.tensor(0.0))
                self.criterion.log_sigma_geo.copy_(torch.tensor(0.0))
                
                if self.climate_segm:
                    self.criterion.log_sigma_tv.copy_(torch.tensor(0.55))
                if self.apply_zoom:
                    self.criterion.log_sigma_zoom.copy_(torch.tensor(-0.5))
                if self.perceptual_loss:
                    self.criterion.log_sigma_perc.copy_(torch.tensor(0.55))
                
            log_sigma_params = [p for n, p in self.criterion.named_parameters() if 'log_sigma' in n]

            if isinstance(self.model, nn.DataParallel) or isinstance(self.model, nn.parallel.DistributedDataParallel):
                stem_encoder_params = list(self.model.module.stem.parameters()) + list(self.model.module.encoder.parameters())
                decoder_params = list(self.model.module.decoder.parameters())
                head_geo_params = list(self.model.module.head_geo.parameters())
                head_recon_params = list(self.model.module.head_recon.parameters())
                head_seg_params = list(self.model.module.head_seg.parameters())
            else:
                stem_encoder_params = list(self.model.stem.parameters()) + list(self.model.encoder.parameters())
                decoder_params = list(self.model.decoder.parameters())
                head_geo_params = list(self.model.head_geo.parameters())
                head_recon_params = list(self.model.head_recon.parameters())
                head_seg_params = list(self.model.head_seg.parameters())



            def split_decay_no_decay(named_params):
                decay_params, no_decay_params = [], []
                for name, param in named_params:
                    if not param.requires_grad:
                        continue
                    # If it's a bias, norm, or 1D weight (like LN/Bias), put it in no_decay
                    if param.ndim == 1 or 'bias' in name or 'norm' in name.lower():
                        no_decay_params.append(param)
                    else:
                        decay_params.append(param)
                return decay_params, no_decay_params


            def make_decay_no_decay_groups(params, lr):
                """
                Given a named-parameter iterable and a target lr,
                return two param-group dicts: one with weight decay,
                and one without.
                """
                # Convert to named list in case user passed just a list of params
                # or param modules. The user might pass in a plain list, 
                # so we handle that:
                if all(isinstance(p, nn.Parameter) for p in params):
                    # If it's a plain list of parameters, artificially name them
                    named_params = [(f"p{i}", p) for i, p in enumerate(params)]
                else:
                    # Otherwise, assume it's something like model.named_parameters()
                    named_params = list(params)


                decay, no_decay = split_decay_no_decay(named_params)
                return [
                    {'params': decay,    'lr': lr, 'weight_decay': 1e-3},
                    {'params': no_decay, 'lr': lr, 'weight_decay': 0.0},
                ]

            # Now create your overall param-group list:
            
            self.lr_mult_sigma = 0.1
            self.lr_mult_weights = 1.0
            
            param_groups = []
            param_groups += make_decay_no_decay_groups(log_sigma_params,     lr=self.learning_rate * self.lr_mult_sigma) # groups 0 and 1
            param_groups += make_decay_no_decay_groups(stem_encoder_params,  lr=self.learning_rate * self.lr_mult_weights) # groups 2 and 3
            param_groups += make_decay_no_decay_groups(decoder_params,       lr=self.learning_rate * self.lr_mult_weights) # groups 4 and 5
            param_groups += make_decay_no_decay_groups(head_geo_params,      lr=self.learning_rate * self.lr_mult_weights) # groups 6 and 7
            param_groups += make_decay_no_decay_groups(head_recon_params,    lr=self.learning_rate * self.lr_mult_weights) # groups 8 and 9
            param_groups += make_decay_no_decay_groups(head_seg_params,      lr=self.learning_rate * self.lr_mult_weights) # groups 10 and 11

            # Finally, instantiate the optimizer
            optimizer = torch.optim.AdamW(param_groups)




            # optimizer = torch.optim.AdamW([
            #     {"params": log_sigma_params,    "lr": self.learning_rate * 1.0},   # param_group 0
            #     {"params": stem_encoder_params, "lr": self.learning_rate * 1.0},   # param_group 1
            #     {"params": decoder_params,      "lr": self.learning_rate * 1.0},   # param_group 2
            #     {"params": head_geo_params,     "lr": self.learning_rate * 1.0},   # param_group 3
            #     {"params": head_recon_params,   "lr": self.learning_rate * 1.0},   # param_group 4
            #     {"params": head_seg_params,     "lr": self.learning_rate * 1.0},   # param_group 5
            # ], weight_decay=1e-4)
            
            # for i, param_group in enumerate(optimizer.param_groups):
            #     print(f"Layer {i + 1} learning rate: {param_group['lr']}")

        else:
            print(f"Fixed task: {self.fixed_task}")
            optimizer = torch.optim.AdamW(self.model.parameters(),
                                        lr=self.learning_rate, eps=1e-06)

        scaler = GradScaler()
        
        # import pdb; pdb.set_trace()
        
        return scaler, optimizer


    def set_scheduler(self):
        if self.fixed_task is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=6, min_lr=1e-6)
        else:
            # 1. Define how each param group chooses its reduce-on plateau metric(s)
            scheduler_type = self.lr_scheduler
            
            # relevant for reduce_on_plateau only
            min_lr = self.min_lr
            factor = 0.1
            patience = 6
            
            # relevant for cosine_annealing only
            T_max = self.epochs - self.warmup_steps - 1
            eta_min = self.min_lr
            
            if self.is_main_process and self.lr_scheduler == 'cosine_annealing':
                print(f"Using cosine annealing scheduler with T_max={T_max}, eta_min={eta_min}")
            elif self.is_main_process and self.learning_rate == 'reduce_on_plateau':
                print(f"Using reduce-on-plateau scheduler with factor={factor}, patience={patience}, min_lr={min_lr}")

            param_groups_config = {
                0: {  # for log_sigmas (same as 1)
                    'scheduler_type': scheduler_type,
                    'metrics': ['total_loss'],
                    'factor': factor,
                    'patience': patience,
                    'min_lr': min_lr,
                    'mode': 'min',
                    'T_max': T_max,
                    'eta_min': eta_min * self.lr_mult_sigma
                },
                1: {  # for log_sigmas (same as 0)
                    'scheduler_type': scheduler_type,
                    'metrics': ['total_loss'],
                    'factor': factor,
                    'patience': patience,
                    'min_lr': min_lr,
                    'mode': 'min',
                    'T_max': T_max,
                    'eta_min': eta_min * self.lr_mult_sigma
                },
                2: {  # for stem+encoder (same as 3)
                    'scheduler_type': scheduler_type,
                    'metrics': ['total_loss'],
                    'factor': factor,
                    'patience': patience,
                    'min_lr': min_lr,
                    'mode': 'min',
                    'T_max': T_max,
                    'eta_min': eta_min * self.lr_mult_weights
                },
                3: {  # for stem+encoder (same as 2)
                    'scheduler_type': scheduler_type,
                    'metrics': ['total_loss'],
                    'factor': factor,
                    'patience': patience,
                    'min_lr': min_lr,
                    'mode': 'min',
                    'T_max': T_max,
                    'eta_min': eta_min * self.lr_mult_weights
                },
                4: {  # for decoder (same as 5)
                    'scheduler_type': scheduler_type,
                    'metrics': ['reconstruction', 'climate'],
                    'factor': factor,
                    'patience': patience,
                    'min_lr': min_lr,
                    'mode': 'min',
                    'T_max': T_max,
                    'eta_min': eta_min * self.lr_mult_weights
                },
                5: {  # for decoder (same as 4)
                    'scheduler_type': scheduler_type,
                    'metrics': ['reconstruction', 'climate'],
                    'factor': factor,
                    'patience': patience,
                    'min_lr': min_lr,
                    'mode': 'min',
                    'T_max': T_max,
                    'eta_min': eta_min * self.lr_mult_weights
                },
                6: {  # for head_geo (same as 7)
                    'scheduler_type': scheduler_type,
                    'metrics': ['geolocation'],
                    'factor': factor,
                    'patience': patience,
                    'min_lr': min_lr,
                    'mode': 'min',
                    'T_max': T_max,
                    'eta_min': eta_min * self.lr_mult_weights
                },
                7: {  # for head_geo (same as 6)
                    'scheduler_type': scheduler_type,
                    'metrics': ['geolocation'],
                    'factor': factor,
                    'patience': patience,
                    'min_lr': min_lr,
                    'mode': 'min',
                    'T_max': T_max,
                    'eta_min': eta_min * self.lr_mult_weights
                },
                8: {  # for head_recon (same as 9)
                    'scheduler_type': scheduler_type,
                    'metrics': ['reconstruction'],
                    'factor': factor,
                    'patience': patience,
                    'min_lr': min_lr,
                    'mode': 'min',
                    'T_max': T_max,
                    'eta_min': eta_min * self.lr_mult_weights
                },
                9: {  # for head_recon (same as 8)
                    'scheduler_type': scheduler_type,
                    'metrics': ['reconstruction'],
                    'factor': factor,
                    'patience': patience,
                    'min_lr': min_lr,
                    'mode': 'min',
                    'T_max': T_max,
                    'eta_min': eta_min * self.lr_mult_weights
                },
                10: {  # for head_seg (same as 11)
                    'scheduler_type': scheduler_type,
                    'metrics': ['climate'],
                    'factor': factor,
                    'patience': patience,
                    'min_lr': min_lr,
                    'mode': 'min',
                    'T_max': T_max,
                    'eta_min': eta_min * self.lr_mult_weights
                },
                11: {  # for head_seg (same as 10)
                    'scheduler_type': scheduler_type,
                    'metrics': ['climate'],
                    'factor': factor,
                    'patience': patience,
                    'min_lr': min_lr,
                    'mode': 'min',
                    'T_max': T_max,
                    'eta_min': eta_min * self.lr_mult_weights
                },
            }


            # 2. Instantiate the custom scheduler
            scheduler = MultiMetricScheduler(self.optimizer, param_groups_config)

            self.lr_groups = [[] for _ in range(len(self.optimizer.param_groups))]
        
        return scheduler

    def get_loss(self, outputs, labels):
        # Now outputs and labels are dictionaries
        loss, log_loss = self.criterion(outputs, labels)
        return loss, log_loss


    def estimate_initial_losses(self, model, loss_fn, dataloader, device, num_batches=None):
        """
        Runs a few batches from 'dataloader' to compute average losses
        for each task, with a progress bar.
        
        Args:
            model: Your multi-task model
            loss_fn: e.g., MultiTaskLoss instance (with log sigmas=0 at first)
            dataloader: A torch DataLoader
            device: 'cuda' or 'cpu'
            num_batches: How many batches to average over (optional)
            
        Returns:
            A dict of average losses for each task, e.g. {
                'reconstruction': float,
                'perceptual': float,
                'climate': float,
                'total_variation': float,
                'geolocation': float,
                'zoom_level': float,
            }
        """
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(0)

        model.eval()

        with torch.no_grad():
            sums = {
                'reconstruction': 0.0,
                'climate': 0.0,
                'geolocation': 0.0,
            }

            if self.apply_zoom:
                sums['zoom_level'] = 0.0
            if self.climate_segm:
                sums['total_variation'] = 0.0
            if self.perceptual_loss:
                sums['perceptual'] = 0.0
            count = 0

            # Using tqdm for progress tracking
            if num_batches is None:
                tot_batches = len(dataloader)
            else:
                tot_batches = num_batches

            if self.is_main_process:
                progress = tqdm(enumerate(dataloader), total=tot_batches, desc="Estimating initial sigmas")

            else:
                progress = enumerate(dataloader)

            for i, (images, labels) in progress:
                images = images.to(device)
                labels = {k: v.to(device) for k, v in labels.items()}
                
                if self.train_mode == 'fp16':
                    images = images.half()
                    # labels = {k: v.half() for k, v in labels.items()}
                    
                # Forward pass
                outputs = model(images)
                                    
                # Get losses -- but here log_sigmas are all zero
                _, log_loss = loss_fn(outputs, labels)  # ignoring the total_loss

                # Accumulate per-task losses
                task_losses = log_loss['loss_components']  # dict of {task: float}
                for key in sums.keys():
                    sums[key] += task_losses[key]
                
                count += 1
                if num_batches is not None and i + 1 >= num_batches:
                    break
            
            # Add over processes (DDP)
            if self.use_ddp:
                loss_keys = list(sums.keys())
                local_losses = torch.tensor([sums[k] for k in loss_keys], device=device)
                dist.all_reduce(local_losses, op=dist.ReduceOp.SUM)
                
                global_avg_losses = local_losses / max(self.world_size, 1)
                # Update sums with the aggregated global averages
                for i, key in enumerate(loss_keys):
                    sums[key] = global_avg_losses[i].item()
                
            # Average
            for key in sums.keys():
                sums[key] /= max(count, 1)
        
        return sums



    def ddp_sync_epoch_log_loss(self, epoch_log_loss):
        """
        All-reduce the nested structure of `epoch_log_loss`, which typically looks like:
        {
            'total_loss': float,
            'loss_components': {'comp1': float, 'comp2': float, ...},
            'log_sigmas':      {'sigma1': float, ...},
            'scaled_loss':     {'some_key': float, ...}
        }

        After this call, each rank has the same, averaged values.

        Returns:
            epoch_log_loss (dict): The same dict but with globally averaged values.
        """

        # Safety check
        if not dist.is_initialized():
            # If you run single-process / single-GPU with no DDP,
            # just return as-is with no synchronization.
            return epoch_log_loss

        world_size = dist.get_world_size()

        def ddp_allreduce_dict(d):
            """
            Recursively all-reduce float values in a nested dictionary (in-place).
            """
            for key, val in d.items():
                # If it's a dict, recurse
                if isinstance(val, dict):
                    ddp_allreduce_dict(val)
                # If it's a float, all-reduce it
                elif isinstance(val, float):
                    t = torch.tensor(val, dtype=torch.float32, device=self.device)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    d[key] = t.item()
                # If it's None or something else you want to skip, handle accordingly
                elif val is None:
                    pass
                else:
                    raise ValueError(f"Unsupported type for ddp_allreduce_dict: {type(val)}")

        #
        # 1) All-reduce the top-level `total_loss`
        #
        if 'total_loss' in epoch_log_loss:
            t = torch.tensor(epoch_log_loss['total_loss'], dtype=torch.float32, device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            epoch_log_loss['total_loss'] = t.item()
        
        #
        # 2) All-reduce each sub-dict
        #
        for subdict_key in ['loss_components', 'log_sigmas', 'scaled_loss']:
            if subdict_key in epoch_log_loss and epoch_log_loss[subdict_key] is not None:
                ddp_allreduce_dict(epoch_log_loss[subdict_key])

        #
        # 3) Divide everything by world_size to get an average
        #
        if 'total_loss' in epoch_log_loss:
            epoch_log_loss['total_loss'] /= world_size

        def ddp_scale_dict(d):
            for key, val in d.items():
                if isinstance(val, dict):
                    ddp_scale_dict(val)
                elif isinstance(val, float):
                    d[key] = val / world_size

        for subdict_key in ['loss_components', 'log_sigmas', 'scaled_loss']:
            if subdict_key in epoch_log_loss and epoch_log_loss[subdict_key] is not None:
                ddp_scale_dict(epoch_log_loss[subdict_key])

        return epoch_log_loss



    def check_inf_nan(self, loss, log_loss, images, labels, outputs):
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"{self.rank}: Loss is NaN or Inf. Stopping training. Loss: {loss}")

            # Define a directory to save files
            debug_dir = "debug_data"
            os.makedirs(debug_dir, exist_ok=True)
            # Save tensors to files
            torch.save(images, os.path.join(debug_dir, f"rank_{self.rank}_images.pt"))
            torch.save(labels, os.path.join(debug_dir, f"rank_{self.rank}_labels.pt"))
            torch.save(outputs, os.path.join(debug_dir, f"rank_{self.rank}_outputs.pt"))
            torch.save(log_loss, os.path.join(debug_dir, f"rank_{self.rank}_log_loss.pt"))
            
            # Save the model
            torch.save(self.model.state_dict(), os.path.join(debug_dir, f"rank_{self.rank}_model.pt"))
            
            # Visualize model stats
            self.model.eval()
            with torch.no_grad():
                stats = visualize.collect_model_stats(self.model)
                visualize.print_stats_table(stats)

            print(f"{self.rank}: DEBUGGING INFO:")
            print(f"{self.rank}: Loss:", loss)
            print(f"{self.rank}: Total loss:", log_loss['total_loss'])
            print(f"{self.rank}: Loss components:", log_loss['loss_components'])
            print(f"{self.rank}: Log sigmas:", log_loss['log_sigmas'])
            print(f"{self.rank}: Scaled loss components:", log_loss['scaled_loss'])
            
            images_invalid = torch.isnan(images) | torch.isinf(images)
            print(f"{self.rank}: Images NaN or INF:", images_invalid.sum(dim=(2, 3)))
            labels_coords_invalid = torch.isnan(labels['coords']) | torch.isinf(labels['coords'])
            print(f"{self.rank}: Labels Coords NaN or INF:", labels_coords_invalid)
            outputs_coords_invalid = torch.isnan(outputs['coords']) | torch.isinf(outputs['coords'])
            print(f"{self.rank}: Outputs Coords NaN or INF:", outputs_coords_invalid)
            labels_climate_invalid = torch.isnan(labels['climate']) | torch.isinf(labels['climate'])
            print(f"{self.rank}: Labels Climate NaN or INF:", labels_climate_invalid)
            outputs_climate_invalid = torch.isnan(outputs['climate']) | torch.isinf(outputs['climate'])
            print(f"{self.rank}: Outputs Climate NaN or INF:", outputs_climate_invalid)
            labels_recons_invalid = torch.isnan(labels['reconstruction']) | torch.isinf(labels['reconstruction'])
            print(f"{self.rank}: Labels Recons NaN or INF:", labels_recons_invalid.sum(dim=(2,3)))
            outputs_recons_invalid = torch.isnan(outputs['reconstruction']) | torch.isinf(outputs['reconstruction'])
            print(f"{self.rank}: Outputs Recons NaN or INF:", outputs_recons_invalid.sum(dim=(2,3)))
            print(f"{self.rank}: Unique Climate Labels:", torch.unique(labels['climate']))
            
            
            raise ValueError("Loss is NaN. Stopping training.")
        else:
            pass


    def log_layer(self):
        import logging
        # Forward pass logger
        forward_logger = logging.getLogger("forward_logger")
        forward_handler = logging.FileHandler("forward_pass.log")
        forward_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        forward_logger.setLevel(logging.INFO)
        forward_logger.addHandler(forward_handler)
        forward_logger.propagate = False  # Prevent duplicate logging

        # Backward pass logger
        backward_logger = logging.getLogger("backward_logger")
        backward_handler = logging.FileHandler("backward_pass.log")
        backward_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        backward_logger.setLevel(logging.INFO)
        backward_logger.addHandler(backward_handler)
        backward_logger.propagate = False

        def forward_hook(module, input, output, layer_name):
            if isinstance(output, dict):
                for key, value in output.items():
                    if torch.is_tensor(value):
                        mean, std = value.mean().item(), value.std().item()
                        forward_logger.info(f"{layer_name} ({module.__class__.__name__}) Output {key} - mean: {mean:.3e}, std: {std:.3e}, min: {value.min().item():.3e}, max: {value.max().item():.3e}")
                        if isinstance(module, torch.nn.BatchNorm2d) and std < 1e-2:
                            forward_logger.warning(f"🚨 {layer_name} ({module.__class__.__name__}) has LOW STD: {std:.3e}. Possible numerical instability! 🚨")
                    else:
                        forward_logger.info(f"{layer_name} ({module.__class__.__name__}) Output {key} - not a tensor, cannot calculate mean/std")
            elif torch.is_tensor(output):
                mean, std = output.mean().item(), output.std().item()
                forward_logger.info(f"{layer_name} ({module.__class__.__name__}) - Output mean: {mean:.3e}, std: {std:.3e}, min: {output.min().item():.3e}, max: {output.max().item():.3e}")
                if isinstance(module, torch.nn.BatchNorm2d) and std < 1e-2:
                    forward_logger.warning(f"🚨 {layer_name} ({module.__class__.__name__}) has LOW STD: {std:.3e}. Possible numerical instability! 🚨")
            else:
                forward_logger.warning(f"{layer_name} ({module.__class__.__name__}) - Output is neither tensor nor dict, check data type")

            for idx, inp in enumerate(input):
                if torch.is_tensor(inp) and torch.isnan(inp).any():
                    forward_logger.warning(f"NaNs in input {idx} of {layer_name} ({module.__class__.__name__})")

        def full_backward_hook(module, grad_input, grad_output, layer_name):
            for idx, grad in enumerate(grad_output):
                if torch.is_tensor(grad):
                    backward_logger.info(f"{layer_name} ({module.__class__.__name__}) Full Grad Output {idx} - mean: {grad.mean().item():.3e}, std: {grad.std().item():.3e}, min: {grad.min().item():.3e}, max: {grad.max().item():.3e}")
                else:
                    backward_logger.info(f"{layer_name} ({module.__class__.__name__}) Full Grad Output {idx} - not a tensor, cannot calculate mean/std")

            for idx, grad in enumerate(grad_input):
                if torch.is_tensor(grad):
                    backward_logger.info(f"{layer_name} ({module.__class__.__name__}) Full Grad Input {idx} - mean: {grad.mean().item():.3e}, std: {grad.std().item():.3e}, min: {grad.min().item():.3e}, max: {grad.max().item():.3e}")
                    if torch.isnan(grad).any():
                        backward_logger.warning(f"NaNs detected in Full Grad Input {idx} of {layer_name} ({module.__class__.__name__})")
                else:
                    backward_logger.info(f"{layer_name} ({module.__class__.__name__}) Full Grad Input {idx} - not a tensor, cannot calculate mean/std")

        for name, layer in self.model.named_modules():
            if name:  # Skip the top-level model itself
                layer.register_forward_hook(lambda module, inp, out, n=name: forward_hook(module, inp, out, n))
                layer.register_full_backward_hook(lambda module, grad_in, grad_out, n=name: full_backward_hook(module, grad_in, grad_out, n))



    def t_loop(self, epoch, s):

        # Initialize the running loss for display
        train_loss = 0.0
        
        # Accumulate log_loss across batches
        total_loss_accum = 0.0
        loss_components_accum = defaultdict(float)
        log_sigmas_accum = defaultdict(float)
        log_scaled_accum = defaultdict(float)
        
        # # Initialize the progress bar for training
        # train_pbar = tqdm(self.train_loader, total=len(self.train_loader),
        #                 desc=f"Epoch {epoch + 1}/{self.epochs}")

        if self.is_main_process:
            train_pbar = tqdm(self.train_loader, total=len(self.train_loader),
                            desc=f"Train {epoch + 1}/{self.epochs}")
        else:
            train_pbar = self.train_loader  # a normal iterator, no tqdm

        # loop training through batches

        # with torch.profiler.profile(
        #     activities=[
        #         torch.profiler.ProfilerActivity.CPU,
        #         torch.profiler.ProfilerActivity.CUDA,
        #     ],
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True,
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./log/epoch_{epoch}")
        # ) as prof:
        
        
        # self.log_layer()
        
        if True:

            for i, (images, labels) in enumerate(train_pbar):
                
                images = images.to(self.device)
                if self.fixed_task is None:
                    labels = {key: value.to(self.device) for key, value in labels.items()}
                else:
                    labels = {self.fixed_task: labels[self.fixed_task].to(self.device)}
                
                # Zero the gradients
                self.optimizer.zero_grad()

                # GET LOSS
                
                # A) FP 32
                if self.train_mode == 'fp32':
                    outputs = self.model(images)
                    loss, log_loss = self.get_loss(outputs, labels)
                    
                    loss.backward()
                    # self.accelerator.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                # B) AMP
                elif self.train_mode == 'amp':
                    with autocast(dtype=torch.float16):
                        outputs = self.model(images)
                        loss, log_loss = self.get_loss(outputs, labels)

                    # for name, param in self.model.named_parameters():
                    #     if torch.isnan(param).any() or torch.isinf(param).any():
                    #         print(f"NaN/Inf detected in {name} before backward pass!")

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    # for name, param in self.model.named_parameters():
                    #     if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    #         print(f"NaN/Inf detected in gradients of {name}!")
                    
                # C) FP 16
                elif self.train_mode == 'fp16':
                    outputs = self.model(images.half())
                    labels = {key: value.half() if value.dtype.is_floating_point else value for key, value in labels.items()}
                    loss, log_loss = self.get_loss(outputs, labels)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()


                # CHECK FOR NaNs or INFs
                # import pdb; pdb.set_trace()
                self.check_inf_nan(loss, log_loss, images, labels, outputs)
                
                train_loss += loss.item()
                
                # ------------------------------
                # Accumulate the log_loss stats
                # ------------------------------
                total_loss_accum += log_loss['total_loss']
                
                # Accumulate each component
                for comp_key, comp_val in log_loss['loss_components'].items():
                    loss_components_accum[comp_key] += comp_val
                
                if self.fixed_task is None:
                    # Accumulate each log_sigma
                    for sigma_key, sigma_val in log_loss['log_sigmas'].items():
                        log_sigmas_accum[sigma_key] += sigma_val
                
                    # Accumulate each scaled component
                    for comp_key, comp_val in log_loss['scaled_loss'].items():
                        log_scaled_accum[comp_key] += comp_val
                
                # display progress on console
                # train_pbar.set_postfix({
                #     "loss": f"{train_loss / (i + 1):.4f}",
                #     "lr": self.optimizer.param_groups[0]['lr']
                # })

                if self.is_main_process and isinstance(train_pbar, tqdm):
                    if self.fixed_task is None:
                        train_pbar.set_postfix({
                            "loss": f"{train_loss / (i + 1):.4f}",
                            "lr": self.optimizer.param_groups[1]['lr']
                        })
                    else:
                        train_pbar.set_postfix({
                            "loss": f"{train_loss / (i + 1):.4f}",
                            "lr": self.optimizer.param_groups[0]['lr']
                        })

                # # Update the scheduler if needed
                # if self.lr_scheduler == 'cosine_annealing':
                #     import pdb; pdb.set_trace()
                #     s.step()

        #         prof.step()
        
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        # Visualization on the last batch (if enabled)
        if self.visualise_validation and self.is_main_process:
            # The loop ends with images, labels from the last batch,
            # but re-run if you want updated outputs after finishing iteration
            # self.model.eval()
            # with torch.no_grad():
            #     outputs = self.model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            self.val_visualize(images, labels, outputs, name=f'/train_images/train_{epoch}')
        
        # ------------------------------------
        # Compute average for each log_loss key
        # ------------------------------------
        num_batches = i + 1  # i starts at 0, so total batches processed = i+1
        
        epoch_log_loss = {
            'total_loss': total_loss_accum / num_batches,
            'loss_components': {},
            'log_sigmas': {},
            'scaled_loss': {}
        }
        
        for comp_key, comp_val in loss_components_accum.items():
            epoch_log_loss['loss_components'][comp_key] = comp_val / num_batches
        
        if self.fixed_task is None:
            for sigma_key, sigma_val in log_sigmas_accum.items():
                epoch_log_loss['log_sigmas'][sigma_key] = sigma_val / num_batches
            
            for scaled_key, scaled_val in log_scaled_accum.items():
                epoch_log_loss['scaled_loss'][scaled_key] = scaled_val / num_batches
        else:
            epoch_log_loss['log_sigmas'] = [None]
            epoch_log_loss['scaled_loss'] = [None]

        # If using DDP, globally average them
        if dist.is_initialized():
            epoch_log_loss = self.ddp_sync_epoch_log_loss(epoch_log_loss)


        # Return the final iteration index, the total train_loss, and the averaged log_loss
        # import pdb; pdb.set_trace()
        return i, train_loss, epoch_log_loss


    def v_loop(self, epoch):
        # Initialize the progress bar

        if self.is_main_process:
            val_pbar = tqdm(self.val_loader, total=len(self.val_loader),
                            desc=f"Val {epoch + 1}/{self.epochs}")
        else:
            val_pbar = self.val_loader  # a normal iterator, no progress bar
        
        # Set the model to eval mode and disable gradient computation
        self.model.eval()
        if self.val_mode == 'fp16' and self.train_mode != 'fp16':
            self.model.half()

        # Initialize accumulators
        val_loss = 0.0
        total_loss_accum = 0.0
        loss_components_accum = defaultdict(float)
        log_sigmas_accum = defaultdict(float)
        log_scaled_accum = defaultdict(float)
        
        with torch.no_grad():
            for j, (images, labels) in enumerate(val_pbar):
                images = images.to(self.device)
                if self.fixed_task is None:
                    labels = {key: value.to(self.device) for key, value in labels.items()}
                else:
                    labels = {self.fixed_task: labels[self.fixed_task].to(self.device)}
                
                # GET LOSS
                
                # A) FP 32
                if self.val_mode == 'fp32':
                    outputs = self.model(images)
                    loss, log_loss = self.get_loss(outputs, labels)
                    
                # B) AMP
                elif self.val_mode == 'amp':
                    with autocast(dtype=torch.float16):
                        outputs = self.model(images)
                        loss, log_loss = self.get_loss(outputs, labels)
                    
                # C) FP 16
                elif self.val_mode == 'fp16':                    
                    outputs = self.model(images.half())
                    outputs = {key: value.float() for key, value in outputs.items()}
                    loss, log_loss = self.get_loss(outputs, labels)

                # Accumulate overall validation loss
                val_loss += loss.item()
                
                # Accumulate the log_loss stats
                total_loss_accum += log_loss['total_loss']
                for comp_key, comp_val in log_loss['loss_components'].items():
                    loss_components_accum[comp_key] += comp_val
                    
                if self.fixed_task is None:
                    for sigma_key, sigma_val in log_loss['log_sigmas'].items():
                        log_sigmas_accum[sigma_key] += sigma_val
                    for scaled_key, scaled_val in log_loss['scaled_loss'].items():
                        log_scaled_accum[scaled_key] += scaled_val

                # Update progress bar
                if self.is_main_process and isinstance(val_pbar, tqdm):
                    if self.fixed_task is None:
                        val_pbar.set_postfix({
                            "loss": f"{val_loss / (j + 1):.4f}",
                            "lr": self.optimizer.param_groups[1]['lr']
                        })
                    else:
                        val_pbar.set_postfix({
                            "loss": f"{val_loss / (j + 1):.4f}",
                            "lr": self.optimizer.param_groups[0]['lr']
                        })



        if self.val_mode == 'fp16' and self.train_mode != 'fp16':
            self.model.float()

        # Visualization on the last batch (if enabled)
        if self.visualise_validation and self.is_main_process:
            # The loop ends with images, labels from the last batch
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            self.val_visualize(images, labels, outputs, name=f'/val_images/val_{epoch}')
        
        # Calculate averaged log_loss for the entire validation set
        num_val_batches = j + 1
        epoch_log_loss = {
            'total_loss': total_loss_accum / num_val_batches,
            'loss_components': {},
            'log_sigmas': {},
            'scaled_loss': {}
        }

        # Average each component
        for comp_key, comp_val in loss_components_accum.items():
            epoch_log_loss['loss_components'][comp_key] = comp_val / num_val_batches
        
        if self.fixed_task is None:
            # Average the log_sigma parameters
            for sigma_key, sigma_val in log_sigmas_accum.items():
                epoch_log_loss['log_sigmas'][sigma_key] = sigma_val / num_val_batches

            # Average the scaled loss components
            for scaled_key, scaled_val in log_scaled_accum.items():
                epoch_log_loss['scaled_loss'][scaled_key] = scaled_val / num_val_batches
            
        else:
            epoch_log_loss['log_sigmas'] = [None]
            epoch_log_loss['scaled_loss'] = [None]
            
        # Return j (last batch index), the cumulative validation loss, and the averaged log_loss
        # import pdb; pdb.set_trace()
        return j, val_loss, epoch_log_loss



    def val_visualize(self, images, labels, outputs, name):
        visualize.visualize_pretrain(x=images, y=labels, y_pred=outputs, images=5, apply_zoom=self.apply_zoom, 
                                     save_path=f"{self.out_folder}/{name}.png", fixed_task=self.fixed_task)


    def train(self):
        
        print("Starting training...")
        # torch.autograd.set_detect_anomaly(True)
        print("")
        
        # from accelerate import Accelerator
        # self.accelerator = Accelerator()
        # self.model, self.optimizer, self.train_loader = self.accelerator.prepare(self.model, self.optimizer, self.train_loader)

        # Additional accumulators for per-task losses
        # e.g. 'reconstruction', 'perceptual', 'climate', 'total_variation', 'geolocation', 'zoom_level'
        self.train_loss_components = defaultdict(list)
        self.val_loss_components = defaultdict(list)

        # For log_sigmas
        self.train_log_sigmas = defaultdict(list)
        self.val_log_sigmas = defaultdict(list)
        
        # For scaled losses
        self.train_scaled_losses = defaultdict(list)
        self.val_scaled_losses = defaultdict(list)

        # For plot losses
        self.train_plot_losses = defaultdict(list)
        self.val_plot_losses = defaultdict(list)

        # init model
        self.model.to(self.device)
        self.model.train()

        # create dst folder for generated files/artifacts
        os.makedirs(self.out_folder, exist_ok=True)
        s = self.scheduler
        
        # Training loop
        for epoch in range(self.epochs):

            if hasattr(self.train_loader, 'sampler') and isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)
            if hasattr(self.val_loader, 'sampler') and isinstance(self.val_loader.sampler, DistributedSampler):
                self.val_loader.sampler.set_epoch(epoch)

            if epoch == 0 and self.warmup == True:
                s = self.scheduler_warmup
                print('Starting linear warmup phase')
            elif epoch == self.warmup_steps and self.warmup == True:
                s = self.scheduler
                self.warmup = False
                print('Warmup finished')

            if any(not param.requires_grad for _, param in self.model.named_parameters()): raise ValueError("Some parameters do not require gradients!")
            if self.is_main_process:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    print(f"LR Group {i + 1}: {param_group['lr']}")

            i, train_loss, train_log_loss = self.t_loop(epoch, s)
            j, val_loss, val_log_loss = self.v_loop(epoch)

            self.tl.append(train_loss / (i + 1))
            self.vl.append(val_loss / (j + 1))
            self.lr.append(self.optimizer.param_groups[0]['lr'])
            for i, param_group in enumerate(self.optimizer.param_groups):
                current_lr = param_group["lr"]
                self.lr_groups[i].append(current_lr)

            # Store per-task losses
            for comp_key, comp_val in train_log_loss['loss_components'].items():
                self.train_loss_components[comp_key].append(comp_val)
            for comp_key, comp_val in val_log_loss['loss_components'].items():
                self.val_loss_components[comp_key].append(comp_val)
            self.val_loss_components['total_loss'].append(val_log_loss['total_loss'])
            
            # PRINT LOSSES
            if self.is_main_process and self.visualise_validation:
                print(" ")
                visualize.tabulate_losses(train_log_loss, val_log_loss, self.climate_segm, self.perceptual_loss)
                # import pdb; pdb.set_trace()
                
                # PRINT GRADIENTS
                print(" ")
                stats = visualize.collect_model_stats(self.model)
                visualize.print_stats_table(stats)
                print(" ")

            # import pdb; pdb.set_trace()

            # print(" ")
            # for name, param in self.model.named_parameters():
            #     if param.grad is not None:
            #         grad_norm = param.grad.data.norm(2).item()
            #         alert = "EXTREMELY LOW - ESSENTIALLY 0" if grad_norm < 1e-9 else "very low" if grad_norm < 1e-5 else "normal" if grad_norm < 0.5 else "high" if grad_norm < 0.98 else "!!EXTREMELY HIGH!! - HAD TO CLIP"
            #         print(f"{name}: Gradient Norm = {grad_norm:.2e} -- {alert}")
            #     else:
            #         print(f"param {name} has None weights")





            # Store log sigmas
            for sigma_key, sigma_val in train_log_loss['log_sigmas'].items():
                self.train_log_sigmas[sigma_key].append(sigma_val)
            for sigma_key, sigma_val in val_log_loss['log_sigmas'].items():
                self.val_log_sigmas[sigma_key].append(sigma_val)
            
            # Store log scaled losses
            for scaled_key, scaled_val in train_log_loss['scaled_loss'].items():
                self.train_scaled_losses[scaled_key].append(scaled_val)
            
            for scaled_key, scaled_val in val_log_loss['scaled_loss'].items():
                self.val_scaled_losses[scaled_key].append(scaled_val)

            # Store plot losses
            if epoch == 0:
                for comp_key, _ in train_log_loss['loss_components'].items():
                    self.train_plot_losses[comp_key].append(1.)
                for comp_key, _ in val_log_loss['loss_components'].items():
                    self.val_plot_losses[comp_key].append(1.)
            else:
                for comp_key, comp_val in train_log_loss['loss_components'].items():
                    self.train_plot_losses[comp_key].append(comp_val / self.train_loss_components[comp_key][0])
                for comp_key, comp_val in val_log_loss['loss_components'].items():
                    self.val_plot_losses[comp_key].append(comp_val / self.val_loss_components[comp_key][0])

            # Update the scheduler
            if self.warmup or self.lr_scheduler == 'cosine_annealing':
                s.step()
            elif self.lr_scheduler == 'reduce_on_plateau':
                if self.fixed_task is None:
                    current_val_loss_dict = {key: val[-1] for key, val in self.val_loss_components.items()}
                    s.step(current_val_loss_dict)
                else:
                    s.step(self.vl[-1])
                    
            #save check point
            self.save_ckpt_ddp(epoch, val_loss / (j + 1))

            # visualize loss & lr curves
            self.plot_curves(epoch)
            self.model.train()
            
            #save artifacts
            self.save_info(self.model_summary, self.n_shot, self.split_ratio, self.warmup, self.init_lr)

            # Early stopping
            if self.epochs_no_improve == self.early_stop:
                print(f'Early stopping triggered after {epoch + 1} epochs.')
                self.last_epoch = epoch + 1
                break



    def test(self):
        # Load the best weights
        self.model.load_state_dict(self.best_sd)

        print("Finished Training. Best epoch: ", self.best_epoch + 1)
        print("")
        print("Starting Testing...")
        self.model.eval()

        if self.is_main_process:
            test_pbar = tqdm(self.test_loader, total=len(self.test_loader),
                            desc=f"Test Set")
        else:
            test_pbar = self.test_loader  # a normal iterator, no progress bar

        with torch.no_grad():

            running_metric = self.get_metrics()

            for k, (images, labels) in enumerate(test_pbar):
                images = images.to(self.device)
                if self.fixed_task is None:
                    labels = {key: value.to(self.device) for key, value in labels.items()}
                else:
                    labels = {self.fixed_task: labels[self.fixed_task].to(self.device)}

                running_metric += self.get_metrics(images,labels)
                
                # if k == 10:
                #     break

            self.test_metrics = self.get_metrics(running_metric=running_metric, k=k)

            if self.visualise_validation and self.is_main_process:
                print(f"Test Loss: {self.test_metrics}")
                outputs = self.model(images)
                self.val_visualize(images, labels, outputs, name=f'test')

        if isinstance(self.model, nn.DataParallel) or isinstance(self.model, nn.parallel.DistributedDataParallel):
            model_sd = self.model.module.state_dict().copy()
        else:
            model_sd = self.model.state_dict().copy()

        torch.save(model_sd, os.path.join(self.out_folder, f"{self.name}_final.pt"))


    def inference(self, dataset = 'train'):
        # Load the best weights
        
        if dataset == 'inference':
            data_loader = self.inference_loader
        elif dataset == 'test':
            data_loader = self.test_loader
        elif dataset == 'val':
            data_loader = self.val_loader
        elif dataset == 'train':
            data_loader = self.train_loader
        else:
            raise ValueError('Invalid dataset name')
        
        if self.is_main_process:
            inf_pbar = tqdm(data_loader, total=len(data_loader),
                            desc=f"Inference ({dataset}) Set")
        else:
            inf_pbar = data_loader  # a normal iterator, no progress bar

        print("Starting Inference...")
        self.model.eval()
        with torch.no_grad():

            # running_metric = self.get_metrics()

            for k, (images, labels) in enumerate(inf_pbar):
                images = images.to(self.device)
                if self.fixed_task is None:
                    labels = {key: value.to(self.device) for key, value in labels.items()}
                else:
                    labels = {self.fixed_task: labels[self.fixed_task].to(self.device)}

                # GET LOSS
                
                # A) FP 32
                if self.val_mode == 'fp32':
                    outputs = self.model(images)
                    loss, log_loss = self.get_loss(outputs, labels)
                    
                # B) AMP
                elif self.val_mode == 'amp':
                    with autocast(dtype=torch.float16):
                        outputs = self.model(images)
                        loss, log_loss = self.get_loss(outputs, labels)
                    
                # C) FP 16
                elif self.val_mode == 'fp16':                    
                    outputs = self.model(images.half())
                    outputs = {key: value.float() for key, value in outputs.items()}
                    loss, log_loss = self.get_loss(outputs, labels)

                if torch.isnan(loss) or torch.isinf(loss):
                    print("Loss is NaN. Stopping training soon.")

                # running_metric += self.get_metrics(images,labels)
                
                # if k == 10:
                #     break

            # self.test_metrics = self.get_metrics(running_metric=running_metric, k=k)


    def get_metrics(self, images=None, labels=None, running_metric=None, k=None):
        """
        Compute metrics for a multi-task model with these tasks:
        - coords (regression): MSE, MAE
        - climate (classification): Pixel accuracy
        - reconstruction (regression): MSE
        - zoom_factor (regression): MSE, MAE

        The function supports three modes:
        1. Initialization (no images, no labels, no running_metric, no k):
        Returns a zero-initialized numpy array to accumulate metrics.
        2. Accumulation (images and labels provided):
        Compute batch metrics and return them as a numpy array to add to running totals.
        3. Finalization (running_metric and k provided):
        Compute averaged metrics from accumulated sums and return a dictionary.
        """

        if (running_metric is not None) and (k is not None):
            
            # Convert running_metric to a PyTorch tensor so we can all_reduce it
            metric_tensor = torch.tensor(running_metric, dtype=torch.float64, device=self.device)
            
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)

            if self.is_main_process:
                # Convert back to CPU/numpy
                total_metric = metric_tensor.cpu().numpy()

                denom = (k + 1) * self.world_size

                final_metrics = {}
                final_metrics['coords_mse']         = total_metric[0] / denom
                final_metrics['coords_mae']         = total_metric[1] / denom
                final_metrics['climate_acc']        = total_metric[2] / denom
                final_metrics['reduced_climate_acc'] = total_metric[4] / denom
                final_metrics['recon_mse']          = total_metric[3] / denom
                if self.apply_zoom:
                    final_metrics['zoom_mse']       = total_metric[5] / denom
                    final_metrics['zoom_mae']       = total_metric[6] / denom
                
                return final_metrics
            else:
                # On non-zero ranks, return an empty dict or None
                return {}

            # running_metric: array with sums of each metric
            # [coords_mse, coords_mae, climate_acc, recon_mse, zoom_mse, zoom_mae]
            final_metrics = {}
            final_metrics['coords_mse'] = running_metric[0] / (k + 1)
            final_metrics['coords_mae'] = running_metric[1] / (k + 1)
            final_metrics['climate_acc'] = running_metric[2] / (k + 1)
            final_metrics['reduced_climate_acc'] = running_metric[4] / (k + 1)
            final_metrics['recon_mse'] = running_metric[3] / (k + 1)
            if self.apply_zoom:
                final_metrics['zoom_mse'] = running_metric[5] / (k + 1)
                final_metrics['zoom_mae'] = running_metric[6] / (k + 1)

            return final_metrics

        elif (images is None) and (labels is None):
            # Initialize metrics accumulator
            # coords_mse, coords_mae, climate_acc, recon_mse, zoom_mse, zoom_mae
            metric_init = np.zeros(7, dtype=np.float64)
            return metric_init
        else:
            # Compute metrics for the given batch
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(images)

                # ----------- Coords (Regression) -----------
                # Suppose coords is shape (B, coords_dim)
                if self.fixed_task is None or self.fixed_task == 'coords':
                    pred_coords = outputs['coords']  # [B, C]
                else:
                    pred_coords = labels['coords']  # [B, C]
                true_coords = labels['coords']   # [B, C]
                coords_error = pred_coords - true_coords
                coords_mse = torch.mean(coords_error**2).item()
                coords_mae = torch.mean(torch.abs(coords_error)).item()

                # ----------- Climate (Classification) -----------
                # Suppose climate output is [B, classes, H, W]
                # and labels['climate'] is one-hot [B, classes, H, W].
                if self.fixed_task is None or self.fixed_task == 'climate':
                    pred_climate = outputs['climate']  # logits or probabilities, [B, classes, H, W]
                else:
                    pred_climate = labels['climate']
                true_climate = labels['climate']   # one-hot, [B, classes, H, W]
                
                # Get predicted class via argmax
                pred_climate_class = torch.argmax(pred_climate, dim=1)  # [B, H, W]
                # true_climate_class = torch.argmax(true_climate, dim=1)   # [B, H, W]
                

                # Compute pixel accuracy
                correct_pixels = (pred_climate_class == true_climate).float().sum().item()
                total_pixels = pred_climate_class.numel()
                climate_acc = correct_pixels / total_pixels

                map_array = np.array([
                    0,  # 0 -> water/no data
                    1,  # 1 -> tropical
                    1,  # 2 -> tropical
                    1,  # 3 -> tropical
                    2,  # 4 -> arid
                    2,  # 5 -> arid
                    2,  # 6 -> arid
                    2,  # 7 -> arid
                    3,  # 8 -> temperate
                    3,  # 9 -> temperate
                    3,  # 10 -> temperate
                    3,  # 11 -> temperate
                    3,  # 12 -> temperate
                    3,  # 13 -> temperate
                    3,  # 14 -> temperate
                    3,  # 15 -> temperate
                    3,  # 16 -> temperate
                    4,  # 17 -> cold
                    4,  # 18 -> cold
                    4,  # 19 -> cold
                    4,  # 20 -> cold
                    4,  # 21 -> cold
                    4,  # 22 -> cold
                    4,  # 23 -> cold
                    4,  # 24 -> cold
                    4,  # 25 -> cold
                    4,  # 26 -> cold
                    4,  # 27 -> cold
                    4,  # 28 -> cold
                    5,  # 29 -> polar
                    5   # 30 -> polar
                ], dtype=np.uint8)
                
                reduced_pred_climate_class = map_array[pred_climate_class.cpu().numpy()]
                reduced_true_climate = map_array[true_climate.cpu().numpy()]

                reduced_correct_pixels = np.sum(reduced_pred_climate_class == reduced_true_climate)
                reduced_total_pixels = reduced_pred_climate_class.size
                reduced_climate_acc = reduced_correct_pixels / reduced_total_pixels
                

                # ----------- Reconstruction (Regression) -----------
                # reconstruction is [B, C, H, W], can be MSE
                # If reconstruction target is None (some samples?), handle that:
                if self.fixed_task is None or self.fixed_task == 'reconstruction':
                    if labels['reconstruction'] is not None:
                        pred_recon = outputs['reconstruction']  # [B, C, H, W]
                        true_recon = labels['reconstruction']   # [B, C, H, W]
                        recon_error = pred_recon - true_recon
                        recon_mse = torch.mean(recon_error**2).item()
                    else:
                        # If reconstruction not used, set recon_mse to 0.0 or skip
                        recon_mse = 0.0
                else:
                    # If reconstruction not used, set recon_mse to 0.0 or skip
                    recon_mse = 0.0
                    
                    
                # ----------- Zoom Factor (Regression) -----------
                # zoom_factor is a scalar per sample (B, 1)
                if self.apply_zoom:
                    pred_zoom = outputs['zoom_factor']  # [B, 1]
                    true_zoom = labels['zoom_factor']   # [B, 1]
                    zoom_error = pred_zoom - true_zoom
                    zoom_mse = torch.mean(zoom_error**2).item()
                    zoom_mae = torch.mean(torch.abs(zoom_error)).item()
                else:
                    # If zoom factor not used, set zoom_mse to 0.0 or skip
                    zoom_mse = 0.0
                    zoom_mae = 0.0

                # Return array of sums for these metrics
                return np.array([
                    coords_mse,
                    coords_mae,
                    climate_acc,
                    recon_mse,
                    reduced_climate_acc,
                    zoom_mse,
                    zoom_mae
                ])

    def plot_curves(self, epoch):
        """
        1) Plot overall train/val loss.
        2) Plot each loss component (train vs. val), legend on the right with final value.
        3) Plot each log_sigma (train vs. val), legend on the right with final value.
        4) Plot learning rate.
        """
        self.e.append(epoch)

        # ---------------------------------------------------------------------
        # 1) Plot Overall Train/Val Loss
        # ---------------------------------------------------------------------
        fig = plt.figure()
        plt.plot(self.e, self.tl, label='Training Loss')
        plt.plot(self.e, self.vl, label='Validation Loss')
        plt.title('Overall Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.out_folder, "loss.png"))
        plt.close('all')

        if self.fixed_task is None:
            # ---------------------------------------------------------------------
            # 2) Plot Each Task-Specific Loss
            # ---------------------------------------------------------------------
            fig = plt.figure()
            plt.title('Task-Specific Loss Components')

            for comp_key in self.train_loss_components.keys():
                # Train line
                train_vals = self.train_loss_components[comp_key]
                if len(train_vals) > 0:
                    last_val_train = train_vals[-1]
                else:
                    last_val_train = 0.0
                train_label = f"Train {comp_key}: {last_val_train:.4f}"
                plt.plot(self.e, train_vals, label=train_label)
                
                # Validation line (if it exists)
                if comp_key in self.val_loss_components:
                    val_vals = self.val_loss_components[comp_key]
                    if len(val_vals) > 0:
                        last_val_val = val_vals[-1]
                    else:
                        last_val_val = 0.0
                    val_label = f"Val {comp_key}: {last_val_val:.4f}"
                    plt.plot(self.e, val_vals, label=val_label)

            plt.xlabel('Epoch')
            plt.ylabel('Loss Value')
            # Put legend outside on the right
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            # Use bbox_inches='tight' so the legend isn't cut off
            plt.savefig(os.path.join(self.out_folder, "loss_components.png"), bbox_inches='tight')
            plt.close('all')

            # ---------------------------------------------------------------------
            # 3) Plot Each Task-Specific Loss
            # ---------------------------------------------------------------------
            fig = plt.figure()
            plt.title('Task-Specific Loss Components (Normalized)')

            for comp_key in self.train_plot_losses.keys():
                # Train line
                train_vals = self.train_plot_losses[comp_key]
                if len(train_vals) > 0:
                    last_val_train = train_vals[-1]
                else:
                    last_val_train = 0.0
                train_label = f"Train {comp_key}: {last_val_train:.4f}"
                plt.plot(self.e, train_vals, label=train_label)
                
                # Validation line (if it exists)
                if comp_key in self.val_plot_losses:
                    val_vals = self.val_plot_losses[comp_key]
                    if len(val_vals) > 0:
                        last_val_val = val_vals[-1]
                    else:
                        last_val_val = 0.0
                    val_label = f"Val {comp_key}: {last_val_val:.4f}"
                    plt.plot(self.e, val_vals, label=val_label)

            plt.xlabel('Epoch')
            plt.ylabel('Loss Value')
            # Put legend outside on the right
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            # Use bbox_inches='tight' so the legend isn't cut off
            plt.savefig(os.path.join(self.out_folder, "loss_components_norm.png"), bbox_inches='tight')
            plt.close('all')

            # ---------------------------------------------------------------------
            # 4) Plot Each Log Sigma
            # ---------------------------------------------------------------------
            fig = plt.figure()
            plt.title('Log Sigmas')

            for sigma_key in self.train_log_sigmas.keys():
                # Train line
                train_vals = self.train_log_sigmas[sigma_key]
                if len(train_vals) > 0:
                    last_val_train = train_vals[-1]
                else:
                    last_val_train = 0.0
                train_label = f"Train {sigma_key}: {last_val_train:.4f}"
                plt.plot(self.e, train_vals, label=train_label)

                # Validation line (if it exists)
                if sigma_key in self.val_log_sigmas:
                    val_vals = self.val_log_sigmas[sigma_key]
                    if len(val_vals) > 0:
                        last_val_val = val_vals[-1]
                    else:
                        last_val_val = 0.0
                    val_label = f"Val {sigma_key}: {last_val_val:.4f}"
                    plt.plot(self.e, val_vals, label=val_label)

            plt.xlabel('Epoch')
            plt.ylabel('Log Sigma Value')
            # Put legend outside on the right
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(os.path.join(self.out_folder, "log_sigmas.png"), bbox_inches='tight')
            plt.close('all')


            # ---------------------------------------------------------------------
            # 5) Plot LR for Each Param Group
            # ---------------------------------------------------------------------
            group_names = {
                0: "     log(σ) - decay",
                1: "     log(σ) - no decay",
                2: "  encoder - decay",
                3: "  encoder - no decay",
                4: "  decoder - decay",
                5: "  decoder - no decay",
                6: "geo head - decay",
                7: "geo head - no decay",
                8: " rec head - decay",
                9: " rec head - no decay",
                10: "  cli head - decay",
                11: "  cli head - no decay"
            }

            fig = plt.figure()
            plt.title('Learning Rate Schedule (All Param Groups)')

            for scale in ["log", "linear"]:
                # Set y-scale
                plt.yscale(scale)

                for i, lr_values in enumerate(self.lr_groups):
                    if not lr_values:
                        continue

                    # Grab the most recent LR for the legend
                    last_lr = lr_values[-1]

                    # Determine legend label
                    # e.g. "log(\u03c3) LR: 0.000010" if group 0, or "encoder LR: 0.000100" for group 1, etc.
                    group_label = group_names.get(i, f"Group {i}")
                    label = f"{group_label} LR: {last_lr:.2e}"

                    # Plot (thicker line for group 0, if you want)
                    linewidth = 2 if i == 1 else 1
                    plt.plot(self.e, lr_values, label=label, linewidth=linewidth)

                plt.xlabel('Epoch')
                plt.ylabel('Learning Rate')
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                # Save with scale in filename
                filename = f"lr_{scale}.png"
                plt.savefig(os.path.join(self.out_folder, filename), bbox_inches='tight')

                # Clear legend and plots to avoid overlapping elements
                plt.clf()

            plt.close()

        else:
            # ---------------------------------------------------------------------
            # 4) Plot LR
            # ---------------------------------------------------------------------
            fig = plt.figure()
            plt.plot(self.e, self.lr, label='Learning Rate')
            plt.legend()
            plt.savefig(os.path.join(self.out_folder, f"lr.png"))
            plt.close('all')



    def save_info(self, model_summary=None, n_shot=None, p_split=None, warmup=None, lr=None):
        if self.test_metrics is not None:
            print("Saving artifacts...")
        
        if not self.is_main_process:
            return
        artifacts = {
            'training_parameters': {
                'model': self.name,
                'lr': lr,
                'scheduler': self.lr_scheduler,
                'warm_up': warmup,
                'optimizer': str(self.optimizer).split(' (')[0],
                'device': str(self.device),
                'training_epochs': self.epochs,
                'early_stop': self.early_stop,
                'train_samples': len(self.train_loader) * model_summary.input_size[0][0],
                'val_samples': len(self.val_loader) * model_summary.input_size[0][0],
                'test_samples': len(self.test_loader) * model_summary.input_size[0][0],
                'n_shot': n_shot,
                'p_split': p_split
            },
            'training_info': {
                'best_val_loss': self.best_loss,
                'best_epoch': self.best_epoch,
                'last_epoch': self.last_epoch
            },
            'test_metrics': self.test_metrics,
            'plot_info': {
                'epochs': self.e,
                'val_losses': self.vl,
                'train_losses': self.tl,
                'lr': self.lr_groups[1] if self.fixed_task is None else self.lr_groups[0],
                'train_loss_components': self.train_loss_components,
                'val_loss_components': self.val_loss_components,
                'train_log_sigmas': self.train_log_sigmas,
                'val_log_sigmas': self.val_log_sigmas,
                'train_scaled_losses': self.train_scaled_losses,
                'val_scaled_losses': self.val_scaled_losses,
                'train_plot_losses': self.train_plot_losses,
                'val_plot_losses': self.val_plot_losses,
                'lr_groups': self.lr_groups
            },
            'model_summary': {
                'batch_size': model_summary.input_size[0][0],
                'input_shape': model_summary.input_size[0],
                'input_size': model_summary.total_input,
                'total_mult_adds': model_summary.total_mult_adds,
                'back_forward_pass_size': model_summary.total_output_bytes,
                'param_bytes': model_summary.total_param_bytes,
                'trainable_params': model_summary.trainable_params,
                'non-trainable_params': model_summary.total_params - model_summary.trainable_params,
                'total_params': model_summary.total_params
            }
        }
        
        with open(f"{self.out_folder}/artifacts_{self.rank}.json", "w") as outfile:
            json.dump(artifacts, outfile, indent=4)

        if self.test_metrics is not None:
            print("Artifacts saved successfully.")

    def save_ckpt_ddp(self, epoch, val_loss):
        # Only save on the main process, so we don't overwrite from multiple ranks
        if not self.is_main_process:
            return

        # Move all parameters to CPU before saving
        model_sd_cpu = {k: v.cpu() for k, v in self.model.state_dict().items()}

        # Compare val_loss & decide best checkpoint
        if self.best_loss is None:
            self.best_epoch = epoch
            self.best_loss = val_loss
            torch.save(model_sd_cpu, os.path.join(self.out_folder, f"{self.name}_best.pt"))
            self.best_sd = model_sd_cpu
        elif self.best_loss > val_loss:
            self.best_epoch = epoch
            self.best_loss = val_loss
            self.epochs_no_improve = 0
            torch.save(model_sd_cpu, os.path.join(self.out_folder, f"{self.name}_best.pt"))
            self.best_sd = model_sd_cpu
        else:
            self.epochs_no_improve += 1

        # Always save a "last" checkpoint as well
        torch.save(model_sd_cpu, os.path.join(self.out_folder, f"{self.name}_last.pt"))



