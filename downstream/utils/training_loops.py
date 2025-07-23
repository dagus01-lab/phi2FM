# Standard Library
import os
from tqdm import tqdm
import builtins

from matplotlib import pyplot as plt

# import PyQt5
# matplotlib.use('QtAgg') 
from tabulate import tabulate

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
# from torch.amp import GradScaler, autocast
from torchvision import transforms
import numpy as np
import json
import torch.distributed as dist

# utils
from utils import visualize
from utils import config_lc


def convert_to_builtin_types(obj):
    if isinstance(obj, dict):
        return {k: convert_to_builtin_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin_types(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_builtin_types(v) for v in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


class TrainBase():

    def __init__(self, model: nn.Module, device: torch.device, train_loader: DataLoader, val_loader: DataLoader,
                 test_loader: DataLoader, inference_loader: DataLoader, epochs:int = 50, early_stop:int=25, lr: float = 0.001, lr_scheduler: str = None, warmup:bool=True,
                 metrics: list = None, name: str="model", out_folder :str ="trained_models/", visualise_validation:bool=True, 
                 warmup_steps:int=5, warmup_gamma:int=10, pos_weight:np.array=None, weights:np.array=None, save_info_vars:tuple = None, apply_zoom:bool=False, 
                 climate_segm:bool=False, fixed_task:str=None, rank:int=None, min_lr:float=1e-6, perceptual_loss:bool=False, num_classes:int=4):
        
        self.train_mode = 'fp32' # choose between 'fp32', 'amp', 'fp16'
        self.val_mode = 'fp32' # choose between 'fp32', 'amp', 'fp16'
        
        if self.train_mode == 'fp16':
            self.model = model.half()
        else:
            self.model = model

        self.rank = rank
        self.num_classes = num_classes
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

            final_metrics = {'mse':running_metric[0] / (k + 1), 'mae':running_metric[1] / (k + 1), 'mave':running_metric[2] / (k + 1), 'acc':running_metric[3]/ (k + 1), 'precision':running_metric[4]/(running_metric[4]+running_metric[5]), 'recall':running_metric[4]/(running_metric[4]+running_metric[6]), 'baseline_mse':running_metric[7] / (k + 1)}
            final_metrics['f1'] = 2 * final_metrics['precision'] * final_metrics['recall'] / (final_metrics['precision'] + final_metrics['recall'])

            return final_metrics

        elif (images == None) and (labels == None):
            intermediary_values = ['mse','mae','mave','acc','tp','fp','fn','baseline_mse']
            metric_init = np.zeros(len(intermediary_values))
            return  metric_init
        
        
        else:            
            outputs = self.model(images)

            assert outputs.shape == labels.shape, (
                f"Shape mismatch in get_metrics:\n"
                f" - outputs shape: {outputs.shape}\n"
                f" - labels shape: {labels.shape}\n"
                f"Check model output dimensions and label preprocessing."
            )

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
                          desc=f"Train {epoch + 1}/{self.epochs}")

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
                          desc=f"Val {epoch + 1}/{self.epochs}")

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
                    "loss": f"{val_loss / (j + 1):.4f}",
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


    def save_info(self, model_summary=None, n_shot=None, p_split=None, warmup=None, lr=None):
        print("Saving artifacts...")
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
                'lr': self.lr
            },
            'model_summary': {
                'batch_size': model_summary.input_size[0],
                'input_size': model_summary.total_input,
                'total_mult_adds': model_summary.total_mult_adds,
                'back_forward_pass_size': model_summary.total_output_bytes,
                'param_bytes': model_summary.total_param_bytes,
                'trainable_params': model_summary.trainable_params,
                'non-trainable_params': model_summary.total_params - model_summary.trainable_params,
                'total_params': model_summary.total_params
            }
        }

        # Convert to JSON-serializable types
        artifacts_serializable = convert_to_builtin_types(artifacts)

        print("artifacts")
        with open(f"{self.out_folder}/artifacts.json", "w") as outfile:
            json.dump(artifacts_serializable, outfile, indent=4)
        print("Artifacts saved successfully.")



class TrainGeoLocate(TrainBase):
    def val_visualize(self, images, labels, outputs, name):
        visualize.visualize_geolocation(x=images, y=labels, y_pred=outputs, images=5,
                                              channel_first=True, save_path=f"{self.out_folder}/{name}.png")

    def get_metrics(self, images=None, labels=None, running_metric=None, k=None):
        
        if (running_metric is not None) and (k is not None):

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
            squared_error = error ** 2
            test_mse = squared_error.mean().item()
            test_mae = error.abs().mean().item()
            test_mave = torch.mean(torch.abs(outputs - labels)).item()

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


class TrainVAE(TrainBase):
    def __init__(self, *args, **kwargs):  # 2048 512
        super(TrainVAE, self).__init__(*args, **kwargs)
        self.CE_loss = nn.CrossEntropyLoss()
        self.MSE_loss = nn.MSELoss()
        self.augmentations = transforms.Compose([transforms.RandomVerticalFlip(p=0.5),
                                                 transforms.RandomHorizontalFlip(p=0.5),
                                                 transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), value='random'),
                                                 transforms.RandomApply([transforms.RandomResizedCrop(128, scale=(0.8, 1.0),
                                                                                                      ratio=(0.9, 1.1),
                                                                                                      interpolation=2,
                                                                                                      antialias=True),
                                                                         transforms.RandomRotation(degrees=20),
                                                                         transforms.GaussianBlur(kernel_size=3),
                                                                         ], p=0.2),

                                                 # transforms.ColorJitter(
                                                 #     brightness=0.25,
                                                 #     contrast=0.25,
                                                 #     saturation=0.5,
                                                 #     hue=0.05,),
                                                 # transforms.RandomAdjustSharpness(0.5, p=0.2),
                                                 # transforms.RandomAdjustSharpness(1.5, p=0.2),

                                                 ])



    def reconstruction_loss(self, reconstruction, original):
        # Binary Cross-Entropy with Logits Loss
        batch_size = original.size(0)


        # BCE = F.binary_cross_entropy_with_logits(reconstruction.reshape(batch_size, -1),
        #                                          original.reshape(batch_size, -1), reduction='mean')

        MSE = F.mse_loss(reconstruction.reshape(batch_size, -1), original.reshape(batch_size, -1), reduction='mean')
        # KLDIV = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE

    def similarity_loss(self, embeddings, embeddings_aug):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings_aug = F.normalize(embeddings_aug, p=2, dim=1)
        loss_cos = 1 - F.cosine_similarity(embeddings, embeddings_aug).mean()

        return loss_cos

    def cr_loss(self, mu, logvar, mu_aug, logvar_aug, gamma=1e-3, eps=1e-6):
        std_orig = logvar.exp() + eps
        std_aug = logvar_aug.exp() + eps

        _cr_loss = 0.5 * torch.sum(
            2 * torch.log(std_orig / std_aug) - 1 + (std_aug ** 2 + (mu_aug - mu) ** 2) / std_orig ** 2, dim=1).mean()
        cr_loss = _cr_loss * gamma

        return cr_loss

    def get_loss_aug(self, images, aug_images, labels):

        reconstruction, meta_data, latent = self.model(images)
        reconstruction_aug, meta_data_aug, latent_aug = self.model(aug_images)

        reconstruction_loss = (self.reconstruction_loss(reconstruction=reconstruction, original=images) +
                               self.reconstruction_loss(reconstruction=reconstruction_aug, original=aug_images)) / 2

        kg_labels = labels[:, :31]
        coord_labels = labels[:, 31:34]
        time_labels = labels[:, 34:]
        coord_out, time_out, kg_out = meta_data
        coord_out_aug, time_out_aug, kg_out_aug = meta_data_aug

        kg_loss = (self.CE_loss(kg_out, kg_labels) + self.CE_loss(kg_out_aug, kg_labels)) / 2
        coord_loss = (self.MSE_loss(coord_out, coord_labels) + self.MSE_loss(coord_out_aug, coord_labels)) / 2
        time_loss = (self.MSE_loss(time_out, time_labels) + self.MSE_loss(time_out_aug, time_labels)) / 2

        contrastive_loss = self.similarity_loss(latent, latent_aug)

        loss = reconstruction_loss + kg_loss + coord_loss + time_loss + contrastive_loss
        outputs = (reconstruction, meta_data, latent)

        return loss, reconstruction_loss, kg_loss, coord_loss, time_loss, contrastive_loss, outputs

    def get_loss(self, images, labels):
        reconstruction, meta_data, scale_skip_loss = self.model(images)

        reconstruction_loss = self.reconstruction_loss(reconstruction=reconstruction, original=images)

        kg_labels = labels[:, :31]
        coord_labels = labels[:, 31:34]
        time_labels = labels[:, 34:]
        coord_out, time_out, kg_out = meta_data

        kg_loss = self.CE_loss(kg_out, kg_labels)
        coord_loss = self.MSE_loss(coord_out, coord_labels)
        time_loss = self.MSE_loss(time_out, time_labels)

        # loss = 0.5*reconstruction_loss + 0.25*kg_loss + 0.125*coord_loss + 0.125*time_loss + scale_skip_loss
        loss = reconstruction_loss + kg_loss + coord_loss + time_loss + scale_skip_loss
        outputs = (reconstruction, meta_data, scale_skip_loss)

        return loss, reconstruction_loss, kg_loss, coord_loss, time_loss, scale_skip_loss, outputs

    def t_loop(self, epoch, s):
        # Initialize the running loss
        train_loss = 0.0
        train_reconstruction_loss = 0.0
        train_kg_loss = 0.0
        train_coord_loss = 0.0
        train_time_loss = 0.0
        train_scale_skip_loss = 0.0

        # Initialize the progress bar for training
        train_pbar = tqdm(self.train_loader, total=len(self.train_loader),
                          desc=f"Epoch {epoch + 1}/{self.epochs}")

        # loop training through batches
        for i, (images, labels) in enumerate(train_pbar):
            # Move inputs and targets to the device (GPU)
            images, labels = images.to(self.device), labels.to(self.device)


            # Zero the gradients
            self.optimizer.zero_grad()
            # get loss
            with autocast(dtype=torch.float16):
                loss, reconstruction_loss, kg_loss, coord_loss, time_loss, scale_skip_loss, outputs = self.get_loss(images, labels)
                # loss, outputs = self.get_loss(images, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            train_loss += loss.item()
            train_kg_loss += kg_loss.item()
            train_coord_loss += coord_loss.item()
            train_time_loss += time_loss.item()
            train_reconstruction_loss += reconstruction_loss.item()
            train_scale_skip_loss += scale_skip_loss.item()

            # display progress on console
            train_pbar.set_postfix({
                "loss": f"{train_loss / (i + 1):.4f}",
                "loss_kg": f"{train_kg_loss / (i + 1):.4f}",
                "loss_coord": f"{train_coord_loss / (i + 1):.4f}",
                "loss_time": f"{train_time_loss / (i + 1):.4f}",
                "loss_reconstruction": f"{train_reconstruction_loss / (i + 1):.4f}",
                "scale_skip_loss": f"{train_scale_skip_loss / (i + 1):.4f}",
                f"lr": self.optimizer.param_groups[0]['lr']})

            # # Update the scheduler
            if self.lr_scheduler == 'cosine_annealing':
                s.step()

            if (i % 10000) == 0 and i != 0:
                self.val_visualize(images, labels, outputs, name=f'/val_images/train_{epoch}_{i}')
                model_sd = self.model.state_dict()
                torch.save(model_sd, os.path.join(self.out_folder, f"{self.name}_ckpt.pt"))

        return i, train_loss

    def v_loop(self, epoch):

        # Initialize the progress bar for training
        val_pbar = tqdm(self.val_loader, total=len(self.val_loader),
                          desc=f"Epoch {epoch + 1}/{self.epochs}")

        with torch.no_grad():
            self.model.eval()
            val_loss = 0
            val_reconstruction_loss = 0.0
            val_kg_loss = 0.0
            val_coord_loss = 0.0
            val_time_loss = 0.0
            val_scale_skip_loss = 0.0

            for j, (images, labels) in enumerate(val_pbar):
                # Move inputs and targets to the device (GPU)
                images, labels = images.to(self.device), labels.to(self.device)

                # get loss
                loss, reconstruction_loss, kg_loss, coord_loss, time_loss, scale_skip_loss, outputs = self.get_loss(images, labels)

                val_loss += loss.item()
                val_kg_loss += kg_loss.item()
                val_coord_loss += coord_loss.item()
                val_time_loss += time_loss.item()
                val_reconstruction_loss += reconstruction_loss.item()
                val_scale_skip_loss += scale_skip_loss.item()

                # display progress on console
                val_pbar.set_postfix({
                    "val_loss": f"{val_loss / (j + 1):.4f}",
                    "loss_kg": f"{val_kg_loss / (j + 1):.4f}",
                    "loss_coord": f"{val_coord_loss / (j + 1):.4f}",
                    "loss_time": f"{val_time_loss / (j + 1):.4f}",
                    "loss_reconstruction": f"{val_reconstruction_loss / (j + 1):.4f}",
                    "scale_skip_loss": f"{val_scale_skip_loss / (j + 1):.4f}",
                    f"lr": self.optimizer.param_groups[0]['lr']})

            if self.visualise_validation:
                self.val_visualize(images, labels, outputs, name=f'/val_images/val_{epoch}')

            return j, val_loss

    def val_visualize(self, images, labels, outputs, name):
        visualize.visualize_vae(images=images, labels=labels, outputs=outputs, num_images=5, channel_first=True,
                                save_path=f"{self.out_folder}/{name}.png")

class TrainLandCover(TrainBase):

    def set_criterion(self):
        return nn.CrossEntropyLoss()

    def get_loss(self, images, labels):
        outputs = self.model(images)
        outputs = outputs.flatten(start_dim=2).squeeze()
        labels = labels.flatten(start_dim=1).squeeze()
        loss = self.criterion(outputs, labels)
        return loss

    def val_visualize(self, images, labels, outputs, name):
        visualize.visualize_lc(x=images, y=labels, y_pred=outputs.argmax(axis=1), images=5,
                               channel_first=True, vmin=0, save_path=f"{self.out_folder}/{name}.png")

    def get_metrics(self, images=None, labels=None, running_metric=None, k=None):
        # When a confusion matrix is provided, compute the metrics
        if (running_metric is not None) and (k is not None):
            
            confmat = running_metric
            
            # Total number of pixels
            total_pixels = np.sum(confmat)

            # True positives (TP) per class
            tp_per_class = np.diagonal(confmat)
            total_tp = tp_per_class.sum()

            # False positives (FP) and false negatives (FN) per class
            fp_per_class = confmat.sum(axis=0) - tp_per_class
            fn_per_class = confmat.sum(axis=1) - tp_per_class

            # Precision and recall for each class
            precision_per_class = tp_per_class / (fp_per_class + tp_per_class + 1e-12)
            recall_per_class    = tp_per_class / (fn_per_class + tp_per_class + 1e-12)

            # Micro-level precision and recall
            precision_micro = total_tp / (fp_per_class.sum() + total_tp + 1e-12)
            recall_micro    = total_tp / (fn_per_class.sum() + total_tp + 1e-12)

            # Macro-level precision and recall (simple mean)
            precision_macro = np.mean(precision_per_class)
            recall_macro    = np.mean(recall_per_class)

            # Accuracy across all pixels
            acc_total = total_tp / (total_pixels + 1e-12)

            # -----------------------------
            #  Compute F1-scores
            # -----------------------------
            # Per-class F1
            f1_per_class = 2 * (precision_per_class * recall_per_class) / (
                precision_per_class + recall_per_class + 1e-12
            )
            # Macro-F1 (average F1 across classes)
            f1_macro = np.mean(f1_per_class)
            # Micro-F1 (global precision & recall)
            f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro + 1e-12)

            # -----------------------------
            #  Compute MCC (Matthews Correlation Coefficient)
            # -----------------------------
            # For each class, treating it as a one-vs-all problem:
            tn_per_class = total_pixels - (tp_per_class + fp_per_class + fn_per_class)
            mcc_per_class = (tp_per_class * tn_per_class - fp_per_class * fn_per_class) / (
                np.sqrt((tp_per_class + fp_per_class) * (tp_per_class + fn_per_class) *
                        (tn_per_class + fp_per_class) * (tn_per_class + fn_per_class)) + 1e-12
            )

            # Macro-average MCC: simple mean of per-class MCCs
            mcc_macro = np.mean(mcc_per_class)
            # Weighted-average MCC: weight each class by its support (total true pixels for that class)
            support = confmat.sum(axis=1)  # true occurrences per class
            mcc_weighted = np.sum(mcc_per_class * support) / (support.sum() + 1e-12)

            # Micro-average MCC: computed using the multi-class MCC formula
            # c = sum(diagonal) and s = total samples
            s = total_pixels
            c = total_tp
            p = confmat.sum(axis=0)  # predicted totals per class
            t = confmat.sum(axis=1)  # true totals per class
            mcc_micro = (c * s - np.sum(p * t)) / (np.sqrt((s**2 - np.sum(p**2)) * (s**2 - np.sum(t**2))) + 1e-12)

            # -----------------------------
            #  Collate final metrics
            # -----------------------------
            final_metrics = {
                'acc': acc_total,
                'precision_per_class': precision_per_class.tolist(),
                'recall_per_class': recall_per_class.tolist(),
                'precision_micro': precision_micro,
                'precision_macro': precision_macro,
                'recall_micro': recall_micro,
                'recall_macro': recall_macro,
                'f1_per_class': f1_per_class.tolist(),
                'f1_micro': f1_micro,
                'f1_macro': f1_macro,
                'mcc_per_class': mcc_per_class.tolist(),
                'mcc_macro': mcc_macro,
                'mcc_weighted': mcc_weighted,
                'mcc_micro': mcc_micro,
                'conf_mat': confmat.tolist()
            }

            return final_metrics

        # If no images/labels are passed, just initialize the confusion matrix
        elif (images is None) and (labels is None):
            num_classes = len(config_lc.lc_raw_classes.keys())
            metric_init = np.zeros((num_classes, num_classes))
            return metric_init

        # Otherwise, compute the confusion matrix from model predictions
        else:
            outputs = self.model(images)
            outputs = outputs.argmax(axis=1).flatten()
            labels = labels.squeeze().flatten()

            num_classes = len(config_lc.lc_raw_classes.keys())
            unique_mapping = labels.to(torch.long) * num_classes + outputs.to(torch.long)
            bins = torch.bincount(unique_mapping, minlength=num_classes**2)
            cfm = bins.reshape(num_classes, num_classes)

            return cfm.cpu().numpy()

class TrainClassificationBuildings(TrainBase):

    def set_criterion(self):
        return nn.CrossEntropyLoss(weight=torch.tensor(self.weights))
        # return nn.CrossEntropyLoss(weight=torch.tensor([2.65209613e-01, 6.95524031e-01,
        #                                                 3.12650858e-02, 7.95257252e-03, 4.86978615e-05]))

    def get_loss(self, images, labels):
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        return loss

    def val_visualize(self, images, labels, outputs, name):
        visualize.visualize_building_classification(x=images, y=labels, y_pred=outputs, images=5,
                                              channel_first=True, num_classes=5,
                                              labels=['no urbanization', 'sparse urbanization',
                                                      'moderate urbanization', 'significant urbanization',
                                                      'extreme urbanization'],
                                              save_path=f"{self.out_folder}/{name}.png")

    def get_metrics(self, images=None, labels=None, running_metric=None, k=None):
        """
        Multi-label version of get_metrics.

        We store a confusion matrix of shape (num_classes, 2, 2), i.e., each class
        has its own 2×2 submatrix:

            [0, 0] -> TN
            [0, 1] -> FP
            [1, 0] -> FN
            [1, 1] -> TP

        The code structure mirrors the original function:
          - If running_metric and k are provided: compute final metrics from the aggregated CM
          - If images and labels are both None: return an initialized CM
          - Otherwise: return the per-batch CM for the current images/labels
        """

        # ------------------------------------------------------------------
        # CASE 1: FINAL METRICS
        # ------------------------------------------------------------------
        if (running_metric is not None) and (k is not None):
            # running_metric is our accumulated confusion matrix
            # Shape: (num_classes, 2, 2)
            confmat = running_metric
            num_classes = confmat.shape[0]

            # Extract per-class TP/FP/FN/TN
            # Indices: [actual, predicted]
            # confmat[i, 1, 1] = TP for class i
            # confmat[i, 0, 1] = FP
            # confmat[i, 1, 0] = FN
            # confmat[i, 0, 0] = TN
            tp_per_class = confmat[:, 1, 1]
            fp_per_class = confmat[:, 0, 1]
            fn_per_class = confmat[:, 1, 0]
            tn_per_class = confmat[:, 0, 0]

            # ---------------------------------------
            #  Compute per-class precision & recall
            # ---------------------------------------
            precision_per_class = tp_per_class / (tp_per_class + fp_per_class + 1e-12)
            recall_per_class    = tp_per_class / (tp_per_class + fn_per_class + 1e-12)

            # ---------------------------------------
            #  Compute micro precision & recall
            #     (sum over classes => treat as single binary problem)
            # ---------------------------------------
            sum_tp = tp_per_class.sum()
            sum_fp = fp_per_class.sum()
            sum_fn = fn_per_class.sum()
            sum_tn = tn_per_class.sum()

            precision_micro = sum_tp / (sum_tp + sum_fp + 1e-12)
            recall_micro    = sum_tp / (sum_tp + sum_fn + 1e-12)

            # ---------------------------------------
            #  Macro precision & recall
            # ---------------------------------------
            precision_macro = precision_per_class.mean()
            recall_macro    = recall_per_class.mean()

            # ---------------------------------------
            #  Overall accuracy
            #     = (TP + TN) / total
            # ---------------------------------------
            total_predictions = confmat.sum()  # sum of all 2×2 submatrices
            acc_total = (sum_tp + sum_tn) / (total_predictions + 1e-12)

            # ---------------------------------------
            #  F1-scores
            # ---------------------------------------
            # Per-class F1
            f1_per_class = 2 * (precision_per_class * recall_per_class) / (
                precision_per_class + recall_per_class + 1e-12
            )
            f1_macro = f1_per_class.mean()

            # Micro-F1
            f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro + 1e-12)

            # ---------------------------------------
            #  MCC (Matthews Correlation Coefficient)
            # ---------------------------------------
            # Per class
            mcc_per_class = (
                (tp_per_class * tn_per_class) - (fp_per_class * fn_per_class)
            ) / (
                np.sqrt(
                    (tp_per_class + fp_per_class)
                    * (tp_per_class + fn_per_class)
                    * (tn_per_class + fp_per_class)
                    * (tn_per_class + fn_per_class)
                ) + 1e-12
            )
            mcc_macro = mcc_per_class.mean()

            # Weighted-average MCC (by class support)
            support = tp_per_class + fn_per_class  # total actual positives for each class
            mcc_weighted = np.sum(mcc_per_class * support) / (support.sum() + 1e-12)

            # Micro-average MCC
            # Standard multi-label micro-MCC formula using the sums:
            numerator = (sum_tp * sum_tn) - (sum_fp * sum_fn)
            denominator = np.sqrt(
                (sum_tp + sum_fp)
                * (sum_tp + sum_fn)
                * (sum_tn + sum_fp)
                * (sum_tn + sum_fn)
            ) + 1e-12
            mcc_micro = numerator / denominator

            # ---------------------------------------
            #  Collate final metrics
            # ---------------------------------------
            final_metrics = {
                'acc': acc_total,
                'precision_per_class': precision_per_class.tolist(),
                'recall_per_class': recall_per_class.tolist(),
                'precision_micro': float(precision_micro),
                'precision_macro': float(precision_macro),
                'recall_micro': float(recall_micro),
                'recall_macro': float(recall_macro),
                'f1_per_class': f1_per_class.tolist(),
                'f1_micro': float(f1_micro),
                'f1_macro': float(f1_macro),
                'mcc_per_class': mcc_per_class.tolist(),
                'mcc_macro': float(mcc_macro),
                'mcc_weighted': float(mcc_weighted),
                'mcc_micro': float(mcc_micro),
                'conf_mat': confmat.tolist(),  # shape: (num_classes, 2, 2)
            }

            return final_metrics

        # ------------------------------------------------------------------
        # CASE 2: RETURN AN INITIAL (ZEROS) CONF MAT
        # ------------------------------------------------------------------
        elif (images is None) and (labels is None):
            # Suppose we know the number of classes in some config or otherwise
            num_classes = 11  # or len(config_lc.lc_raw_classes.keys())
            metric_init = np.zeros((num_classes, 2, 2), dtype=np.float32)
            return metric_init

        # ------------------------------------------------------------------
        # CASE 3: COMPUTE A PER-BATCH CONF MAT
        # ------------------------------------------------------------------
        else:
            # Model forward
            outputs = self.model(images)  # shape: (batch_size, num_classes)
            # Convert logits to binary predictions
            threshold = 0.5
            pred = (outputs > threshold).int()  # (batch_size, num_classes)
            true = labels.int()                 # (batch_size, num_classes)

            # Compute confusion matrix for each class
            batch_size, num_classes = true.shape
            cfm = torch.zeros(num_classes, 2, 2, device=self.device, dtype=torch.float)

            # Vectorized counts for each class:
            #   TP = (pred == 1 & true == 1)
            #   FP = (pred == 1 & true == 0)
            #   FN = (pred == 0 & true == 1)
            #   TN = (pred == 0 & true == 0)
            tp = (pred * true).sum(dim=0)  # shape: (num_classes,)
            fp = (pred * (1 - true)).sum(dim=0)
            fn = ((1 - pred) * true).sum(dim=0)
            tn = ((1 - pred) * (1 - true)).sum(dim=0)

            # Assign into cfm[i, ...]
            cfm[:, 1, 1] = tp
            cfm[:, 0, 1] = fp
            cfm[:, 1, 0] = fn
            cfm[:, 0, 0] = tn

            return cfm.cpu().numpy()




class TrainClassificationLC(TrainClassificationBuildings):

    def set_criterion(self):
        # return nn.CrossEntropyLoss()
        return nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
    
    def val_visualize(self, images, labels, outputs, name):
        visualize.visualize_lc_classification(x=images, y=labels, y_pred=outputs, images=5,
                                              channel_first=True, num_classes=11,
                                              labels=['Tree cover', 'Shrubland', 'Grassland', 'Cropland', 'Built-up',
                                                      'Bare/sparse', 'snow/ice','Perm water', 'Wetland', 'Mangroves',
                                                      'Moss'],
                                              save_path=f"{self.out_folder}/{name}.png")


class TrainClassificationRoads(TrainClassificationBuildings):

    def set_criterion(self):
        return nn.CrossEntropyLoss(weight=torch.tensor([0.37228453, 0.62771547]))

    def val_visualize(self, images, labels, outputs, name):
        visualize.visualize_lc_classification(x=images, y=labels, y_pred=outputs, images=5,
                                              channel_first=True, num_classes=2,
                                              labels=['No Roads', 'Roads'],
                                              save_path=f"{self.out_folder}/{name}.png")

class TrainSegmentationBurned(TrainBase):
    def set_criterion(self):
        return nn.CrossEntropyLoss(torch.tensor(self.weights))# nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
    def get_loss(self, images, labels):
        """
        images: [B, 10, 256, 256]       (example Sentinel‐2 input)
        labels: [B,  4, 256, 256]       (binary mask, channel‐first)
        """
        # 1) Forward pass → raw logits
        #print(images.shape)
         #print(outputs.shape)
        outputs = self.model(images)  # shape [B, 4, H, W]
    
        # 2) Convert one-hot (or channel‑first mask) to integer indices [B, H, W]
        #    If your labels come in as one-hot: labels.shape == [B, 4, H, W]
        labels_idx = torch.argmax(labels, dim=1).long()  # [B, H, W]
    
        # 3) Compute CE loss
        #    outputs: [B, C, H, W], labels_idx: [B, H, W]
        loss = self.criterion(outputs, labels_idx)
        return loss
        
        ## 1) Forward pass → raw logits
        #outputs = self.model(images)         # shape = [B, 4, 256, 256]
        ##print(outputs)
        ## 2) Flatten spatial dims → [B, 4, (256*256) = 65536]
        #outputs_flat = outputs.flatten(start_dim=2).permute(0, 2, 1)  # [B,65536,4]

        ## 3) Make sure labels are in [B, 4, 256, 256].
        #labels_flat = labels.flatten(start_dim=2).permute(0, 2, 1)     # [B,65536,4]
        ##print(self.pos_weight)
        ##print(outputs_flat.shape)
        ##print(labels_flat.shape)
        ## 4) Now shapes match perfectly: both [B, 4, 65536].
        #loss = self.criterion(outputs_flat, labels_flat)
        #return loss
    def val_visualize(self, images, labels, outputs, name, labels_list=['Background', 'Burned Area', 'Clouds', 'Waterbodies']):
        #print(outputs.shape)
        #print(labels.shape)
        outputs_tensor = torch.from_numpy(outputs)  # now a FloatTensor [B, C, H, W]
        outputs_labels = torch.argmax(outputs_tensor, dim=1)  # [B, H, W]
        labels_tensor = torch.from_numpy(labels)
        y_labels = torch.argmax(labels_tensor, dim=1)  # [B, H, W]
        #outputs_labels = torch.argmax(outputs, dim=1)  # [B, H, W] class indices
        visualize.visualize_burned_area(
            x=images,
            y=y_labels,      # labels to class indices for visualization
            y_pred=outputs_labels.cpu().numpy(),
            images=5,
            channel_first=True,
            #num_classes=4,
            labels=labels_list,
            save_path=f"{self.out_folder}/{name}.png"
        )
    def get_metrics(self, images=None, labels=None, running_metric=None, k=None):
        # When a confusion matrix is provided, compute the metrics
        if (running_metric is not None) and (k is not None):
            
            confmat = running_metric
            
            # Total number of pixels
            total_pixels = np.sum(confmat)

            # True positives (TP) per class
            tp_per_class = np.diagonal(confmat)
            total_tp = tp_per_class.sum()

            # False positives (FP) and false negatives (FN) per class
            fp_per_class = confmat.sum(axis=0) - tp_per_class
            fn_per_class = confmat.sum(axis=1) - tp_per_class

            # Precision and recall for each class
            precision_per_class = tp_per_class / (fp_per_class + tp_per_class + 1e-12)
            recall_per_class    = tp_per_class / (fn_per_class + tp_per_class + 1e-12)

            # Micro-level precision and recall
            precision_micro = total_tp / (fp_per_class.sum() + total_tp + 1e-12)
            recall_micro    = total_tp / (fn_per_class.sum() + total_tp + 1e-12)

            # Macro-level precision and recall (simple mean)
            precision_macro = np.mean(precision_per_class)
            recall_macro    = np.mean(recall_per_class)

            # Accuracy across all pixels
            acc_total = total_tp / (total_pixels + 1e-12)

            # -----------------------------
            #  Compute F1-scores
            # -----------------------------
            # Per-class F1
            f1_per_class = 2 * (precision_per_class * recall_per_class) / (
                precision_per_class + recall_per_class + 1e-12
            )
            # Macro-F1 (average F1 across classes)
            f1_macro = np.mean(f1_per_class)
            # Micro-F1 (global precision & recall)
            f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro + 1e-12)

            # -----------------------------
            #  Compute MCC (Matthews Correlation Coefficient)
            # -----------------------------
            # For each class, treating it as a one-vs-all problem:
            tn_per_class = total_pixels - (tp_per_class + fp_per_class + fn_per_class)
            mcc_per_class = (tp_per_class * tn_per_class - fp_per_class * fn_per_class) / (
                np.sqrt((tp_per_class + fp_per_class) * (tp_per_class + fn_per_class) *
                        (tn_per_class + fp_per_class) * (tn_per_class + fn_per_class)) + 1e-12
            )

            # Macro-average MCC: simple mean of per-class MCCs
            mcc_macro = np.mean(mcc_per_class)
            # Weighted-average MCC: weight each class by its support (total true pixels for that class)
            support = confmat.sum(axis=1)  # true occurrences per class
            mcc_weighted = np.sum(mcc_per_class * support) / (support.sum() + 1e-12)

            # Micro-average MCC: computed using the multi-class MCC formula
            # c = sum(diagonal) and s = total samples
            s = total_pixels
            c = total_tp
            p = confmat.sum(axis=0)  # predicted totals per class
            t = confmat.sum(axis=1)  # true totals per class
            mcc_micro = (c * s - np.sum(p * t)) / (np.sqrt((s**2 - np.sum(p**2)) * (s**2 - np.sum(t**2))) + 1e-12)

            # -----------------------------
            #  Collate final metrics
            # -----------------------------
            final_metrics = {
                'acc': acc_total,
                'precision_per_class': precision_per_class.tolist(),
                'recall_per_class': recall_per_class.tolist(),
                'precision_micro': precision_micro,
                'precision_macro': precision_macro,
                'recall_micro': recall_micro,
                'recall_macro': recall_macro,
                'f1_per_class': f1_per_class.tolist(),
                'f1_micro': f1_micro,
                'f1_macro': f1_macro,
                'mcc_per_class': mcc_per_class.tolist(),
                'mcc_macro': mcc_macro,
                'mcc_weighted': mcc_weighted,
                'mcc_micro': mcc_micro,
                'conf_mat': confmat.tolist()
            }

            return final_metrics

        # If no images/labels are passed, just initialize the confusion matrix
        elif (images is None) and (labels is None):
            num_classes = self.num_classes # len(config_lc.lc_raw_classes.keys())
            metric_init = np.zeros((num_classes, num_classes))
            return metric_init

        # Otherwise, compute the confusion matrix from model predictions
        else:
            outputs = self.model(images)
            outputs = torch.argmax(outputs, dim=1).long()
            labels = torch.argmax(labels, dim=1).long()
            print(outputs.shape)
            print(labels.shape)
            #outputs = outputs.argmax(axis=1).flatten()
            #labels = labels.squeeze().flatten()

            outputs = outputs.flatten()
            labels = labels.flatten()
            
            num_classes = self.num_classes
            unique_mapping = labels * num_classes + outputs
            bins = torch.bincount(unique_mapping, minlength=num_classes**2)
            cfm = bins.reshape(num_classes, num_classes)
            
            return cfm.cpu().numpy()

class TrainSegmentationWorldfloods(TrainSegmentationBurned):
    def val_visualize(self, images, labels, outputs, name):
        #print(outputs.shape)
        #print(labels.shape)
        outputs_tensor = torch.from_numpy(outputs)  # now a FloatTensor [B, C, H, W]
        outputs_labels = torch.argmax(outputs_tensor, dim=1)  # [B, H, W]
        labels_tensor = torch.from_numpy(labels)
        y_labels = torch.argmax(labels_tensor, dim=1)  # [B, H, W]
        #outputs_labels = torch.argmax(outputs, dim=1)  # [B, H, W] class indices
        visualize.visualize_burned_area(
            x=images,
            y=y_labels,      # labels to class indices for visualization
            y_pred=outputs_labels.cpu().numpy(),
            images=5,
            channel_first=True,
            num_classes=3,
            labels=['Clouds', 'Land', 'Water'],
            save_path=f"{self.out_folder}/{name}.png"
        )
class TrainClassificationFire(TrainClassificationBuildings):
    def set_criterion(self):
        # BCEWithLogitsLoss for binary segmentation
        print(self.weights)
        return nn.CrossEntropyLoss(torch.tensor(self.weights))
    def get_loss(self, images, labels):
        """
        For single‐label classification (one class per image), assume:
          - images: [B, C_in, H, W]
          - labels: [B] (integer class indices)
    
        Returns:
          - loss computed via CrossEntropyLoss
        """
        # 1) Forward pass → raw logits of shape [B, num_classes]
        outputs = self.model(images)  # assume model returns [B, num_classes]
        #print(f"Ok: {images.shape}, {labels.shape}")
        #print(outputs)
        #print(labels)
        # 2) Ensure labels are 1D long tensor of ints
        #    If labels currently come as shape [B, 1] or [B, 1, 1], squeeze to [B].
        if labels.ndim > 1:
            labels = labels.squeeze()
        labels = labels.long()  # CrossEntropyLoss needs LongTensor
    
        # 3) Compute classification loss
        loss = self.criterion(outputs, labels)
        return loss
    def val_visualize(self, images, labels, outputs, name):
        # Visualize classification results
        visualize.visualize_fire(
            x=images, 
            y=labels, 
            y_pred=outputs, 
            images=5,
            channel_first=True,
            #num_classes=4,
            labels=['safe', 'fire', 'burnt', 'water'],
            save_path=f"{self.out_folder}/{name}.png"
        )
    def get_metrics(self, images=None, labels=None, running_metric=None, k=None):
        """
        get_metrics for single-label multiclass classification.
    
        Confusion matrix shape: (num_classes, num_classes)
        Each [i, j] entry counts how often class i (true) was predicted as class j
    
        Final metrics computed from this confusion matrix.
        """
        import numpy as np
        import torch
    
        # ------------------------------------------------------------------
        # CASE 1: FINAL METRICS
        # ------------------------------------------------------------------
        if (running_metric is not None) and (k is not None):
            confmat = running_metric  # shape: (num_classes, num_classes)
            num_classes = confmat.shape[0]
    
            # True Positives, False Positives, etc., per class
            tp = np.diag(confmat)
            fp = confmat.sum(axis=0) - tp
            fn = confmat.sum(axis=1) - tp
            tn = confmat.sum() - (tp + fp + fn)
    
            precision_per_class = tp / (tp + fp + 1e-12)
            recall_per_class    = tp / (tp + fn + 1e-12)
    
            precision_micro = tp.sum() / (tp.sum() + fp.sum() + 1e-12)
            recall_micro    = tp.sum() / (tp.sum() + fn.sum() + 1e-12)
    
            precision_macro = precision_per_class.mean()
            recall_macro    = recall_per_class.mean()
    
            acc_total = tp.sum() / (confmat.sum() + 1e-12)
    
            f1_per_class = 2 * (precision_per_class * recall_per_class) / (
                precision_per_class + recall_per_class + 1e-12)
            f1_macro = f1_per_class.mean()
            f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro + 1e-12)
    
            mcc_per_class = (
                (tp * tn - fp * fn) /
                (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-12)
            )
            mcc_macro = mcc_per_class.mean()
            support = confmat.sum(axis=1)
            mcc_weighted = (mcc_per_class * support).sum() / (support.sum() + 1e-12)
    
            numerator = (tp.sum() * tn.sum()) - (fp.sum() * fn.sum())
            denominator = np.sqrt(
                (tp.sum() + fp.sum()) * (tp.sum() + fn.sum()) * (tn.sum() + fp.sum()) * (tn.sum() + fn.sum())
            ) + 1e-12
            mcc_micro = numerator / denominator
    
            final_metrics = {
                'acc': acc_total,
                'precision_per_class': precision_per_class.tolist(),
                'recall_per_class': recall_per_class.tolist(),
                'precision_micro': float(precision_micro),
                'precision_macro': float(precision_macro),
                'recall_micro': float(recall_micro),
                'recall_macro': float(recall_macro),
                'f1_per_class': f1_per_class.tolist(),
                'f1_micro': float(f1_micro),
                'f1_macro': float(f1_macro),
                'mcc_per_class': mcc_per_class.tolist(),
                'mcc_macro': float(mcc_macro),
                'mcc_weighted': float(mcc_weighted),
                'mcc_micro': float(mcc_micro),
                'conf_mat': confmat.tolist(),  # shape: (num_classes, num_classes)
            }
    
            return final_metrics
    
        # ------------------------------------------------------------------
        # CASE 2: INITIAL CONFUSION MATRIX
        # ------------------------------------------------------------------
        elif (images is None) and (labels is None):
            num_classes = 4
            return np.zeros((num_classes, num_classes))
        elif (images is not None) and (labels is not None):
            self.model.eval()
            outputs = self.model(images)
            preds = torch.argmax(outputs, dim=1)
        
            num_classes = 4
        
            # Compute confusion matrix for the batch
            confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confmat[t.long(), p.long()] += 1
        
            return confmat.cpu().numpy().astype(np.int64)
    
        # ------------------------------------------------------------------
        # CASE 3: COMPUTE BATCH CONFUSION MATRIX
        # ------------------------------------------------------------------
        else:
            outputs = self.model(images)  # shape: (batch_size, num_classes)
            pred = torch.argmax(outputs, dim=1)  # (batch_size,)
            true = labels.long()                # (batch_size,)
    
            num_classes = outputs.shape[1]
            confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=self.device)
    
            for t, p in zip(true, pred):
                confmat[t, p] += 1
    
            return confmat.cpu().numpy().astype(np.int64)

class TrainCloudSegmentation(TrainSegmentationBurned):
    
    def val_visualize(self, images, labels, outputs, name):
        # Visualize binary segmentation results
        outputs_tensor = torch.from_numpy(outputs)  # now a FloatTensor [B, C, H, W]
        outputs_labels = torch.argmax(outputs_tensor, dim=1)  # [B, H, W]
        labels_tensor = torch.from_numpy(labels)
        y_labels = torch.argmax(labels_tensor, dim=1)  # [B, H, W]
        #outputs_labels = torch.argmax(outputs, dim=1)  # [B, H, W] class indices
        visualize.visualize_burned_area(
            x=images,
            y=y_labels,      # labels to class indices for visualization
            y_pred=outputs_labels.cpu().numpy(),
            images=5,
            channel_first=True,
            num_classes=5,
            labels=['No cloud', 'Cloud', 'value2', 'value3', 'value4'],
            save_path=f"{self.out_folder}/{name}.png"
        )
            
class TrainViT(TrainBase):
    def get_loss(self, images, labels):
        outputs = self.model(images)
        labels = self.model.patchify(labels)
        loss = self.criterion(outputs, labels)
        return loss

    def val_visualize(self, images, labels, outputs, name):
        outputs = self.model.unpatchify(torch.from_numpy(outputs), c=labels.shape[1])
        visualize.visualize(x=images, y=labels, y_pred=outputs.detach().cpu().numpy(), images=5,
                               channel_first=True, vmin=0, save_path=f"{self.out_folder}/{name}.png")

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
                outputs = self.model(images[:, :, 16:-16, 16:-16])

                if type(outputs) is tuple:
                    outputs = outputs[0]

                self.val_visualize(images.detach().cpu().numpy(), labels.detach().cpu().numpy(), outputs.detach().cpu().numpy(), name=f'/val_images/val_{epoch}')

            return j, val_loss


class TrainSatMAE(TrainBase):
    def get_loss(self, images, labels):
        images = images[:, :, 16:-16, 16:-16]
        labels = labels[:, :, 16:-16, 16:-16]
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        return loss

    def val_visualize(self, images, labels, outputs, name):
        images = images[:, :, 16:-16, 16:-16]
        labels = labels[:, :, 16:-16, 16:-16]
        visualize.visualize(x=images, y=labels, y_pred=outputs.detach().cpu().numpy(), images=5,
                               channel_first=True, vmin=0, save_path=f"{self.out_folder}/{name}.png")

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
                outputs = self.model(images[:, :, 16:-16, 16:-16])

                if type(outputs) is tuple:
                    outputs = outputs[0]

                self.val_visualize(images.detach().cpu().numpy(), labels.detach().cpu().numpy(), outputs.detach().cpu().numpy(), name=f'/val_images/val_{epoch}')

            return j, val_loss


class TrainSatMAE_lc(TrainLandCover):
    def get_loss(self, images, labels):
        images = images[:, :, 16:-16, 16:-16]
        labels = labels[:, :, 16:-16, 16:-16]
        outputs = self.model(images)
        outputs = outputs.flatten(start_dim=2).squeeze()
        labels = labels.flatten(start_dim=1).squeeze()
        loss = self.criterion(outputs, labels)
        return loss

    def val_visualize(self, images, labels, outputs, name):
        images = images[:, :, 16:-16, 16:-16]
        labels = labels[:, :, 16:-16, 16:-16]
        visualize.visualize_lc(x=images, y=labels, y_pred=outputs.argmax(axis=1), images=5,
                               channel_first=True, vmin=0, save_path=f"{self.out_folder}/{name}.png")

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
                outputs = self.model(images[:, :, 16:-16, 16:-16])

                if type(outputs) is tuple:
                    outputs = outputs[0]

                self.val_visualize(images.detach().cpu().numpy(), labels.detach().cpu().numpy(), outputs.detach().cpu().numpy(), name=f'/val_images/val_{epoch}')

            return j, val_loss

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
                images = images[:, :, 16:-16, 16:-16].to(self.device)
                labels = labels[:, :, 16:-16, 16:-16].to(self.device)

                running_metric += self.get_metrics(images, labels)

            self.test_metrics = self.get_metrics(running_metric=running_metric, k=k)

            print(f"Test Loss: {self.test_metrics}")
            outputs = self.model(images)
            self.val_visualize(images.detach().cpu().numpy(), labels.detach().cpu().numpy(),
                               outputs.detach().cpu().numpy(), name='test')


class TrainViTLandCover(TrainBase):

    def set_criterion(self):
        return nn.CrossEntropyLoss()

    def get_loss(self, images, labels):
        outputs = self.model.unpatchify(self.model(images), c=11).flatten(start_dim=2).squeeze()
        labels = labels.flatten(start_dim=1).squeeze()
        loss = self.criterion(outputs, labels)
        return loss

    def val_visualize(self, images, labels, outputs, name):
        outputs = self.model.unpatchify(torch.from_numpy(outputs), c=11)
        visualize.visualize_lc(x=images, y=labels, y_pred=outputs.detach().cpu().numpy().argmax(axis=1), images=5,
                               channel_first=True, vmin=0, save_path=f"{self.out_folder}/{name}.png")

    def get_metrics(self, images=None, labels=None, running_metric=None, k=None):
        
        if (running_metric is not None) and (k is not None):
            metric_names = ['acc','precision','recall','baseline_mse']
            # intermediary_values = ['confusion_matrix']

            confmat = running_metric

            total_pixels = np.sum(confmat)
            
            tp_per_class = np.diagonal(confmat)
            total_tp = tp_per_class.sum()

            fp_per_class = confmat.sum(axis=0) - tp_per_class
            fn_per_class = confmat.sum(axis=1) - tp_per_class
            

            precision_per_class = tp_per_class/(fp_per_class+tp_per_class)
            recall_per_class = tp_per_class/(fn_per_class+tp_per_class)

            precision_micro = total_tp/(fp_per_class.sum() + total_tp)
            recall_micro = total_tp/(fn_per_class.sum() + total_tp)
            precision_macro = np.mean(precision_per_class)
            recall_macro = np.mean(recall_per_class)

            acc_total = total_tp/total_pixels

            final_metrics = {'acc':acc_total, 'precision_per_class':precision_per_class.tolist(),'recall_per_class':recall_per_class.tolist() ,'precision_micro':precision_micro, 'precision_macro':precision_macro, 'recall_micro':recall_micro, 'recall_macro':recall_macro, 'conf_mat':confmat.tolist()}

            return final_metrics


        elif (images == None) and (labels == None):
            intermediary_values = ['confusion_matrix']
            num_classes = len(config_lc.lc_raw_classes.keys())
            metric_init = np.zeros((num_classes,num_classes)) # 
            return  metric_init
        
        
        else:
            outputs = self.model.unpatchify(self.model(images), c=11)
            outputs = outputs.argmax(axis=1).flatten()
            labels = labels.squeeze().flatten()
            
            # stolen from pytorch confusion matrix
            num_classes = len(config_lc.lc_raw_classes.keys())
            unique_mapping = labels.to(torch.long) * num_classes + outputs.to(torch.long)
            bins = torch.bincount(unique_mapping, minlength=num_classes**2) 
            cfm = bins.reshape(num_classes, num_classes)

            return cfm.cpu().numpy()