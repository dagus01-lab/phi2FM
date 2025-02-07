import math
import torch
from torch.optim import Optimizer

import math
from torch.optim import Optimizer

class MultiMetricScheduler:
    """
    A more general multi-metric, multi-group scheduler that can either:

    1) Reduce LR on plateau (per param group).
    2) Use cosine annealing (per param group).

    Example usage:
    --------------
        param_groups_config = {
            0: {  
                # param_group 0 (stem & encoder)
                'scheduler_type': 'reduce_on_plateau',
                'metrics': ['total_loss'],
                'factor': 0.1,
                'patience': 6,
                'min_lr': 1e-6,
                'mode': 'min',
            },
            1: {
                # param_group 1 (decoder)
                'scheduler_type': 'reduce_on_plateau',
                'metrics': ['reconstruction', 'perceptual', 'climate_segmentation'],
                'factor': 0.1,
                'patience': 6,
                'min_lr': 1e-6,
                'mode': 'min',
            },
            2: {
                # param_group 2 uses cosine annealing
                'scheduler_type': 'cosine_annealing',
                'T_max': 50,         # required by cosine annealing
                'eta_min': 1e-6,     # minimum lr for cosine annealing
                'metrics': None,     # not required for cosine, but must define key
            },
            # ... and so on
        }

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = MultiMetricScheduler(optimizer, param_groups_config)

    Then after each validation epoch:
        val_loss_dict = {...}
        scheduler.step(val_loss_dict)
    """

    def __init__(self, optimizer: Optimizer, param_groups_config: dict):
        """
        Args:
            optimizer (Optimizer): The optimizer with multiple param groups.
            param_groups_config (dict): A dict mapping group index -> config, e.g.:
                {
                  0: {
                    'scheduler_type': 'reduce_on_plateau' or 'cosine_annealing',
                    'metrics': [...],  # only relevant for reduce_on_plateau
                    'factor': ...,
                    'patience': ...,
                    'min_lr': ...,
                    'mode': 'min' or 'max',
                    'T_max': ...,    # only relevant for cosine_annealing
                    'eta_min': ...
                  },
                  1: {
                    ...
                  }
                }
        """
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")

        self.optimizer = optimizer
        self.param_groups_config = param_groups_config
        self.num_groups = len(self.optimizer.param_groups)

        # Tracking for "reduce_on_plateau"
        self.best_values = [math.inf] * self.num_groups
        self.num_bad_epochs = [0] * self.num_groups

        # Tracking for "cosine_annealing"
        # We'll track a global epoch. If needed, you can track separate epochs for each group.
        self.global_epoch = 1

        # Set defaults for each group if not specified
        for i in range(self.num_groups):
            if i not in self.param_groups_config:
                # Default to reduce-on-plateau on 'total_loss'
                self.param_groups_config[i] = {
                    'scheduler_type': 'cosine_annealing',
                    'metrics': ['total_loss'],
                    'factor': 0.1,
                    'patience': 6,
                    'min_lr': 1e-6,
                    'mode': 'min',
                    # Cosine defaults (ignored if scheduler_type != 'cosine_annealing')
                    'T_max': 10,
                    'eta_min': 1e-6,
                }
            else:
                # Fill in missing keys with defaults
                cfg = self.param_groups_config[i]
                cfg.setdefault('scheduler_type', 'cosine_annealing')
                cfg.setdefault('metrics', ['total_loss'])
                cfg.setdefault('factor', 0.1)
                cfg.setdefault('patience', 6)
                cfg.setdefault('min_lr', 1e-6)
                cfg.setdefault('mode', 'min')
                # Cosine-annealing specific
                cfg.setdefault('T_max', 10)
                cfg.setdefault('eta_min', 1e-6)

    def step(self, val_loss_dict: dict = None):
        """
        Step the scheduler for each param group.

        For 'reduce_on_plateau':
            - Check improvement on the sum of the configured metric(s).
            - If no improvement for `patience` epochs, reduce LR.

        For 'cosine_annealing':
            - Update LR via the standard PyTorch formula.

        Args:
            val_loss_dict (dict): dictionary of validation losses, e.g.:
                {
                  'total_loss': 1.0,
                  'reconstruction': 0.45,
                  'perceptual': 0.32,
                  ...
                }
                (Optional for 'cosine_annealing', required for 'reduce_on_plateau')
        """
        for group_idx, param_group in enumerate(self.optimizer.param_groups):
            config = self.param_groups_config[group_idx]
            scheduler_type = config['scheduler_type'].lower()
            # import pdb; pdb.set_trace()

            if scheduler_type == 'reduce_on_plateau':
                # Safety check
                if val_loss_dict is None:
                    raise ValueError(
                        "val_loss_dict is required for 'reduce_on_plateau' scheduler."
                    )

                metrics = config['metrics']
                factor = config['factor']
                patience = config['patience']
                min_lr = config['min_lr']
                mode = config['mode']

                # Combine the relevant metrics (e.g. sum). 
                # If you'd prefer average, just divide by len(metrics).
                current_val = 0.0
                for m in metrics:
                    current_val += val_loss_dict.get(m, 0.0)

                # Check improvement
                is_improvement = False
                if mode == 'min':
                    if current_val < self.best_values[group_idx]:
                        is_improvement = True
                else:  # 'max'
                    if current_val > self.best_values[group_idx]:
                        is_improvement = True

                # Update best and num_bad_epochs
                if is_improvement:
                    self.best_values[group_idx] = current_val
                    self.num_bad_epochs[group_idx] = 0
                else:
                    self.num_bad_epochs[group_idx] += 1

                # If patience exceeded, reduce LR for this group
                if self.num_bad_epochs[group_idx] > patience:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * factor, min_lr)
                    if old_lr > new_lr:
                        param_group['lr'] = new_lr
                    # Reset or continue counting bad epochs
                    self.num_bad_epochs[group_idx] = 0

            elif scheduler_type == 'cosine_annealing':
                # We will mimic PyTorch's CosineAnnealingLR logic:
                # new_lr = eta_min + (initial_lr - eta_min) * 
                #          (1 + cos(pi * (current_epoch / T_max))) / 2
                T_max = config['T_max']
                eta_min = config['eta_min']

                # If you want to store the "initial_lr" somewhere,
                # you could store it in the config dict the first time.
                if 'initial_lr' not in config:
                    config['initial_lr'] = param_group['lr']

                initial_lr = config['initial_lr']
                # Cosine formula
                cos_inner = math.pi * self.global_epoch / T_max
                cos_out = (1 + math.cos(cos_inner)) / 2
                new_lr = eta_min + (initial_lr - eta_min) * cos_out

                param_group['lr'] = new_lr
                

            else:
                raise ValueError(
                    f"Unknown scheduler_type '{scheduler_type}' for param group {group_idx}"
                )
            
        # Increment the global epoch (shared across all param groups)
        self.global_epoch += 1 # increment here and don't take as argument in case there is warmup
        if self.global_epoch == T_max + 1:
            # Reset epoch
            self.global_epoch = 1
