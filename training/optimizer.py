"""
Custom Optimizers and Schedulers
"""

import torch
import torch.optim as optim


class CustomAdam(optim.Adam):
    """Custom Adam optimizer with weight decay"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)
        
    def step(self, closure=None):
        # TODO: Implement custom Adam step
        pass


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Learning rate scheduler with warmup"""
    
    def __init__(self, optimizer, warmup_steps, d_model, factor=1.0, 
                 last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.factor = factor
        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self):
        # TODO: Implement warmup learning rate calculation
        pass 