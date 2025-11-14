import torch
import math
import numpy as np


def adjust_learning_rate(
        optimizer: torch.optim,
        epoch: int,
        total_epoch: int,
        lr_scheduler: str='cosine',
        lr_init=1e-4,
        lr_min=1e-6,
        verbose: bool=False
    ):

    if lr_scheduler == 'constant':
        lr = lr_init
    elif lr_scheduler == 'cosine':
        phase = math.pi * epoch / (total_epoch - 1)
        cosine_decay = 0.5 * (1 + math.cos(phase))
        lr = (lr_init - lr_min) * cosine_decay + lr_min
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if verbose:
        print(f'{lr_scheduler} scheduler | Updating learning rate to {lr:.4e}')

