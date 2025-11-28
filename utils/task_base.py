
import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
from typing import Union

class Task():
    def __init__(self, args):
        self.args = args
        self.device = self._get_device()
        self.model = None

        if hasattr(args, "root_path"):
            self.root_path = args.root_path
            self.data_path = os.path.join(self.root_path, getattr(args, "data_path", "dataset"))
            self.result_path = os.path.join(self.root_path, getattr(args, "result_path", "results"))
            self.ckpt_path = os.path.join(self.root_path, getattr(args, "ckpt_path", "checkpoints"))
            os.makedirs(self.data_path, exist_ok=True)
            os.makedirs(self.result_path, exist_ok=True)
            os.makedirs(self.ckpt_path, exist_ok=True)
        
        self._init_lr()
        self.seed_fix()
        
    def count_parameters(self):
        assert self.model is not None, "Model is not defined"
        params = [p.numel() for p in self.model.parameters() if p.requires_grad]
        print(f"Number of Trainable Params: {sum(params)}")    

    def print_model(self):
        assert self.model is not None, "Model is not defined"
        print(str(self.model))

    def seed_fix(self):
        seed = self.args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _init_lr(self):
        if self.args.lr is None:
            eff_batch_size = self.args.batch_size * getattr(self.args, "accum_iter", 1)
            self.args.lr = self.args.blr * eff_batch_size / 256
        if not hasattr(self.args, "lr_min") or self.args.lr_min is None:
            self.args.lr_min = self.args.lr * 0.01

    def _get_device(self):
        if self.args.device.lower() == 'cuda' and torch.cuda.is_available():
            device = torch.device(f"cuda:{getattr(self.args, 'gpu_num', 0)}")
        else:
            device = torch.device('cpu')
        return device
    
    def _select_optimizer(self):
        assert self.model is not None, "Model is not defined"

        if self.args.optimizer == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9)
        elif self.args.optimizer == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, betas=tuple(self.args.betas))
        elif self.args.optimizer == "adamw":
            optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, betas=tuple(self.args.betas))
        return optimizer
    
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def _save_checkpoint(self, 
                         model: nn.Module, 
                         optimizer: optim.Optimizer, 
                         epoch: int, 
                         scaler=None, 
                         verbose=False):
        ckpt_name = f"{self.args.model.lower()}_e{epoch}_s{self.args.seq_len}_p{self.args.pred_len}.pth"
        save_path = os.path.join(self.ckpt_path, ckpt_name)

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(self.args)
        }

        if scaler is not None:
            checkpoint["scaler"] = scaler.state_dict()

        torch.save(checkpoint, save_path)
        if verbose:
            print(f"Checkpoint saved at {save_path}")

    def _load_checkpoint(self, 
                         ckpt_name: str, 
                         model: nn.Module, 
                         optimizer: Union[optim.Optimizer, None]=None, 
                         scaler=None, 
                         verbose=True):
        ckpt_list = list(map(lambda x: os.path.splitext(x)[0], os.listdir(self.ckpt_path)))
        assert ckpt_name in ckpt_list, f"{ckpt_name} is not exist in {self.ckpt_path}"
        
        load_path = os.path.join(self.ckpt_path, ckpt_name + ".pth")
        checkpoint = torch.load(load_path, map_location="cpu", weights_only=True)

        model.load_state_dict(checkpoint["model"])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        self.args.ckpt_args = checkpoint.get("args", None)

        if verbose:
            print(f"Loaded checkpoint from {load_path}")
        return checkpoint.get("epoch", 0)
    
    def _build_model(self):
        pass

    def _get_data_loader(self):
        pass

    def train(self, val: bool=True):
        pass

    def val(self):
        pass

    def test(self):
        pass