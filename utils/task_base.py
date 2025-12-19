
import torch
import torch.nn as nn
import torch.optim as optim

import os
import json
import numpy as np
from typing import Union

from utils.distributed import (
    is_main_process,
    get_rank,
    print_rank0
)

class Task():
    def __init__(self, args):
        self.args = args
        self.device = self._get_device()
        self.model = None

        if hasattr(args, "root_path"):
            self.root_path = args.root_path
            self.data_path = os.path.join(self.root_path, getattr(args, "data_path", "dataset"))
            self.result_path = os.path.join(self.root_path, getattr(args, "result_path", "results"))

            # Build checkpoint path based on dataset type
            base_ckpt = getattr(args, "ckpt_path", "checkpoints")

            # If user provides a custom ckpt_path (not default), use it directly
            if base_ckpt != "checkpoints":
                self.ckpt_path = os.path.join(self.root_path, base_ckpt)
            else:
                # Auto-build path based on dataset type
                dataset = getattr(args, "dataset", "").lower()
                city = getattr(args, "city", "").lower()
                model_name = getattr(args, "model", "model").lower()

                if dataset == "weather" and city:
                    # Determine dataset type from city
                    korean_cities = ('korea', 'seoul', 'busan', 'daegu', 'gangneung', 'gwangju')
                    global_cities = ('global', 'berlin', 'la', 'newyork', 'tokyo')

                    if city in korean_cities:
                        dataset_type = 'korean'
                    elif city in global_cities:
                        dataset_type = 'global'
                    else:
                        dataset_type = city

                    # Build path: checkpoints/{dataset_type}/{model}_v1
                    self.ckpt_path = os.path.join(self.root_path, base_ckpt, dataset_type, f"{model_name}_v1")
                else:
                    self.ckpt_path = os.path.join(self.root_path, base_ckpt)

            os.makedirs(self.data_path, exist_ok=True)
            os.makedirs(self.result_path, exist_ok=True)
            os.makedirs(self.ckpt_path, exist_ok=True)
        
        self._init_lr()
        self.seed_fix()
        
    def count_parameters(self):
        assert self.model is not None, "Model is not defined"
        # Handle DDP model wrapper
        model_to_count = self.model.module if hasattr(self.model, 'module') else self.model
        params = [p.numel() for p in model_to_count.parameters() if p.requires_grad]
        print_rank0(f"Number of Trainable Params: {sum(params)}")    

    def print_model(self):
        assert self.model is not None, "Model is not defined"
        if is_main_process():
            model_to_print = self.model.module if hasattr(self.model, 'module') else self.model
            print(str(model_to_print))

    def seed_fix(self):
        seed = self.args.seed
        # Add rank offset for different data shuffling per GPU in DDP
        if getattr(self.args, 'distributed', False):
            seed = seed + get_rank()

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _init_lr(self):
        if self.args.lr is None:
            eff_batch_size = self.args.batch_size * getattr(self.args, "accum_iter", 1)
            self.args.lr = self.args.blr * eff_batch_size / 256
        if not hasattr(self.args, "lr_min") or self.args.lr_min is None:
            self.args.lr_min = self.args.lr * 0.01

    def _get_device(self):
        """Get the appropriate device for training."""
        if self.args.device.lower() == 'cuda' and torch.cuda.is_available():
            # In DDP mode, use gpu_id (physical GPU); otherwise use gpu_num
            # gpu_id accounts for multi-process per GPU setups
            if getattr(self.args, 'distributed', False):
                gpu_id = getattr(self.args, 'gpu_id', getattr(self.args, 'local_rank', 0))
            else:
                gpu_id = getattr(self.args, 'gpu_num', 0)
            device = torch.device(f"cuda:{gpu_id}")
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
        """Save checkpoint only on main process."""
        # Only save on rank 0 in DDP mode
        if not is_main_process():
            return

        ckpt_name = f"{self.args.model.lower()}_e{epoch}_s{self.args.seq_len}_p{self.args.pred_len}.pth"
        save_path = os.path.join(self.ckpt_path, ckpt_name)

        # Handle DDP model wrapper - get underlying model state_dict
        model_to_save = model.module if hasattr(model, 'module') else model

        checkpoint = {
            "epoch": epoch,
            "model": model_to_save.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(self.args)
        }

        if scaler is not None:
            checkpoint["scaler"] = scaler.state_dict()

        torch.save(checkpoint, save_path)
        if verbose:
            print(f"Checkpoint saved at {save_path}")

    def _save_training_args(self):
        """Save training arguments to a JSON file in the checkpoint directory."""
        if not is_main_process():
            return

        args_dict = {}
        for key, value in vars(self.args).items():
            # Convert non-serializable types to strings
            if isinstance(value, torch.dtype):
                args_dict[key] = str(value)
            elif hasattr(value, '__dict__'):
                args_dict[key] = str(value)
            else:
                try:
                    json.dumps(value)
                    args_dict[key] = value
                except (TypeError, ValueError):
                    args_dict[key] = str(value)

        args_path = os.path.join(self.ckpt_path, "training_args.json")
        with open(args_path, 'w') as f:
            json.dump(args_dict, f, indent=2)
        print_rank0(f"Training args saved to {args_path}")

    def _load_checkpoint(self,
                         ckpt_name: str,
                         model: nn.Module,
                         optimizer: Union[optim.Optimizer, None]=None,
                         scaler=None,
                         verbose=True):
        """Load checkpoint, handling DDP model wrapper."""
        ckpt_list = list(map(lambda x: os.path.splitext(x)[0], os.listdir(self.ckpt_path)))
        assert ckpt_name in ckpt_list, f"{ckpt_name} is not exist in {self.ckpt_path}"

        load_path = os.path.join(self.ckpt_path, ckpt_name + ".pth")

        # Load to CPU first, then move to device
        try:
            checkpoint = torch.load(load_path, map_location="cpu", weights_only=True)
        except TypeError:
            # Fallback for older PyTorch versions without weights_only parameter
            checkpoint = torch.load(load_path, map_location="cpu")

        # Handle DDP model wrapper
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(checkpoint["model"])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        self.args.ckpt_args = checkpoint.get("args", None)

        if verbose and is_main_process():
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