
import torch
import torch.nn as nn
import torch.optim as optim

import os
import time
import numpy as np

from dataset.data_provider import data_provider
from engine_prediction import train_one_epoch, validation
from models import iTransformer, TimeXer, WaveFormer

from utils.lr_scheduler import adjust_learning_rate
from utils.task_base import Task

class Long_Term_Forecast(Task):
    def __init__(self, args):
        super().__init__(args)

        self.model = self._build_model().to(self.device)
    
    def _build_model(self):
        model_name = self.args.model.lower()

        if model_name == 'itransformer':
            model = iTransformer(
                seq_len=self.args.seq_len,
                pred_len=self.args.pred_len,
                d_model=self.args.d_model,
                d_ff=self.args.d_ff,
                dropout=self.args.dropout,
                scale_factor=self.args.scale_factor,
                n_heads=self.args.n_heads,
                activation=self.args.activation,
                e_layers=self.args.e_layers
            )
        else:
            raise ValueError(f"Unknown model type {model_name}")
        
        return model

    def _get_data_loader(self, flag='train'):
        country = 'Canada'
        store = 'Discount_Stickers'
        dataset, dataloader = data_provider(data_path=self.data_path,
                                            country=country,
                                            store=store,
                                            seq_len=self.args.seq_len,
                                            label_len=self.args.label_len,
                                            pred_len=self.args.pred_len,
                                            num_workers=self.args.num_workers,
                                            flag=flag)
        return dataset, dataloader


    def train(self, val=True):
        
        _, train_loader = self._get_data_loader(flag='train')
        if val:
            _, val_loader = self._get_data_loader(flag='val')
        
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()

        start_epoch = self.args.start_epoch
        last_epoch = start_epoch + self.args.epochs
        
        train_loss_list = list()
        val_loss_list = list()
        prev_loss = np.inf

        for epoch in range(start_epoch, last_epoch):
            start_time = time.perf_counter()
            
            adjust_learning_rate(
                optimizer=optimizer,
                epoch=epoch-start_epoch,
                total_epoch=self.args.epochs,
                lr_scheduler=self.args.lr_scheduler,
                lr_init=self.args.lr,
                lr_min=self.args.lr_min,
                verbose=self.args.verbose  
            )

            train_loss = train_one_epoch(
                model=self.model,
                data_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=self.device,
                epoch=epoch,
                args=self.args
            )
            train_loss_list.append(train_loss)
            train_time = time.perf_counter() - start_time
        
            if val:
                val_loss = validation(
                    model=self.model,
                    data_loader=val_loader,
                    criterion=criterion,
                    device=self.device,
                    args=self.args
                )
                val_time = time.perf_counter() - train_time - start_time
                val_loss_list.append(val_loss)

            log = (
                f"Epoch: [{epoch:3d}/{last_epoch:3d}]  "
                f"Train Loss: {train_loss:.5f}\t"
                f"Train Time: {train_time:.3f}s\t"
                f"Val Loss: {val_loss:.5f}\tVal Time: {val_time:.3f}s" if val else ""
                )
            print(log)

            curr_loss = val_loss if val else train_loss
            save_condition = curr_loss < prev_loss
            if save_condition:
                prev_loss = curr_loss
                self._save_checkpoint(model=self.model,
                                      optimizer=optimizer,
                                      epoch=epoch,
                                      verbose=self.args.verbose)
            
            if self.args.verbose:
                print()

    def test(self):
        pass
    
    def predict(self):
        pass

