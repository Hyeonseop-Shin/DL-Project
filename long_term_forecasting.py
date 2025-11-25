
import torch
import torch.nn as nn
import torch.optim as optim

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from dataset.data_provider import data_provider
from engine_forecasting import train_one_epoch, evaluate, forecasting
from models import iTransformer, TimeXer, WaveFormer

from models.WaveFormer.WaveNet import WaveNetForecaster
from models.WaveFormer.TimesNet import TimesNet

from utils.lr_scheduler import adjust_learning_rate
from utils.task_base import Task
from utils.tools import save_test_result

class Long_Term_Forecasting(Task):
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

        elif model_name == 'wavenet':
            model = WaveNetForecaster(
                seq_len=self.args.seq_len,
                pred_len=self.args.pred_len,
                c_in=self.args.input_dim,
                d_model=self.args.d_model,
                dropout=self.args.dropout,
                layers=self.args.e_layers,
                kernel_size=self.args.wave_kernel_size
            )

        elif model_name == 'timesnet':
            model = TimesNet(
                seq_len=self.args.seq_len,
                pred_len=self.args.pred_len,
                c_in=self.args.input_dim,
                c_out=self.args.input_dim,
                d_model=self.args.d_model,
                d_ff=self.args.d_ff,
                top_k=self.args.top_k,
                num_kernels=self.args.time_inception,
                e_layers=self.args.e_layers,
                dropout=self.args.dropout
            )

        else:
            raise ValueError(f"Unknown model type {model_name}")
        
        return model

    def _get_data_loader(self, flag='train'):
        self.args.country = self.args.country.lower()
        self.args.store = self.args.store.lower()
        dataset, dataloader = data_provider(data_path=self.data_path,
                                            country=self.args.country,
                                            store=self.args.store,
                                            seq_len=self.args.seq_len,
                                            label_len=self.args.label_len,
                                            pred_len=self.args.pred_len,
                                            forecast_len=self.args.forecast_len,
                                            num_workers=self.args.num_workers,
                                            batch_size=self.args.batch_size,
                                            flag=flag)
        return dataset, dataloader


    def train(self, val=True):
        print("Start training...")
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
                val_loss, val_metric = evaluate(
                    model=self.model,
                    data_loader=val_loader,
                    criterion=criterion,
                    device=self.device,
                    args=self.args
                )
                mae, mse, rmse, mape, mspe, corr = val_metric
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
        print("Start testing...")
        self._load_checkpoint(ckpt_name=self.args.ckpt_name, model=self.model)
        _, test_loader = self._get_data_loader(flag='test')
        criterion = self._select_criterion()

        test_loss, test_metric = evaluate(
            model=self.model,
            criterion=criterion,
            data_loader=test_loader,
            device=self.device,
            args=self.args)
        mae, mse, rmse, mape, mspe, corr = test_metric

        test_metric_log = (
            f"Test MAE: {mae:.5f}\t"
            f"Test MSE: {mse:.5f}\t"
            f"Test RMSE: {rmse:.5f}\t"
            f"Test MAPE: {mape:.5f}\t"
            f"Test MSPE: {mspe:.5f}\t"
            f"Test CORR: {corr:.5f}"
        )
        print(test_metric_log)
        save_test_result(
            metrics=test_metric,
            training_args=self.args.ckpt_args,
            ckpt_name=self.args.ckpt_name,
            result_path=self.args.result_path
        )
        
    
    def forecast(self):
        print("Start forecasting...")
        self._load_checkpoint(ckpt_name=self.args.ckpt_name, model=self.model)
        forecast_dataset, forecast_loader = self._get_data_loader(flag='forecast')
        criterion = self._select_criterion()

        forecast_data = forecasting(
            model=self.model,
            data_loader=forecast_loader,
            device=self.device,
            forecast_len=self.args.forecast_len,
            bootstrapping_step=self.args.bootstrapping_step,
            args=self.args
        ).detach().cpu().numpy()
        forecast_data = torch.tensor(forecast_dataset.inverse_transform(forecast_data))
        label_data = forecast_dataset.get_tail_data(tail_len=self.args.forecast_len,
                                                    var_type=torch.Tensor)
        forecast_loss = criterion(forecast_data, label_data)

        past_data = forecast_dataset.get_whole_data_without_tail(tail_len=self.args.forecast_len,
                                                                 var_type=torch.Tensor)
        
        self.paint_result(forecast_data, past_data, label_data)

    
    def paint_result(self, forecast_data, past_data, label_data):
        
        def to_np(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)
        
        painting_past_window = 3*self.args.forecast_len
        past_data = past_data[-painting_past_window:, :]

        forecast_np = to_np(forecast_data)  # [T_pred, D]
        past_np = to_np(past_data)          # [T_past, D]
        label_np = to_np(label_data)        # [T_pred, D]

        full_gt = np.concatenate([past_np, label_np], axis=0)

        T_past  = past_np.shape[0]
        T_total, D = full_gt.shape

        timesteps_full = np.arange(1, T_total + 1)
        pred_start = T_past + 1
        timesteps_pred = np.arange(pred_start, T_total + 1)

        fig, axes = plt.subplots(D, 1, figsize=(10, 2.5 * D), sharex=True)
        if D == 1:
            axes = [axes]

        for i in range(D):
            ax = axes[i]
            ax.plot(timesteps_full, full_gt[:, i],
                    label="ground truth", alpha=0.7)
            ax.plot(timesteps_pred, forecast_np[:, i],
                    linestyle="--", label="forecast", alpha=0.9)
            ax.axvline(pred_start, color='red', linestyle=':', linewidth=1.5,
                    label="prediction start" if i == 0 else None)

            ax.set_ylabel(f"dim {i}")
            ax.grid(True)
            ax.legend()

        axes[-1].set_xlabel("Timestep")
        result_name = f"forecasting_{self.args.ckpt_name}_f{self.args.forecast_len}_b{self.args.bootstrapping_step}"
        fig.suptitle(f"{result_name}")
        plt.tight_layout()

        plt.savefig(os.path.join(self.args.result_path, result_name + ".png"))
        

        

