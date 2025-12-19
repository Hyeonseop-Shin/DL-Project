
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from dataset import weather_provider, sticker_provider
from dataset.data_provider import Dataset_Sticker
from dataset.data_provider_weather import Dataset_Weather
from engine_forecasting import train_one_epoch, evaluate, forecasting
from models import iTransformer, TimeXer, WaveFormer, WaveNet, TimesNet, TimesFormer, WaTiFormer_Unified, TimeXerWithWaveNet, TaXer, TimeXerWithHybridFeatures

from utils.lr_scheduler import adjust_learning_rate
from utils.task_base import Task
from utils.tools import save_test_result
from utils.distributed import (
    is_main_process,
    is_dist_available_and_initialized,
    print_rank0,
    barrier
)
from utils.visualization import plot_forecast

class Long_Term_Forecasting(Task):
    def __init__(self, args):
        super().__init__(args)

        # Build and move model to device
        self.model = self._build_model().to(self.device)

        # Wrap model with DDP if distributed
        if getattr(args, 'distributed', False):
            # Use gpu_id (physical GPU) instead of local_rank for multi-process per GPU support
            gpu_id = getattr(args, 'gpu_id', args.local_rank)
            self.model = DDP(
                self.model,
                device_ids=[gpu_id],
                output_device=gpu_id,
                find_unused_parameters=getattr(args, 'find_unused_parameters', False)
            )
    
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
        elif model_name == 'timexer':
            model = TimeXer(
                seq_len=self.args.seq_len,
                pred_len=self.args.pred_len,
                d_model=self.args.d_model,
                d_ff=self.args.d_ff,
                dropout=self.args.dropout,
                n_heads=self.args.n_heads,
                activation=self.args.activation,
                e_layers=self.args.e_layers,
                patch_len=self.args.patch_len,
                use_norm=self.args.use_norm
            )
        elif model_name == 'wavenet':
            model = WaveNet(
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
        elif model_name == 'waveformer':
            model = WaveFormer(
                seq_len=self.args.seq_len,
                pred_len=self.args.pred_len,
                c_in=self.args.input_dim,
                d_model=self.args.d_model,
                n_heads=self.args.n_heads,
                d_ff=self.args.d_ff,
                dropout=self.args.dropout,
                wave_layers=4,   # Number of WaveNet Blocks
                trans_layers=2,  # Number of Transformer Layers
                kernel_size=self.args.wave_kernel_size
            )
        elif model_name == 'timesformer':
            model = TimesFormer(
                seq_len=self.args.seq_len,
                pred_len=self.args.pred_len,
                c_in=self.args.input_dim,
                d_model=self.args.d_model,
                d_ff=self.args.d_ff,
                top_k=self.args.top_k,
                num_kernels=self.args.time_inception,
                times_layers=2,  # Number of TimesBlocks
                trans_layers=2,  # Number of Transformer Layers
                n_heads=self.args.n_heads,
                dropout=self.args.dropout,
            )
        elif model_name == 'watiformer':
            model = WaTiFormer_Unified(
                seq_len=self.args.seq_len,
                pred_len=self.args.pred_len, 
                c_in=self.args.input_dim,
                d_model=self.args.d_model,
                d_ff=self.args.d_ff,
                n_layers=self.args.e_layers,
                top_k=self.args.top_k,
                num_kernels=self.args.time_inception, # TimesBlock Inception kernel num
                n_heads=self.args.n_heads,
                dropout=self.args.dropout,
            )

        elif model_name == 'waxer':
            model = TimeXerWithWaveNet(
                # --- TimeXer Standard Args ---
                seq_len=self.args.seq_len,
                pred_len=self.args.pred_len,
                d_model=self.args.d_model,
                d_ff=self.args.d_ff,
                dropout=self.args.dropout,
                n_heads=self.args.n_heads,
                e_layers=self.args.e_layers,
                patch_len=self.args.patch_len,
                use_norm=self.args.use_norm,
                
                # --- WaveNet Specific Args ---
                # enc_in: 데이터셋의 변수 개수 (Input Channel)
                wavenet_c_in=self.args.input_dim,  
                # wavenet_d_model: WaveNet 내부의 Residual Channel 크기
                wavenet_d_model=self.args.wavenet_d_model, 
                # wavenet_layers: WaveNet의 층 깊이 (Receptive Field 결정)
                wavenet_layers=self.args.wavenet_layers 
            )

        elif model_name == 'taxer':
            model = TaXer(
                # TimeXer Args
                seq_len=self.args.seq_len,
                pred_len=self.args.pred_len,
                d_model=self.args.d_model,
                e_layers=self.args.e_layers,
                # ... (기타 TimeXer args)

                # TimesNet Args (Feature Extractor용)
                times_c_in=self.args.input_dim,            # 데이터셋의 변수 개수 (기존 args.enc_in 사용)
                times_d_model=self.args.times_d_model,  # TimesNet 내부 차원
                times_d_ff=self.args.times_d_ff,        # Inception 블록 내부 차원
                times_top_k=self.args.times_top_k,      # 주요 주기(Period) 개수
                times_num_kernels=self.args.times_num_kernels, # 커널 개수
                times_layers=self.args.times_layers     # Feature Extractor 층 수
            )

        elif model_name == 'timexer_hybrid':
            model = TimeXerWithHybridFeatures(
                # TimeXer
                seq_len=self.args.seq_len,
                pred_len=self.args.pred_len,
                d_model=self.args.d_model,
                d_ff=self.args.d_ff,
                dropout=self.args.dropout,
                n_heads=self.args.n_heads,
                e_layers=self.args.e_layers,
                patch_len=self.args.patch_len,
                use_norm=self.args.use_norm,
                
                # WaveNet
                wavenet_c_in=self.args.input_dim,
                wavenet_d_model=self.args.wavenet_d_model,
                wavenet_layers=self.args.wavenet_layers,
                
                # TimesNet
                times_d_model=self.args.times_d_model,
                times_d_ff=self.args.times_d_ff,
                times_top_k=self.args.times_top_k,
                times_num_kernels=self.args.times_num_kernels,
                times_layers=self.args.times_layers
            )

        else:
            raise ValueError(f"Unknown model type {model_name}")
        
        return model

    def _get_data_loader(self, flag='train'):
        self.args.country = self.args.country.lower()
        self.args.store = self.args.store.lower()

        # Create dataset
        batch_size = self.args.batch_size if flag == 'train' else 1

        if self.args.dataset.lower() == 'sticker':
            dataset = Dataset_Sticker(
                data_path=self.data_path,
                country=self.args.country,
                store=self.args.store,
                seq_size=[self.args.seq_len, self.args.label_len,
                          self.args.pred_len, self.args.forecast_len],
                scale=True,
                train_ratio=self.args.train_ratio,
                flag=flag
            )
        elif self.args.dataset.lower() == 'weather':
            dataset = Dataset_Weather(
                data_path=self.data_path,
                city=self.args.city,
                seq_size=[self.args.seq_len, self.args.label_len,
                          self.args.pred_len, self.args.forecast_len],
                scale=True,
                train_ratio=self.args.train_ratio,
                sample_rate=self.args.sample_rate,
                flag=flag
            )
        else:
            raise ValueError(f"Unknown dataset: {self.args.dataset}")

        # Create sampler for distributed training
        sampler = None
        shuffle = (flag == 'train')

        if getattr(self.args, 'distributed', False) and flag == 'train':
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.args.world_size,
                rank=self.args.rank,
                shuffle=True
            )
            shuffle = False  # DistributedSampler handles shuffling

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.args.num_workers,
            drop_last=(flag == 'train'),  # Drop last for consistent batch sizes in DDP
            pin_memory=True  # Faster data transfer to GPU
        )

        return dataset, dataloader, sampler


    def train(self, val=True):
        print_rank0("Start training...")
        self.count_parameters()
        self._save_training_args()  # Save training args at start

        _, train_loader, train_sampler = self._get_data_loader(flag='train')
        val_dataset, val_loader, _ = None, None, None
        if val:
            val_dataset, val_loader, _ = self._get_data_loader(flag='val')

        optimizer = self._select_optimizer()
        criterion = self._select_criterion()

        start_epoch = self.args.start_epoch
        last_epoch = start_epoch + self.args.epochs
        save_interval = getattr(self.args, 'save_interval', 8)
        viz_interval = getattr(self.args, 'viz_interval', 32)

        train_loss_list = list()
        val_loss_list = list()

        for epoch in range(start_epoch, last_epoch):
            # Set epoch for DistributedSampler to ensure proper shuffling
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            start_time = time.perf_counter()

            adjust_learning_rate(
                optimizer=optimizer,
                epoch=epoch - start_epoch,
                total_epoch=self.args.epochs,
                lr_scheduler=self.args.lr_scheduler,
                lr_init=self.args.lr,
                lr_min=self.args.lr_min,
                verbose=self.args.verbose and is_main_process()
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

            if val and val_loader is not None:
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

            # Only log on main process
            if is_main_process():
                log = (
                    f"Epoch: [{epoch:3d}/{last_epoch:3d}]  "
                    f"Train Loss: {train_loss:.5f}\t"
                    f"Train Time: {train_time:.3f}s\t"
                )
                if val:
                    log += f"Val Loss: {val_loss:.5f}\tVal Time: {val_time:.3f}s"
                print(log)

            # Save checkpoint every save_interval epochs
            if (epoch + 1) % save_interval == 0 or epoch == last_epoch - 1:
                self._save_checkpoint(
                    model=self.model,
                    optimizer=optimizer,
                    epoch=epoch,
                    verbose=self.args.verbose
                )

            # Save visualization every viz_interval epochs (separate from checkpoint)
            if (epoch + 1) % viz_interval == 0 or epoch == last_epoch - 1:
                # Synchronize before visualization (only rank 0 does visualization)
                barrier()
                # Save loss curve and sample predictions (only on main process)
                if is_main_process():
                    self._save_loss_curve(train_loss_list, val_loss_list, epoch)
                    # Use val_loader if available, otherwise use train_loader for predictions
                    pred_loader = val_loader if val_loader is not None else train_loader
                    self._save_sample_prediction(pred_loader, criterion, epoch)
                # Synchronize after visualization before next epoch
                barrier()

            if self.args.verbose and is_main_process():
                print()

            # Synchronize all processes at end of epoch
            barrier()

    def _save_loss_curve(self, train_losses, val_losses, epoch):
        """Save training/validation loss curve to checkpoint directory."""
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)

        ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        if val_losses:
            ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{self.args.model.upper()} Training Progress (Epoch {epoch+1})', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        save_path = os.path.join(self.ckpt_path, f'loss_curve_e{epoch}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Loss curve saved to {save_path}")

    def _save_sample_prediction(self, data_loader, criterion, epoch, num_samples=3):
        """Save sample prediction visualizations to checkpoint directory."""
        if data_loader is None:
            return

        # Use underlying model for inference (avoid DDP issues on single rank)
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.eval()
        samples_collected = 0
        predictions = []
        ground_truths = []

        with torch.no_grad():
            for batch in data_loader:
                # Dataloader returns (seq_x, seq_y, seq_x_mark, seq_y_mark)
                batch_x = batch[0].float().to(self.device)
                batch_y = batch[1].float().to(self.device)

                # Get prediction
                outputs = model(batch_x)

                predictions.append(outputs.cpu().numpy())
                ground_truths.append(batch_y.cpu().numpy())
                samples_collected += batch_x.size(0)

                if samples_collected >= num_samples:
                    break

        model.train()

        if not predictions:
            return

        # Concatenate and select samples
        preds = np.concatenate(predictions, axis=0)[:num_samples]
        gts = np.concatenate(ground_truths, axis=0)[:num_samples]

        # Create visualization
        num_vars = min(preds.shape[-1], 4)  # Show at most 4 variables
        fig, axes = plt.subplots(num_samples, num_vars, figsize=(4*num_vars, 3*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        if num_vars == 1:
            axes = axes.reshape(-1, 1)

        for i in range(num_samples):
            for j in range(num_vars):
                ax = axes[i, j]
                time_steps = range(preds.shape[1])
                ax.plot(time_steps, gts[i, :, j], 'b-', label='Ground Truth', linewidth=2)
                ax.plot(time_steps, preds[i, :, j], 'r--', label='Prediction', linewidth=2)
                if i == 0:
                    ax.set_title(f'Variable {j+1}', fontsize=10)
                if j == 0:
                    ax.set_ylabel(f'Sample {i+1}', fontsize=10)
                if i == 0 and j == 0:
                    ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

        fig.suptitle(f'{self.args.model.upper()} Predictions (Epoch {epoch+1})', fontsize=14)
        plt.tight_layout()

        save_path = os.path.join(self.ckpt_path, f'prediction_e{epoch}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Sample predictions saved to {save_path}")


    def test(self):
        print_rank0("Start testing...")
        self._load_checkpoint(ckpt_name=self.args.ckpt_name, model=self.model)
        _, test_loader, _ = self._get_data_loader(flag='test')
        criterion = self._select_criterion()

        test_loss, test_metric = evaluate(
            model=self.model,
            criterion=criterion,
            data_loader=test_loader,
            device=self.device,
            args=self.args
        )
        mae, mse, rmse, mape, mspe, corr = test_metric

        if is_main_process():
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
        print_rank0("Start forecasting...")
        self._load_checkpoint(ckpt_name=self.args.ckpt_name, model=self.model)
        forecast_dataset, forecast_loader, _ = self._get_data_loader(flag='forecast')
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

        # Only paint result on main process
        if is_main_process():
            self.paint_result(forecast_data, past_data, label_data)

    
    def paint_result(self, forecast_data, past_data, label_data):
        """
        Create publication-quality forecast visualization.

        Args:
            forecast_data: Predicted values [T_pred, D]
            past_data: Historical data [T_past, D]
            label_data: Ground truth for prediction period [T_pred, D]
        """
        # Select past window for visualization (3x forecast length)
        painting_past_window = 3 * self.args.forecast_len
        past_data = past_data[-painting_past_window:, :]

        # Generate result filename
        result_name = f"forecasting_{self.args.ckpt_name}_f{self.args.forecast_len}_b{self.args.bootstrapping_step}"
        save_path = os.path.join(self.args.result_path, result_name + ".png")

        # Use the new visualization module for publication-quality plots
        fig = plot_forecast(
            forecast_data=forecast_data,
            ground_truth=label_data,
            past_data=past_data,
            save_path=save_path,
            title=f"{self.args.model.upper()} Forecast (horizon={self.args.forecast_len}, bootstrap={self.args.bootstrapping_step})",
            show_metrics=True
        )

        # Also save as PDF for high-quality printing
        pdf_path = os.path.join(self.args.result_path, result_name + ".pdf")
        fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"PDF saved to {pdf_path}")

        plt.close(fig)
        

        

