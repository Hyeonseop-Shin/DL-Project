
import torch

import os
import numpy as np


def save_test_result(metrics, training_args, ckpt_name, result_path):
    
    result_name = f"testing_{ckpt_name}.txt"
    save_path = os.path.join(result_path, result_name)

    mae, mse, rmse, mape, mspe, corr = metrics

    with open(save_path, "w") as f:
        f.write("===== Evaluation Metrics =====\n")
        f.write(f"MAE:  {mae}\n")
        f.write(f"MSE:  {mse}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"MAPE: {mape}\n")
        f.write(f"MSPE: {mspe}\n")
        f.write(f"CORR: {corr}\n")

        f.write("\n===== Training Arguments =====\n")
        for key, value in training_args.items():
            f.write(f"{key}: {value}\n")

class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.delta = delta

    def __call__(self, loss, model, path, name, val=True):
        score = -loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model, path, name, val=val)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(loss, model, path, name, val=val)
            self.counter = 0

    def save_checkpoint(self, loss, model, path, name, val=True):
        loss_type = 'Validation' if val else 'Train'
        if self.verbose:
            print(f'{loss_type} loss decreased ({self.loss_min:.5f} --> {loss:.5f}), Save Model')
        torch.save(model.state_dict(), path + f'{name}.pth')
        self.loss_min = loss