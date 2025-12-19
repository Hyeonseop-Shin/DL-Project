
import torch
import json
import csv

import os
import numpy as np


def save_test_result(metrics, training_args, ckpt_name, result_path):
    """
    Save test results in multiple formats (TXT, JSON, CSV).

    Args:
        metrics: Tuple of (mae, mse, rmse, mape, mspe, corr)
        training_args: Dict of training arguments
        ckpt_name: Checkpoint name for file naming
        result_path: Directory to save results
    """
    mae, mse, rmse, mape, mspe, corr = metrics

    # Create metrics dict
    metrics_dict = {
        'MAE': float(mae),
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAPE': float(mape),
        'MSPE': float(mspe),
        'CORR': float(corr)
    }

    # Save as TXT (human-readable)
    txt_path = os.path.join(result_path, f"testing_{ckpt_name}.txt")
    with open(txt_path, "w") as f:
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

    # Save as JSON (machine-readable)
    json_path = os.path.join(result_path, f"testing_{ckpt_name}.json")
    json_data = {
        'checkpoint': ckpt_name,
        'metrics': metrics_dict,
        'training_args': {k: str(v) for k, v in training_args.items()}
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    # Save as CSV (for easy import to spreadsheet)
    csv_path = os.path.join(result_path, f"testing_{ckpt_name}.csv")
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for metric_name, metric_value in metrics_dict.items():
            writer.writerow([metric_name, metric_value])

    print(f"Results saved to: {txt_path}, {json_path}, {csv_path}")

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