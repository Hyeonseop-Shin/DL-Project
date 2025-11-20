
import math
import sys
from tqdm import tqdm

import torch
import torch.nn as nn

from utils.metrics import metric


def to_device(x, device, dtype, non_blocking=True):
    return x.to(device, dtype=dtype, non_blocking=non_blocking)


def prepare_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, device, precision):
    batch_x = to_device(batch_x, device=device, dtype=precision)
    batch_y = to_device(batch_y, device=device, dtype=precision)
    batch_x_mark = to_device(batch_x_mark, device=device, dtype=precision)
    batch_y_mark = to_device(batch_y_mark, device=device, dtype=precision)

    return (batch_x, batch_y, batch_x_mark, batch_y_mark)


def train_one_epoch(
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim,
        criterion: nn.Module,
        device: torch.device,
        epoch: int,
        args=None
    ) -> float:
    
    model.train()
    total_loss = .0
    optimizer.zero_grad()

    print_iter = args.print_iter
    accum_iter = args.accum_iter
    precision = args.precision
    device_type = args.device

    for data_iter, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
        batch_x, batch_y, batch_x_mark, batch_y_mark = prepare_batch(
            batch_x, batch_y, batch_x_mark, batch_y_mark, 
            device=device, precision=precision
        )

        with torch.amp.autocast(dtype=precision, device_type=device_type):
            output = model(batch_x, batch_x_mark)
            loss = criterion(output, batch_y)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss = {loss_value}, stopping training")
            sys.exit(1)

        loss = loss / accum_iter
        loss.backward()
        if (data_iter + 1) % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss_value
    
    avg_train_loss = total_loss / len(data_loader)
    return avg_train_loss


@torch.no_grad()
def evaluate(
        model: nn.Module,
        criterion: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        args
    ) -> float:

    model.eval()
    val_loss = 0.0
    precision = args.precision
    device_type = args.device

    output_list = list()
    label_list = list()

    for (batch_x, batch_y, batch_x_mark, batch_y_mark) in data_loader:
        batch_x, batch_y, batch_x_mark, batch_y_mark = prepare_batch(
            batch_x, batch_y, batch_x_mark, batch_y_mark, 
            device=device, precision=precision
        )

        with torch.amp.autocast(dtype=precision, device_type=device_type):
            output = model(batch_x, batch_x_mark)
            loss = criterion(output, batch_y)

            output_list.append(output)
            label_list.append(batch_y)

        
        loss_value = loss.item()
        val_loss += loss_value

    avg_val_loss = val_loss / len(data_loader)

    output_list = torch.cat(output_list, dim=0)
    label_list = torch.cat(label_list, dim=0)
    metric_result = metric(pred=output_list, true=label_list)

    return avg_val_loss, metric_result


@torch.no_grad()
def forecasting(
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        forecast_len: int=1,
        bootstrapping_step: int=10,
        args=None
    ):

    model.eval()
    precision = args.precision
    device_type = args.device

    all_pred = list()
    
    for batch in data_loader:
        pass
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    batch_x, batch_y, batch_x_mark, batch_y_mark = prepare_batch(
        batch_x, batch_y, batch_x_mark, batch_y_mark, 
        device=device, precision=precision
    )

    for _ in tqdm(range(forecast_len // bootstrapping_step + 1)):
        with torch.amp.autocast(dtype=precision, device_type=device_type):
            output = model(batch_x, batch_x_mark)
        
        bootstrapping = output[:, :bootstrapping_step, :]
        all_pred.append(bootstrapping)
        batch_x = torch.cat([batch_x[:, bootstrapping_step:, :], bootstrapping], dim=1)


    all_pred = torch.cat(all_pred, dim=1)
    all_pred = all_pred[:, :forecast_len, :]

    return all_pred.squeeze(dim=0)

