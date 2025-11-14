
import math
import sys

import torch
import torch.nn as nn

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

        batch_x = batch_x.to(device, non_blocking=True, dtype=precision)
        batch_y = batch_y.to(device, non_blocking=True, dtype=precision)
        batch_x_mark = batch_x_mark.to(device, non_blocking=True, dtype=precision)
        batch_y_mark = batch_y_mark.to(device, non_blocking=True, dtype=precision)

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
def validation(
        model: nn.Module,
        criterion: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        args
    ) -> float:

    model.eval()
    val_loss = .0
    precision = args.precision
    device_type = args.device

    for (batch_x, batch_y, batch_x_mark, batch_y_mark) in data_loader:
        batch_x = batch_x.to(device, non_blocking=True, dtype=precision)
        batch_y = batch_y.to(device, non_blocking=True, dtype=precision)
        batch_x_mark = batch_x_mark.to(device, non_blocking=True, dtype=precision)
        batch_y_mark = batch_y_mark.to(device, non_blocking=True, dtype=precision)

        with torch.amp.autocast(dtype=precision, device_type=device_type):
            output = model(batch_x, batch_x_mark)
            loss = criterion(output, batch_y)
        
        loss_value = loss.item()
        val_loss += loss_value

    avg_val_loss = val_loss / len(data_loader)
    return avg_val_loss