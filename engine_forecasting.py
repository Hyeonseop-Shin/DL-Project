
import math
import sys
from tqdm import tqdm
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.distributed as dist

from utils.eval.metrics import metric

# Handle different PyTorch versions for autocast
def get_autocast_context(dtype, device_type):
    """Get autocast context manager compatible with different PyTorch versions."""
    if dtype == torch.float32:
        # No autocast needed for float32
        return nullcontext()
    try:
        # New API (PyTorch >= 1.10)
        return torch.amp.autocast(dtype=dtype, device_type=device_type)
    except AttributeError:
        # Old API (PyTorch < 1.10)
        if device_type == 'cuda':
            return torch.cuda.amp.autocast(enabled=True)
        else:
            return nullcontext()
from utils.distributed import (
    is_dist_available_and_initialized,
    reduce_tensor,
    is_main_process,
    get_world_size
)


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
    total_loss = 0.0
    num_batches = 0
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

        with get_autocast_context(dtype=precision, device_type=device_type):
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
        num_batches += 1

    # Average loss across all batches
    avg_train_loss = total_loss / num_batches

    # Reduce loss across all processes in DDP
    if is_dist_available_and_initialized():
        loss_tensor = torch.tensor([avg_train_loss], device=device)
        avg_train_loss = reduce_tensor(loss_tensor).item()

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
    num_batches = 0
    precision = args.precision
    device_type = args.device

    output_list = list()
    label_list = list()

    for (batch_x, batch_y, batch_x_mark, batch_y_mark) in data_loader:
        batch_x, batch_y, batch_x_mark, batch_y_mark = prepare_batch(
            batch_x, batch_y, batch_x_mark, batch_y_mark,
            device=device, precision=precision
        )

        with get_autocast_context(dtype=precision, device_type=device_type):
            output = model(batch_x, batch_x_mark)
            loss = criterion(output, batch_y)

            output_list.append(output)
            label_list.append(batch_y)

        loss_value = loss.item()
        val_loss += loss_value
        num_batches += 1

    avg_val_loss = val_loss / num_batches

    # Reduce loss across all processes
    if is_dist_available_and_initialized():
        loss_tensor = torch.tensor([avg_val_loss], device=device)
        avg_val_loss = reduce_tensor(loss_tensor).item()

    # Concatenate outputs and labels
    output_list = torch.cat(output_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    # Gather predictions from all processes for metric calculation
    if is_dist_available_and_initialized():
        world_size = get_world_size()

        # Gather all outputs
        output_gathered = [torch.zeros_like(output_list) for _ in range(world_size)]
        label_gathered = [torch.zeros_like(label_list) for _ in range(world_size)]

        dist.all_gather(output_gathered, output_list)
        dist.all_gather(label_gathered, label_list)

        # Concatenate on all ranks for metric calculation
        output_list = torch.cat(output_gathered, dim=0)
        label_list = torch.cat(label_gathered, dim=0)

    # Calculate metrics
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

