
import torch

import os
import argparse

from long_term_forecasting import Long_Term_Forecasting
from utils.distributed import setup_distributed, cleanup_distributed

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2dtype(v):
    v = v.lower()
    if v == 'float16':
        return torch.float16
    elif v == 'float32':
        return torch.float32
    elif v == 'float64':
        return torch.float64
    else:
        raise ValueError(f"Unkown dtype {v}")

def strLower(v):
    return v.lower()

def arg_parser():
    parser = argparse.ArgumentParser('Long Term Prediction', add_help=False)
    
    # Training hyperparameters
    parser.add_argument('--seed', type=int, default=0,
                        help='seed')
    parser.add_argument('--precision', type=str2dtype, default=torch.float32,
                        help='Model data precision')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='starting epoch')
    parser.add_argument('--epochs', type=int, default=5,
                        help='training epochs')
    parser.add_argument('--accum_iter', type=int, default=1,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--print_iter', type=int, default=10,
                        help='Log printing iterations')
    parser.add_argument('--save_interval', type=int, default=8,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--viz_interval', type=int, default=32,
                        help='Save loss curve and prediction visualization every N epochs')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--gpu_num', type=int, default=0,
                        help='GPU num to use')
    parser.add_argument('--val', type=str2bool, default=True,
                        help='Whether to use validation during training')


    # Model hyperparameters
    parser.add_argument('--model', type=strLower, default='watiformer',
                        help='Name of model to train')
    parser.add_argument('--seq_len', type=int, default=512,
                        help="Input sequence length (lookback window)")
    parser.add_argument('--label_len', type=int, default=16,
                        help="Decoder lookback window, no use for this project")
    parser.add_argument('--pred_len', type=int, default=16,
                        help="Model prediction length")
    parser.add_argument('--forecast_len', type=int, default=96,
                        help="Forecasting length for forecast")
    parser.add_argument('--bootstrapping_step', type=int, default=1,
                        help="How many steps are used for bootstrapping")

    parser.add_argument('--d_model', type=int, default=32,
                        help="Dimension of attention layer")
    parser.add_argument('--d_ff', type=int, default=32,
                        help="Dimension of feed forward network")
    parser.add_argument('--scale_factor', type=int, default=1,
                        help="Scaling factor of Transformer")
    parser.add_argument('--n_heads', type=int, default=1,
                        help="Number of Heads in MultiHead Attention")
    parser.add_argument('--dropout', type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument('--activation', type=strLower, default='relu',
                        help="Activation function of model")
    parser.add_argument('--e_layers', type=int, default=1,
                        help="Number of encoder layers")
    
    # TimeXer hyperparameters
    parser.add_argument('--patch_len', type=int, default=16,
                        help="Patch length for TimeXer encoder")
    parser.add_argument('--use_norm', type=str2bool, default=True,
                        help="Apply normalization before TimeXer encoder")
    
    # WaveFormer hyperparameters
    parser.add_argument('--top_k', type=int, default=2,
                        help="Number of peaks to focus on during FFT")
    parser.add_argument('--wave_kernel_size', type=int, default=3,
                        help="Kernel size of Wave-Block CNN")
    parser.add_argument('--time_inception', type=int, default=5,
                        help="Number of inception CNN in Time-Block")
    parser.add_argument('--input_dim', type=int, default=30,
                        help="Number of input variables")
    
    # WaXer hyperparameters
    parser.add_argument('--wavenet_d_model', type=int, default=64, help='dimension of wavenet hidden states')
    parser.add_argument('--wavenet_layers', type=int, default=3, help='num of wavenet layers')

    # TimesNet hyperparameters
    parser.add_argument('--times_d_model', type=int, default=64, help='dimension of timesnet hidden states')
    parser.add_argument('--times_d_ff', type=int, default=64, help='dimension of timesnet fcn (inception block)')
    parser.add_argument('--times_top_k', type=int, default=3, help='number of top k periods in timesnet')
    parser.add_argument('--times_num_kernels', type=int, default=4, help='number of inception kernels in timesnet')
    parser.add_argument('--times_layers', type=int, default=2, help='num of timesnet layers used for extraction')

    # Optimizer hyperparameters
    parser.add_argument('--optimizer', type=strLower, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='optimizer')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=None, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.95], 
                        help='betas for Adam, AdamW')
    parser.add_argument('--lr_scheduler', type=strLower, default='constant',
                        help='Type of lr scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')
    

    # Saving hyperparameters
    parser.add_argument('--root_path', default="./",
                        help="root path")
    parser.add_argument('--data_path', default="dataset",
                        help="dataset path")
    parser.add_argument('--result_path', default="results",
                        help="result saving path")
    parser.add_argument('--ckpt_path', default="checkpoints",
                        help="checkpoint saving path")
    parser.add_argument('--ckpt_name', default="itransformer_e4_s512_p16",
                        help="checkpoint name")
    
    # Dataset parameters
    parser.add_argument('--dataset', type=strLower, default='weather',
                        help='dataset type')
    parser.add_argument('--country', type=strLower, default="canada",
                        help="dataset country name")
    parser.add_argument('--store', type=strLower, default="all",
                        help="dataset store name")
    parser.add_argument('--city', type=strLower, default="korea",
                        help="dataset city name")
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help="train dataset ratio")
    parser.add_argument('--sample_rate', type=int, default=8,
                        help="data sampling rate")

    # Others
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--verbose', type=str2bool, default=True)
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'forecast'])

    # DDP Arguments
    parser.add_argument('--dist_backend', type=str, default='nccl',
                        choices=['nccl', 'gloo'],
                        help='Distributed backend (nccl for GPU, gloo for CPU)')
    parser.add_argument('--find_unused_parameters', type=str2bool, default=False,
                        help='Find unused parameters in DDP (may slow down training)')
    parser.add_argument('--processes_per_gpu', type=int, default=1,
                        help='Number of processes per GPU (default: 1). '
                             'Use >1 for multi-process per GPU setup. '
                             'E.g., with 8 GPUs and processes_per_gpu=4, '
                             'use torchrun --nproc_per_node=32')

    return parser

if __name__ == "__main__":
    args = arg_parser()
    args = args.parse_args()

    # Initialize distributed training (auto-detects torchrun env vars)
    rank, local_rank, world_size, gpu_id = setup_distributed(
        args.dist_backend,
        processes_per_gpu=args.processes_per_gpu
    )

    # Store distributed info in args
    args.rank = rank
    args.local_rank = local_rank
    args.world_size = world_size
    args.gpu_id = gpu_id  # Physical GPU ID (may differ from local_rank)
    args.distributed = world_size > 1

    # Override gpu_num with gpu_id for device assignment
    if args.distributed:
        args.gpu_num = gpu_id

    try:
        task = Long_Term_Forecasting(args)
        if args.mode == 'train':
            task.train(val=args.val)
        elif args.mode == 'test':
            task.test()
        elif args.mode == 'forecast':
            task.forecast()
    finally:
        cleanup_distributed()
    