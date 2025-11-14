
import torch

import os
import argparse

from long_term_prediction import Long_Term_Forecast

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


def arg_parser():
    parser = argparse.ArgumentParser('Long Term Prediction', add_help=False)
    
    # Training hyperparameters
    parser.add_argument('--seed', type=int, default=0,
                        help='seed')
    parser.add_argument('--precision', type=str2dtype, default=torch.float32,
                        help='Model data precision')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='starting epoch')
    parser.add_argument('--epochs', type=int, default=5,
                        help='training epochs')
    parser.add_argument('--accum_iter', type=int, default=1,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--print_iter', type=int, default=10,
                        help='Log printing iterations')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--gpu_num', type=int, default=0,
                        help='GPU num to use')


    # Model hyperparameters
    parser.add_argument('--model', type=str, default='itransformer',
                        help='Name of model to train')
    parser.add_argument('--seq_len', type=int, default=32,
                        help="Input sequence length (lookback window)")
    parser.add_argument('--label_len', type=int, default=16,
                        help="Decoder lookback window, no use for this project")
    parser.add_argument('--pred_len', type=int, default=16,
                        help="Model prediction length")
    parser.add_argument('--d_model', type=int, default=512,
                        help="Dimension of attention layer")
    parser.add_argument('--d_ff', type=int, default=512,
                        help="Dimension of feed forward network")
    parser.add_argument('--scale_factor', type=int, default=1,
                        help="Scaling factor of Transformer")
    parser.add_argument('--n_heads', type=int, default=8,
                        help="Number of Heads in MultiHead Attention")
    parser.add_argument('--dropout', type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument('--activation', type=str, default='relu',
                        help="Activation function of model")
    parser.add_argument('--e_layers', type=int, default=2,
                        help="Number of encoder layers")
    

    # Optimizer hyperparameters
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['Adam', 'AdamW', 'SGD'],
                        help='optimizer')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=None, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.95], 
                        help='betas for Adam, AdamW')
    parser.add_argument('--lr_scheduler', type=str, default='constant',
                        help='Type of lr scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')
    

    # Saving hyperparameters
    parser.add_argument('--root_path', default="./",
                        help="root path")
    parser.add_argument('--data_path', default="dataset",
                        help="dataset path")
    parser.add_argument('--result_path', default="results",
                        help="reslut saving path")
    parser.add_argument('--ckpt_path', default="checkpoints",
                        help="checkpoint saving path")
    
    # Others
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--verbose', type=str2bool, default=True)
    
    return parser

if __name__ == "__main__":
    args = arg_parser()
    args = args.parse_args()

    task = Long_Term_Forecast(args)
    task.train()
    