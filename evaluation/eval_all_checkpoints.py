"""
Batch evaluation script for all model checkpoints.
Evaluates all checkpoints and saves metrics to CSV files.
"""

import os
import sys
import glob
import csv
import json
import torch
import argparse

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from long_term_forecasting import Long_Term_Forecasting
from engine_forecasting import evaluate
from utils.distributed import is_main_process


# Model configurations matching training scripts
MODEL_CONFIGS = {
    'itransformer': {
        'd_model': 32, 'd_ff': 32, 'e_layers': 1, 'n_heads': 1, 'top_k': 2,
        'find_unused_parameters': False
    },
    'timexer': {
        'd_model': 32, 'd_ff': 32, 'e_layers': 1, 'n_heads': 1,
        'find_unused_parameters': False
    },
    'wavenet': {
        'd_model': 32, 'd_ff': 32, 'e_layers': 3, 'n_heads': 1,
        'find_unused_parameters': True
    },
    'timesnet': {
        'd_model': 64, 'd_ff': 64, 'e_layers': 2, 'n_heads': 1, 'top_k': 3,
        'find_unused_parameters': True
    },
    'waxer': {
        'd_model': 32, 'd_ff': 32, 'e_layers': 1, 'n_heads': 1,
        'wavenet_d_model': 64, 'wavenet_layers': 3,
        'find_unused_parameters': True
    },
    'taxer': {
        'd_model': 32, 'd_ff': 32, 'e_layers': 1, 'n_heads': 1,
        'times_d_model': 64, 'times_d_ff': 64, 'times_top_k': 3,
        'times_num_kernels': 4, 'times_layers': 2,
        'find_unused_parameters': True
    }
}

# Dataset configurations
DATASET_CONFIGS = {
    'sticker': {'input_dim': 15, 'dataset': 'sticker', 'country': 'canada', 'store': 'all'},
    'korean': {'input_dim': 30, 'dataset': 'weather', 'city': 'korea'},
    'global': {'input_dim': 18, 'dataset': 'weather', 'city': 'global'}
}


def load_args_from_checkpoint(ckpt_path, ckpt_name):
    """Load saved args from checkpoint file."""
    ckpt_file = os.path.join(ckpt_path, ckpt_name + ".pth")
    if not os.path.exists(ckpt_file):
        return None

    try:
        checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        return checkpoint.get("args", None)
    except Exception as e:
        print(f"Warning: Could not load args from checkpoint: {e}")
        return None


def create_args(model_name, ckpt_path, ckpt_name, dataset_name='sticker', saved_args=None):
    """Create argument namespace for model evaluation.

    If saved_args is provided, use those values for model config.
    Otherwise fall back to hardcoded MODEL_CONFIGS.
    """
    config = MODEL_CONFIGS.get(model_name, {})
    dataset_config = DATASET_CONFIGS[dataset_name]

    class Args:
        pass

    args = Args()

    # Basic settings
    args.seed = 0
    args.model = model_name
    args.mode = 'test'
    args.device = 'cuda'
    args.gpu_num = 0
    args.precision = torch.float32
    args.val = False

    # Data settings from dataset config
    args.dataset = dataset_config.get('dataset', 'sticker')
    args.country = dataset_config.get('country', 'canada')
    args.city = dataset_config.get('city', 'korea')
    args.store = dataset_config.get('store', 'all')
    args.root_path = './'
    args.data_path = 'dataset'
    args.result_path = 'results'
    args.train_ratio = 0.8
    args.sample_rate = 8
    args.num_workers = 0
    args.batch_size = 32

    # Sequence settings - prefer saved_args if available
    args.seq_len = saved_args.get('seq_len', 512) if saved_args else 512
    args.label_len = saved_args.get('label_len', 16) if saved_args else 16
    args.pred_len = saved_args.get('pred_len', 16) if saved_args else 16
    args.forecast_len = 96
    args.patch_len = saved_args.get('patch_len', 16) if saved_args else 16
    args.input_dim = saved_args.get('input_dim', dataset_config.get('input_dim', 15)) if saved_args else dataset_config.get('input_dim', 15)

    # Model settings - prefer saved_args, then config, then defaults
    if saved_args:
        args.d_model = saved_args.get('d_model', config.get('d_model', 128))
        args.d_ff = saved_args.get('d_ff', config.get('d_ff', 128))
        args.n_heads = saved_args.get('n_heads', config.get('n_heads', 1))
        args.e_layers = saved_args.get('e_layers', config.get('e_layers', 3))
        args.dropout = saved_args.get('dropout', 0.2)
        args.top_k = saved_args.get('top_k', config.get('top_k', 2))
        args.wave_kernel_size = saved_args.get('wave_kernel_size', 3)
        args.time_inception = saved_args.get('time_inception', 5)
        # WaXer/TaXer specific
        args.wavenet_d_model = saved_args.get('wavenet_d_model', config.get('wavenet_d_model', 64))
        args.wavenet_layers = saved_args.get('wavenet_layers', config.get('wavenet_layers', 3))
        args.times_d_model = saved_args.get('times_d_model', config.get('times_d_model', 64))
        args.times_d_ff = saved_args.get('times_d_ff', config.get('times_d_ff', 64))
        args.times_top_k = saved_args.get('times_top_k', 3)
        args.times_num_kernels = saved_args.get('times_num_kernels', 4)
        args.times_layers = saved_args.get('times_layers', 2)
    else:
        args.d_model = config.get('d_model', 128)
        args.d_ff = config.get('d_ff', 128)
        args.n_heads = config.get('n_heads', 1)
        args.e_layers = config.get('e_layers', 3)
        args.dropout = 0.2
        args.top_k = config.get('top_k', 2)
        args.wave_kernel_size = 3
        args.time_inception = 5
        # WaXer/TaXer specific
        args.wavenet_d_model = config.get('wavenet_d_model', 64)
        args.wavenet_layers = config.get('wavenet_layers', 3)
        args.times_d_model = config.get('times_d_model', 64)
        args.times_d_ff = config.get('times_d_ff', 64)
        args.times_top_k = 3
        args.times_num_kernels = 4
        args.times_layers = 2

    args.activation = 'relu'
    args.use_norm = True

    # Optimizer settings (required even for eval)
    args.optimizer = 'adamw'
    args.lr = 0.0003
    args.blr = 0.00015
    args.min_lr = None
    args.lr_min = 3e-6
    args.betas = [0.9, 0.95]
    args.lr_scheduler = 'constant'
    args.warmup_epochs = 5
    args.accum_iter = 1

    # Training settings
    args.epochs = 128
    args.start_epoch = 0
    args.print_iter = 10
    args.save_interval = 8

    # Checkpoint settings
    args.ckpt_path = ckpt_path
    args.ckpt_name = ckpt_name
    args.ckpt_args = None

    # DDP settings (single GPU)
    args.distributed = False
    args.rank = 0
    args.local_rank = 0
    args.world_size = 1
    args.gpu_id = 0
    args.find_unused_parameters = config.get('find_unused_parameters', False)
    args.dist_backend = 'nccl'
    args.processes_per_gpu = 1

    args.verbose = False
    args.bootstrapping_step = 1
    args.scale_factor = 1

    return args


def evaluate_checkpoint(model_name, ckpt_path, ckpt_name, dataset_name='sticker'):
    """Evaluate a single checkpoint and return metrics."""
    # Load saved args from checkpoint to ensure model config matches
    saved_args = load_args_from_checkpoint(ckpt_path, ckpt_name)
    args = create_args(model_name, ckpt_path, ckpt_name, dataset_name, saved_args)

    try:
        task = Long_Term_Forecasting(args)
        task._load_checkpoint(ckpt_name=ckpt_name, model=task.model, verbose=False)
        _, test_loader, _ = task._get_data_loader(flag='test')
        criterion = task._select_criterion()

        test_loss, test_metric = evaluate(
            model=task.model,
            criterion=criterion,
            data_loader=test_loader,
            device=task.device,
            args=args
        )
        mae, mse, rmse, mape, mspe, corr = test_metric

        return {
            'checkpoint': ckpt_name,
            'epoch': int(ckpt_name.split('_e')[1].split('_')[0]),
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'mspe': float(mspe),
            'corr': float(corr),
            'loss': float(test_loss)
        }
    except Exception as e:
        print(f"Error evaluating {ckpt_name}: {e}")
        return None


def evaluate_model(model_name, dataset='sticker', output_dir='results/eval_results'):
    """Evaluate all checkpoints for a model."""
    ckpt_dir = f'checkpoints/{dataset}/{model_name}_v1'

    if not os.path.exists(ckpt_dir):
        print(f"Checkpoint directory not found: {ckpt_dir}")
        return []

    # Find all checkpoint files
    ckpt_files = glob.glob(os.path.join(ckpt_dir, f'{model_name}_e*_s512_p16.pth'))
    ckpt_files.sort(key=lambda x: int(x.split('_e')[1].split('_')[0]))

    print(f"\n{'='*60}")
    print(f"Evaluating {model_name.upper()} ({len(ckpt_files)} checkpoints)")
    print(f"{'='*60}")

    results = []
    for ckpt_file in ckpt_files:
        ckpt_name = os.path.splitext(os.path.basename(ckpt_file))[0]
        print(f"  Evaluating {ckpt_name}...", end=' ')

        result = evaluate_checkpoint(model_name, ckpt_dir, ckpt_name, dataset)
        if result:
            results.append(result)
            print(f"MSE: {result['mse']:.6f}, MAE: {result['mae']:.6f}")
        else:
            print("FAILED")

    # Save to CSV
    if results:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f'{model_name}_eval.csv')

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        print(f"\nResults saved to {csv_path}")

        # Find best checkpoint
        best = min(results, key=lambda x: x['mse'])
        print(f"Best checkpoint: {best['checkpoint']} (MSE: {best['mse']:.6f})")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate all model checkpoints')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Models to evaluate (default: all)')
    parser.add_argument('--dataset', type=str, default='sticker',
                        choices=['sticker', 'korean', 'global'],
                        help='Dataset to evaluate (default: sticker)')
    parser.add_argument('--output_dir', default='results/eval_results',
                        help='Output directory for results')
    args = parser.parse_args()

    models = args.models or list(MODEL_CONFIGS.keys())

    all_results = {}
    for model_name in models:
        if model_name not in MODEL_CONFIGS:
            print(f"Unknown model: {model_name}")
            continue

        results = evaluate_model(model_name, args.dataset, args.output_dir)
        if results:
            all_results[model_name] = results

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    summary = []
    for model_name, results in all_results.items():
        best = min(results, key=lambda x: x['mse'])
        summary.append({
            'model': model_name,
            'best_checkpoint': best['checkpoint'],
            'best_mse': best['mse'],
            'best_mae': best['mae'],
            'best_rmse': best['rmse'],
            'best_mape': best['mape'],
            'best_corr': best['corr']
        })

    # Sort by MSE
    summary.sort(key=lambda x: x['best_mse'])

    print(f"\n{'Model':<15} {'Best Checkpoint':<30} {'MSE':<12} {'MAE':<12} {'CORR':<12}")
    print("-"*80)
    for s in summary:
        print(f"{s['model']:<15} {s['best_checkpoint']:<30} {s['best_mse']:<12.6f} {s['best_mae']:<12.6f} {s['best_corr']:<12.6f}")

    # Save summary JSON
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    main()
