"""
Batch training script for weather datasets.
Trains all 6 models on Korean and Global weather datasets.
"""

import os
import subprocess
import sys

# Configuration
MODELS = ['itransformer', 'timexer', 'waxer', 'taxer', 'timesnet', 'wavenet']

# Dataset configurations: (city, input_dim, description)
DATASETS = {
    'korean': ('korea', 30, '5 Korean cities merged'),
    'global': ('global', 18, '3 Global cities merged (berlin, seoul, tokyo)')
}

# Common training parameters
COMMON_PARAMS = {
    'seq_len': 512,
    'pred_len': 16,
    'sample_rate': 12,
    'epochs': 128,
    'batch_size': 32,
    'lr': 3e-4,
    'save_interval': 8,
    'viz_interval': 32,
    'train_ratio': 0.8,
    'val': True,
    'dataset': 'weather',
}

# Model-specific parameters
MODEL_CONFIGS = {
    'itransformer': {
        'd_model': 128,
        'd_ff': 128,
        'n_heads': 4,
        'e_layers': 3,
        'dropout': 0.2,
    },
    'timexer': {
        'd_model': 90,
        'd_ff': 90,
        'n_heads': 3,
        'e_layers': 2,
        'patch_len': 16,
        'dropout': 0.1,
    },
    'waxer': {
        'd_model': 90,
        'd_ff': 90,
        'n_heads': 3,
        'e_layers': 2,
        'wavenet_d_model': 64,
        'wavenet_layers': 3,
        'dropout': 0.1,
    },
    'taxer': {
        'd_model': 90,
        'd_ff': 90,
        'n_heads': 3,
        'e_layers': 2,
        'times_d_model': 64,
        'times_d_ff': 64,
        'times_top_k': 5,
        'times_num_kernels': 4,
        'times_layers': 2,
        'dropout': 0.1,
    },
    'timesnet': {
        'd_model': 128,
        'd_ff': 128,
        'e_layers': 2,
        'top_k': 5,
        'dropout': 0.2,
    },
    'wavenet': {
        'd_model': 128,
        'd_ff': 128,
        'e_layers': 2,
        'dropout': 0.2,
    },
}


def build_command(model, city, input_dim, gpu_num=0):
    """Build training command for a model and dataset."""
    cmd = ['python', 'main.py']

    # Add common parameters
    cmd.extend(['--model', model])
    cmd.extend(['--dataset', COMMON_PARAMS['dataset']])
    cmd.extend(['--city', city])
    cmd.extend(['--input_dim', str(input_dim)])
    cmd.extend(['--seq_len', str(COMMON_PARAMS['seq_len'])])
    cmd.extend(['--pred_len', str(COMMON_PARAMS['pred_len'])])
    cmd.extend(['--sample_rate', str(COMMON_PARAMS['sample_rate'])])
    cmd.extend(['--epochs', str(COMMON_PARAMS['epochs'])])
    cmd.extend(['--batch_size', str(COMMON_PARAMS['batch_size'])])
    cmd.extend(['--lr', str(COMMON_PARAMS['lr'])])
    cmd.extend(['--save_interval', str(COMMON_PARAMS['save_interval'])])
    cmd.extend(['--viz_interval', str(COMMON_PARAMS['viz_interval'])])
    cmd.extend(['--train_ratio', str(COMMON_PARAMS['train_ratio'])])
    cmd.extend(['--val', str(COMMON_PARAMS['val'])])
    cmd.extend(['--gpu_num', str(gpu_num)])
    cmd.extend(['--mode', 'train'])

    # Add model-specific parameters
    if model in MODEL_CONFIGS:
        for key, value in MODEL_CONFIGS[model].items():
            cmd.extend([f'--{key}', str(value)])

    return cmd


def train_model(model, dataset_name, city, input_dim, gpu_num=0):
    """Train a single model on a dataset."""
    print("=" * 60)
    print(f"Training {model.upper()} on {dataset_name} dataset")
    print(f"  City: {city}, Input dim: {input_dim}")
    print("=" * 60)

    cmd = build_command(model, city, input_dim, gpu_num)
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n{model.upper()} on {dataset_name} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: {model.upper()} on {dataset_name} failed with code {e.returncode}\n")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Batch weather training')
    parser.add_argument('--models', nargs='+', default=MODELS,
                        help='Models to train (default: all)')
    parser.add_argument('--datasets', nargs='+', default=['korean', 'global'],
                        choices=['korean', 'global'],
                        help='Datasets to train on (default: both)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use (default: 0)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("WEATHER DATASET TRAINING")
    print("=" * 60)
    print(f"Models: {args.models}")
    print(f"Datasets: {args.datasets}")
    print(f"GPU: {args.gpu}")
    print(f"Seq_len: {COMMON_PARAMS['seq_len']}, Pred_len: {COMMON_PARAMS['pred_len']}")
    print(f"Sample_rate: {COMMON_PARAMS['sample_rate']} (coverage: ~256 days)")
    print(f"Save checkpoint every {COMMON_PARAMS['save_interval']} epochs")
    print(f"Save visualization every {COMMON_PARAMS['viz_interval']} epochs")
    print("=" * 60 + "\n")

    results = []

    for dataset_name in args.datasets:
        city, input_dim, desc = DATASETS[dataset_name]
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset_name.upper()} - {desc}")
        print(f"{'='*60}\n")

        for model in args.models:
            if model not in MODELS:
                print(f"WARNING: Unknown model {model}, skipping...")
                continue

            if args.dry_run:
                cmd = build_command(model, city, input_dim, args.gpu)
                print(f"[DRY RUN] {' '.join(cmd)}\n")
                continue

            success = train_model(model, dataset_name, city, input_dim, args.gpu)
            results.append({
                'model': model,
                'dataset': dataset_name,
                'success': success
            })

    if not args.dry_run and results:
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        for r in results:
            status = "SUCCESS" if r['success'] else "FAILED"
            print(f"  {r['model']:<15} on {r['dataset']:<10}: {status}")
        print("=" * 60)

        # Print checkpoint locations
        print("\nCheckpoint locations:")
        print("  Korean: ./checkpoints/korean/{model}_v1/")
        print("  Global: ./checkpoints/global/{model}_v1/")


if __name__ == '__main__':
    main()
