"""
Generate prediction visualizations for weather dataset models.
Includes sample predictions and full sequence predictions for the entire evaluation set.

Usage:
    python plot_weather_predictions.py --dataset korean
    python plot_weather_predictions.py --dataset global
"""

import os
import json
import csv
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from long_term_forecasting import Long_Term_Forecasting
from eval_weather_checkpoints import WEATHER_MODEL_CONFIGS, DATASET_CONFIGS, create_args


def load_best_checkpoints(eval_dir):
    """Load best checkpoint info for each model."""
    models = ['itransformer', 'timexer', 'waxer', 'taxer', 'timesnet', 'wavenet']
    best_ckpts = {}

    for model in models:
        csv_path = os.path.join(eval_dir, f'{model}_eval.csv')
        if not os.path.exists(csv_path):
            continue

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        best = min(rows, key=lambda x: float(x['mse']))
        best_ckpts[model] = {
            'checkpoint': best['checkpoint'],
            'mse': float(best['mse']),
            'mae': float(best['mae']),
            'corr': float(best['corr'])
        }

    return best_ckpts


def get_all_predictions(model_name, ckpt_name, dataset_type):
    """Get ALL predictions from a model checkpoint (entire test set)."""
    args = create_args(model_name, None, ckpt_name, dataset_type)

    try:
        task = Long_Term_Forecasting(args)
        task._load_checkpoint(ckpt_name=ckpt_name, model=task.model, verbose=False)
        _, test_loader, _ = task._get_data_loader(flag='test')

        # Get underlying model for inference
        model = task.model.module if hasattr(task.model, 'module') else task.model
        model.eval()

        predictions = []
        ground_truths = []

        with torch.no_grad():
            for batch in test_loader:
                batch_x = batch[0].float().to(task.device)
                batch_y = batch[1].float().to(task.device)

                outputs = model(batch_x)

                predictions.append(outputs.cpu().numpy())
                ground_truths.append(batch_y.cpu().numpy())

        model.train()

        preds = np.concatenate(predictions, axis=0)
        gts = np.concatenate(ground_truths, axis=0)

        return preds, gts

    except Exception as e:
        print(f"Error getting predictions for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def plot_all_predictions(best_ckpts, dataset_type, num_samples=3, num_vars=3, output_path=None):
    """Create comprehensive prediction visualization for all models."""
    models = list(best_ckpts.keys())
    n_models = len(models)

    # Create figure with subplots: rows = models, cols = samples
    fig = plt.figure(figsize=(16, 4 * n_models))
    gs = GridSpec(n_models, num_samples, figure=fig, hspace=0.3, wspace=0.25)

    # Colors for different variables
    var_colors = plt.cm.tab10(np.linspace(0, 1, num_vars))

    for model_idx, model_name in enumerate(models):
        ckpt_info = best_ckpts[model_name]
        ckpt_name = ckpt_info['checkpoint']

        print(f"Getting predictions for {model_name.upper()}...")
        preds, gts = get_all_predictions(model_name, ckpt_name, dataset_type)

        if preds is None:
            continue

        for sample_idx in range(min(num_samples, preds.shape[0])):
            ax = fig.add_subplot(gs[model_idx, sample_idx])

            pred = preds[sample_idx]  # [pred_len, n_vars]
            gt = gts[sample_idx]      # [pred_len, n_vars]
            pred_len = pred.shape[0]
            x = np.arange(pred_len)

            # Plot first few variables
            for var_idx in range(min(num_vars, pred.shape[1])):
                ax.plot(x, gt[:, var_idx], '-', color=var_colors[var_idx],
                       alpha=0.7, linewidth=2, label=f'Actual V{var_idx+1}')
                ax.plot(x, pred[:, var_idx], '--', color=var_colors[var_idx],
                       alpha=0.9, linewidth=2, label=f'Pred V{var_idx+1}')

            # Styling
            if sample_idx == 0:
                ax.set_ylabel(f'{model_name.upper()}\n(MSE: {ckpt_info["mse"]:.4f})',
                             fontsize=10, fontweight='bold')
            if model_idx == 0:
                ax.set_title(f'Sample {sample_idx + 1}', fontsize=11, fontweight='bold')
            if model_idx == n_models - 1:
                ax.set_xlabel('Time Step', fontsize=10)

            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

            # Add legend only for first subplot
            if model_idx == 0 and sample_idx == 0:
                ax.legend(loc='upper right', fontsize=7, ncol=2)

    plt.suptitle(f'Model Predictions vs Ground Truth - {dataset_type.upper()} Weather Dataset\n(Solid: Actual, Dashed: Predicted)',
                fontsize=14, fontweight='bold', y=1.02)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Prediction visualization saved to {output_path}")


def plot_single_model_detail(model_name, ckpt_info, dataset_type, num_samples=6, output_dir=None):
    """Create detailed prediction plot for a single model."""
    ckpt_name = ckpt_info['checkpoint']
    preds, gts = get_all_predictions(model_name, ckpt_name, dataset_type)

    if preds is None:
        return

    n_samples = min(num_samples, preds.shape[0])
    n_vars = min(5, preds.shape[2])  # Show up to 5 variables

    fig, axes = plt.subplots(n_samples, n_vars, figsize=(3*n_vars, 2.5*n_samples))

    for sample_idx in range(n_samples):
        for var_idx in range(n_vars):
            ax = axes[sample_idx, var_idx] if n_samples > 1 else axes[var_idx]

            pred = preds[sample_idx, :, var_idx]
            gt = gts[sample_idx, :, var_idx]
            x = np.arange(len(pred))

            ax.plot(x, gt, 'b-', linewidth=1.5, label='Actual', alpha=0.8)
            ax.plot(x, pred, 'r--', linewidth=1.5, label='Predicted', alpha=0.8)
            ax.fill_between(x, gt, pred, alpha=0.2, color='gray')

            # Calculate per-sample error
            mse = np.mean((pred - gt) ** 2)
            ax.set_title(f'S{sample_idx+1} V{var_idx+1} (MSE: {mse:.4f})', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

            if sample_idx == 0 and var_idx == n_vars - 1:
                ax.legend(loc='upper right', fontsize=7)

    plt.suptitle(f'{model_name.upper()} Detailed Predictions - {dataset_type.upper()}\n'
                f'Best Checkpoint: {ckpt_name} (MSE: {ckpt_info["mse"]:.6f})',
                fontsize=12, fontweight='bold')
    plt.tight_layout()

    if output_dir:
        output_path = os.path.join(output_dir, f'{model_name}_predictions_detail.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Detailed prediction saved to {output_path}")


def plot_full_sequence(model_name, ckpt_info, dataset_type, num_vars=6, output_dir=None):
    """
    Plot predictions for ENTIRE evaluation set as continuous time series.
    Shows all test predictions to visualize full prediction sequence.
    """
    ckpt_name = ckpt_info['checkpoint']
    preds, gts = get_all_predictions(model_name, ckpt_name, dataset_type)

    if preds is None:
        return

    n_samples, pred_len, n_total_vars = preds.shape
    n_vars_to_plot = min(num_vars, n_total_vars)

    # Create figure with one subplot per variable
    fig, axes = plt.subplots(n_vars_to_plot, 1, figsize=(20, 3 * n_vars_to_plot), sharex=True)
    if n_vars_to_plot == 1:
        axes = [axes]

    # Flatten predictions and ground truths for continuous view
    # Each sample predicts pred_len steps ahead
    total_len = n_samples * pred_len

    for var_idx in range(n_vars_to_plot):
        ax = axes[var_idx]

        # Flatten this variable across all samples
        pred_flat = preds[:, :, var_idx].flatten()
        gt_flat = gts[:, :, var_idx].flatten()
        x = np.arange(len(pred_flat))

        # Plot ground truth and predictions
        ax.plot(x, gt_flat, 'b-', linewidth=1, label='Ground Truth', alpha=0.8)
        ax.plot(x, pred_flat, 'r-', linewidth=1, label='Prediction', alpha=0.7)

        # Add vertical lines every pred_len to show prediction boundaries
        for i in range(1, n_samples):
            ax.axvline(x=i * pred_len, color='gray', linestyle=':', alpha=0.3)

        # Calculate MSE for this variable
        var_mse = np.mean((pred_flat - gt_flat) ** 2)

        ax.set_ylabel(f'Var {var_idx+1}\n(MSE: {var_mse:.4f})', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        if var_idx == 0:
            ax.legend(loc='upper right', fontsize=9)

    axes[-1].set_xlabel('Time Step (All Test Predictions)', fontsize=11)

    plt.suptitle(f'{model_name.upper()} Full Sequence Prediction - {dataset_type.upper()}\n'
                f'{n_samples} predictions × {pred_len} steps = {total_len} total points | '
                f'Overall MSE: {ckpt_info["mse"]:.4f}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_dir:
        output_path = os.path.join(output_dir, f'{model_name}_full_sequence.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Full sequence prediction saved to {output_path}")


def plot_all_models_full_sequence(best_ckpts, dataset_type, num_vars=3, output_path=None):
    """
    Create combined full sequence visualization for ALL models.
    Shows side-by-side comparison of predictions across entire test set.
    """
    models = list(best_ckpts.keys())
    n_models = len(models)

    # First, load all predictions
    all_preds = {}
    all_gts = {}
    n_samples = None
    pred_len = None

    for model_name in models:
        ckpt_name = best_ckpts[model_name]['checkpoint']
        print(f"Loading predictions for {model_name.upper()}...")
        preds, gts = get_all_predictions(model_name, ckpt_name, dataset_type)
        if preds is not None:
            all_preds[model_name] = preds
            all_gts[model_name] = gts
            if n_samples is None:
                n_samples, pred_len, n_total_vars = preds.shape

    if not all_preds:
        print("No predictions available!")
        return

    n_vars_to_plot = min(num_vars, n_total_vars)

    # Create figure: rows = models, cols = variables
    fig, axes = plt.subplots(n_models, n_vars_to_plot, figsize=(7 * n_vars_to_plot, 3 * n_models))

    # Model colors
    colors = {
        'itransformer': '#e74c3c',
        'timexer': '#3498db',
        'waxer': '#2ecc71',
        'taxer': '#9b59b6',
        'timesnet': '#f39c12',
        'wavenet': '#1abc9c'
    }

    for model_idx, model_name in enumerate(models):
        if model_name not in all_preds:
            continue

        preds = all_preds[model_name]
        gts = all_gts[model_name]
        mse = best_ckpts[model_name]['mse']

        for var_idx in range(n_vars_to_plot):
            ax = axes[model_idx, var_idx] if n_models > 1 else axes[var_idx]

            # Flatten
            pred_flat = preds[:, :, var_idx].flatten()
            gt_flat = gts[:, :, var_idx].flatten()
            x = np.arange(len(pred_flat))

            # Plot
            ax.plot(x, gt_flat, 'b-', linewidth=0.8, label='Ground Truth', alpha=0.7)
            ax.plot(x, pred_flat, color=colors.get(model_name, 'red'),
                   linewidth=0.8, label='Prediction', alpha=0.8)

            var_mse = np.mean((pred_flat - gt_flat) ** 2)

            if var_idx == 0:
                ax.set_ylabel(f'{model_name.upper()}\n(MSE: {mse:.4f})',
                             fontsize=10, fontweight='bold')
            if model_idx == 0:
                ax.set_title(f'Variable {var_idx + 1} (MSE: {var_mse:.4f})',
                            fontsize=11, fontweight='bold')
            if model_idx == n_models - 1:
                ax.set_xlabel('Time Step', fontsize=10)

            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

            if model_idx == 0 and var_idx == 0:
                ax.legend(loc='upper right', fontsize=8)

    plt.suptitle(f'Full Sequence Prediction Comparison - {dataset_type.upper()} Weather Dataset\n'
                f'{n_samples} test samples × {pred_len} prediction steps',
                fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Full sequence comparison saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate prediction visualizations for weather models')
    parser.add_argument('--dataset', type=str, required=True, choices=['korean', 'global'],
                        help='Dataset to visualize (korean/global)')
    args = parser.parse_args()

    dataset_type = args.dataset
    ds_config = DATASET_CONFIGS[dataset_type]
    eval_dir = os.path.join(ds_config['output_base'], 'eval_results')
    output_dir = ds_config['output_base']

    # Load best checkpoints
    best_ckpts = load_best_checkpoints(eval_dir)

    if not best_ckpts:
        print(f"No evaluation results found in {eval_dir}. Run eval_weather_checkpoints.py first.")
        return

    print("\n" + "="*60)
    print(f"Generating Prediction Visualizations - {dataset_type.upper()}")
    print("="*60)

    # 1. Create combined sample visualization
    print("\n1. Creating sample prediction comparison...")
    plot_all_predictions(
        best_ckpts,
        dataset_type,
        num_samples=3,
        num_vars=3,
        output_path=os.path.join(output_dir, 'full_prediction_comparison.png')
    )

    # 2. Create detailed plots for each model
    print("\n2. Creating detailed plots for each model...")
    for model_name, ckpt_info in best_ckpts.items():
        plot_single_model_detail(model_name, ckpt_info, dataset_type, num_samples=4, output_dir=output_dir)

    # 3. Create full sequence plots for each model
    print("\n3. Creating full sequence plots for each model...")
    for model_name, ckpt_info in best_ckpts.items():
        plot_full_sequence(model_name, ckpt_info, dataset_type, num_vars=6, output_dir=output_dir)

    # 4. Create combined full sequence comparison
    print("\n4. Creating combined full sequence comparison...")
    plot_all_models_full_sequence(
        best_ckpts,
        dataset_type,
        num_vars=3,
        output_path=os.path.join(output_dir, 'full_sequence_prediction.png')
    )

    print("\n" + "="*60)
    print("All visualizations complete!")
    print("="*60)


if __name__ == '__main__':
    main()
