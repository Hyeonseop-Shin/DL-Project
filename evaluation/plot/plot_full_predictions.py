"""
Generate full prediction visualizations for all models.
Shows predicted vs actual values for the best checkpoint of each model.
"""

import os
import json
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from long_term_forecasting import Long_Term_Forecasting
from eval_all_checkpoints import create_args, MODEL_CONFIGS


def load_best_checkpoints(eval_dir='results/results_v1/eval_results'):
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


def get_predictions(model_name, ckpt_name, num_samples=5):
    """Get predictions from a model checkpoint."""
    ckpt_path = f'checkpoints/{model_name}_v1'
    args = create_args(model_name, ckpt_path, ckpt_name)

    try:
        task = Long_Term_Forecasting(args)
        task._load_checkpoint(ckpt_name=ckpt_name, model=task.model, verbose=False)
        _, test_loader, _ = task._get_data_loader(flag='test')

        # Get underlying model for inference
        model = task.model.module if hasattr(task.model, 'module') else task.model
        model.eval()

        predictions = []
        ground_truths = []
        samples_collected = 0

        with torch.no_grad():
            for batch in test_loader:
                batch_x = batch[0].float().to(task.device)
                batch_y = batch[1].float().to(task.device)

                outputs = model(batch_x)

                predictions.append(outputs.cpu().numpy())
                ground_truths.append(batch_y.cpu().numpy())
                samples_collected += batch_x.size(0)

                if samples_collected >= num_samples:
                    break

        model.train()

        preds = np.concatenate(predictions, axis=0)[:num_samples]
        gts = np.concatenate(ground_truths, axis=0)[:num_samples]

        return preds, gts

    except Exception as e:
        print(f"Error getting predictions for {model_name}: {e}")
        return None, None


def plot_all_predictions(best_ckpts, num_samples=3, num_vars=3,
                        output_path='results/results_v1/full_prediction_comparison.png'):
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
        preds, gts = get_predictions(model_name, ckpt_name, num_samples)

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

    plt.suptitle('Model Predictions vs Ground Truth\n(Solid: Actual, Dashed: Predicted)',
                fontsize=14, fontweight='bold', y=1.02)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nPrediction visualization saved to {output_path}")


def plot_single_model_detail(model_name, ckpt_info, num_samples=6,
                             output_dir='results/results_v1'):
    """Create detailed prediction plot for a single model."""
    ckpt_name = ckpt_info['checkpoint']
    preds, gts = get_predictions(model_name, ckpt_name, num_samples)

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

    plt.suptitle(f'{model_name.upper()} Detailed Predictions\n'
                f'Best Checkpoint: {ckpt_name} (MSE: {ckpt_info["mse"]:.6f})',
                fontsize=12, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'{model_name}_predictions_detail.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Detailed prediction saved to {output_path}")


def main():
    # Load best checkpoints
    best_ckpts = load_best_checkpoints()

    if not best_ckpts:
        print("No evaluation results found. Run eval_all_checkpoints.py first.")
        return

    print("\n" + "="*60)
    print("Generating Prediction Visualizations")
    print("="*60)

    # Create combined visualization
    plot_all_predictions(best_ckpts, num_samples=3, num_vars=3)

    # Create detailed plots for each model
    print("\nGenerating detailed plots for each model...")
    for model_name, ckpt_info in best_ckpts.items():
        plot_single_model_detail(model_name, ckpt_info, num_samples=4)

    print("\n" + "="*60)
    print("All visualizations complete!")
    print("="*60)


if __name__ == '__main__':
    main()
