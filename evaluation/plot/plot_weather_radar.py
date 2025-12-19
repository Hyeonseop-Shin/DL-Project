"""
Create radar/polygon chart comparing weather model performance.
Pentagon plot with 5 metrics: MSE, MAE, MAPE, RMSE, CORR

Usage:
    python plot_weather_radar.py --dataset korean
    python plot_weather_radar.py --dataset global
"""

import os
import json
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from math import pi


DATASET_CONFIGS = {
    'korean': {
        'eval_dir': 'results/korean/results_v1/eval_results',
        'output_dir': 'results/korean/results_v1'
    },
    'global': {
        'eval_dir': 'results/global/results_v1/eval_results',
        'output_dir': 'results/global/results_v1'
    }
}


def load_best_results(eval_dir):
    """Load best checkpoint results for each model."""
    models = ['itransformer', 'timexer', 'waxer', 'taxer']
    results = {}

    for model in models:
        csv_path = os.path.join(eval_dir, f'{model}_eval.csv')
        if not os.path.exists(csv_path):
            continue

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Find best by MSE
        best = min(rows, key=lambda x: float(x['mse']))
        results[model] = {
            'checkpoint': best['checkpoint'],
            'epoch': int(best['epoch']),
            'mse': float(best['mse']),
            'mae': float(best['mae']),
            'rmse': float(best['rmse']),
            'mape': float(best['mape']),
            'corr': float(best['corr'])
        }

    return results


def normalize_metrics(results):
    """
    Normalize metrics for radar chart plotting.
    For error metrics (MSE, MAE, MAPE, RMSE): lower is better, so we invert
    For CORR: higher is better (keep as is after normalization)
    Scale to [0.3, 1.0] range to emphasize differences.
    """
    metrics = ['mse', 'mae', 'mape', 'rmse', 'corr']
    error_metrics = ['mse', 'mae', 'mape', 'rmse']

    # Collect all values per metric
    all_values = {m: [] for m in metrics}
    for model, data in results.items():
        for m in metrics:
            all_values[m].append(data[m])

    # Calculate min/max for normalization
    normalized = {}
    for model, data in results.items():
        normalized[model] = {'raw': {}, 'scaled': {}}
        for m in metrics:
            min_val = min(all_values[m])
            max_val = max(all_values[m])
            range_val = max_val - min_val if max_val != min_val else 1

            raw = data[m]
            normalized[model]['raw'][m] = raw

            # Normalize to 0-1
            norm = (raw - min_val) / range_val

            # Invert error metrics so larger = better
            if m in error_metrics:
                norm = 1 - norm

            # Scale to [0.3, 1.0] range to emphasize differences
            norm = 0.3 + norm * 0.7

            normalized[model]['scaled'][m] = norm

    return normalized


def create_radar_chart(results, output_path, dataset_type):
    """Create radar/polygon chart."""

    # Normalize metrics
    normalized = normalize_metrics(results)

    # Define metrics for radar chart (5 vertices = pentagon)
    metrics = ['MSE', 'MAE', 'MAPE', 'RMSE', 'CORR']
    metric_keys = ['mse', 'mae', 'mape', 'rmse', 'corr']
    N = len(metrics)

    # Calculate angles for each metric
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the loop

    # Model colors - blue, green, purple, red
    colors = {
        'itransformer': '#e74c3c',  # Red
        'timexer': '#3498db',       # Blue
        'waxer': '#2ecc71',         # Green
        'taxer': '#9b59b6'          # Purple
    }

    # Create figure (smaller size)
    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))

    # Plot each model with solid line and circle markers
    for model, data in normalized.items():
        values = [data['scaled'][k] for k in metric_keys]
        values += values[:1]  # Complete the loop

        ax.plot(angles, values, 'o-',
                linewidth=2.5,
                label=model.upper(),
                color=colors.get(model, 'gray'),
                markersize=8)
        ax.fill(angles, values, alpha=0.08, color=colors.get(model, 'gray'))

    # Set metric labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')

    # Set radial limits
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)

    # Add title and legend
    title = f'Model Performance Comparison - {dataset_type.upper()} Weather Dataset\n(Larger area = Better performance)'
    plt.title(title, fontsize=14, fontweight='bold', pad=20)

    # Create legend with line handles
    handles = []
    labels = []
    for model, data in normalized.items():
        mse = data['raw']['mse']
        corr = data['raw']['corr']
        handle = plt.Line2D([0], [0],
                           color=colors.get(model, 'gray'),
                           linestyle='-',
                           marker='o',
                           linewidth=2.5,
                           markersize=6)
        handles.append(handle)
        labels.append(f"{model.upper()}\nMSE: {mse:.4f} | CORR: {corr:.4f}")

    ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.4, 1.05),
              fontsize=8, framealpha=0.9, handlelength=2)

    # Add raw values as annotations
    add_value_annotations(ax, normalized, angles, metric_keys)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Radar chart saved to {output_path}")


def add_value_annotations(ax, normalized, angles, metric_keys):
    """Add raw value annotations near best model for each metric."""
    # Find best model for each metric
    best_per_metric = {}
    for m_idx, m_key in enumerate(metric_keys):
        if m_key == 'corr':
            # Higher is better for CORR
            best_model = max(normalized.keys(),
                           key=lambda x: normalized[x]['raw'][m_key])
        else:
            # Lower is better for error metrics
            best_model = min(normalized.keys(),
                           key=lambda x: normalized[x]['raw'][m_key])
        best_per_metric[m_key] = (best_model, normalized[best_model]['raw'][m_key])

    # Add small text annotations at each vertex showing best value
    for m_idx, m_key in enumerate(metric_keys):
        angle = angles[m_idx]
        best_model, best_val = best_per_metric[m_key]

        ax.annotate(f'Best: {best_val:.4f}\n({best_model})',
                   xy=(angle, 1.05), fontsize=7,
                   ha='center', va='bottom', color='darkgray')


def create_detailed_table(results, output_path, dataset_type):
    """Create a detailed comparison table as an image."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    # Prepare data
    models = list(results.keys())
    metrics = ['MSE', 'MAE', 'RMSE', 'MAPE', 'CORR', 'Best Epoch']

    # Table data
    cell_text = []
    for model in models:
        data = results[model]
        row = [
            f"{data['mse']:.6f}",
            f"{data['mae']:.6f}",
            f"{data['rmse']:.6f}",
            f"{data['mape']:.6f}",
            f"{data['corr']:.6f}",
            str(data['epoch'])
        ]
        cell_text.append(row)

    # Create table
    table = ax.table(
        cellText=cell_text,
        rowLabels=[m.upper() for m in models],
        colLabels=metrics,
        cellLoc='center',
        loc='center'
    )

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Color best values in each column
    for col_idx, metric in enumerate(metrics[:-1]):  # Exclude 'Best Epoch'
        values = [float(cell_text[row_idx][col_idx]) for row_idx in range(len(models))]

        if metric == 'CORR':
            best_idx = values.index(max(values))
        else:
            best_idx = values.index(min(values))

        table[(best_idx + 1, col_idx)].set_facecolor('#90EE90')  # Light green

    plt.title(f'Model Performance Comparison - {dataset_type.upper()} Weather Dataset\n(Best Checkpoints)',
              fontsize=14, fontweight='bold', pad=20)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Comparison table saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Create radar chart for weather models')
    parser.add_argument('--dataset', type=str, required=True, choices=['korean', 'global'],
                        help='Dataset to visualize (korean/global)')
    args = parser.parse_args()

    dataset_type = args.dataset
    config = DATASET_CONFIGS[dataset_type]

    # Load results
    results = load_best_results(config['eval_dir'])

    if not results:
        print(f"No evaluation results found in {config['eval_dir']}. Run eval_weather_checkpoints.py first.")
        return

    print(f"\n{'='*60}")
    print(f"Best checkpoint results - {dataset_type.upper()}")
    print("="*60)
    for model, data in sorted(results.items(), key=lambda x: x[1]['mse']):
        print(f"{model.upper():<15} Epoch {data['epoch']:>3}  "
              f"MSE: {data['mse']:.6f}  MAE: {data['mae']:.6f}  "
              f"CORR: {data['corr']:.6f}")
    print("-" * 60)

    # Create visualizations
    radar_path = os.path.join(config['output_dir'], f'{dataset_type}_model_radar_comparison.png')
    table_path = os.path.join(config['output_dir'], f'{dataset_type}_model_comparison_table.png')

    create_radar_chart(results, radar_path, dataset_type)
    create_detailed_table(results, table_path, dataset_type)

    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
