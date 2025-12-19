"""
Visualization utilities for presentation-quality graphs.

This module provides functions for creating publication-quality
visualizations of time series forecasting results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette for consistent styling
COLORS = {
    'ground_truth': '#2E86AB',  # Blue
    'forecast': '#E94F37',      # Red
    'past': '#A23B72',          # Purple
    'highlight': '#F18F01',     # Orange
    'background': '#F5F5F5',    # Light gray
}

MODEL_COLORS = {
    'iTransformer': '#2E86AB',
    'TimeXer': '#E94F37',
    'WaveNet': '#A23B72',
    'TimesNet': '#F18F01',
    'WaXer': '#4CAF50',
    'TaXer': '#9C27B0',
}


def plot_forecast(
    forecast_data,
    ground_truth,
    past_data=None,
    dim_names=None,
    metrics=None,
    save_path=None,
    title=None,
    show_metrics=True,
    figsize=None
):
    """
    Create publication-quality forecast visualization.

    Args:
        forecast_data: Predicted values [T_pred, D] or [T_pred]
        ground_truth: Ground truth values [T_pred, D] or [T_pred]
        past_data: Historical data [T_past, D] or [T_past] (optional)
        dim_names: Names for each dimension (optional)
        metrics: Dict with metric values {'MAE': ..., 'RMSE': ...} (optional)
        save_path: Path to save the figure (optional)
        title: Figure title (optional)
        show_metrics: Whether to show metrics on plot
        figsize: Figure size tuple (optional)

    Returns:
        matplotlib figure object
    """
    # Convert to numpy if needed
    forecast_np = _to_numpy(forecast_data)
    gt_np = _to_numpy(ground_truth)
    past_np = _to_numpy(past_data) if past_data is not None else None

    # Handle 1D arrays
    if forecast_np.ndim == 1:
        forecast_np = forecast_np[:, np.newaxis]
        gt_np = gt_np[:, np.newaxis]
        if past_np is not None:
            past_np = past_np[:, np.newaxis]

    T_pred, D = forecast_np.shape
    T_past = past_np.shape[0] if past_np is not None else 0

    # Set figure size
    if figsize is None:
        figsize = (12, 3 * D)

    fig, axes = plt.subplots(D, 1, figsize=figsize, sharex=True)
    if D == 1:
        axes = [axes]

    # Create time indices
    if past_np is not None:
        full_gt = np.concatenate([past_np, gt_np], axis=0)
        time_full = np.arange(T_past + T_pred)
        time_pred = np.arange(T_past, T_past + T_pred)
        pred_start = T_past
    else:
        full_gt = gt_np
        time_full = np.arange(T_pred)
        time_pred = time_full
        pred_start = 0

    for i, ax in enumerate(axes):
        # Plot ground truth
        ax.plot(time_full, full_gt[:, i], color=COLORS['ground_truth'],
                linewidth=1.5, label='Ground Truth', alpha=0.8)

        # Plot forecast
        ax.plot(time_pred, forecast_np[:, i], color=COLORS['forecast'],
                linewidth=1.5, linestyle='--', label='Forecast', alpha=0.9)

        # Highlight prediction region
        ax.axvspan(pred_start, T_past + T_pred if past_np is not None else T_pred,
                   alpha=0.1, color=COLORS['highlight'], label='Prediction Region')

        # Add vertical line at prediction start
        if past_np is not None:
            ax.axvline(pred_start, color=COLORS['highlight'],
                       linestyle=':', linewidth=1.5, alpha=0.7)

        # Compute per-dimension metrics if not provided
        if show_metrics:
            mae = np.mean(np.abs(forecast_np[:, i] - gt_np[:, i]))
            rmse = np.sqrt(np.mean((forecast_np[:, i] - gt_np[:, i])**2))
            ax.text(0.02, 0.98, f'MAE: {mae:.4f}  RMSE: {rmse:.4f}',
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Labels
        dim_name = dim_names[i] if dim_names and i < len(dim_names) else f'Dim {i}'
        ax.set_ylabel(dim_name)

        if i == 0:
            ax.legend(loc='upper right', framealpha=0.9)

    axes[-1].set_xlabel('Timestep')
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Title
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    elif metrics:
        fig.suptitle(f"Forecast Results | MAE: {metrics.get('MAE', 'N/A'):.4f}, "
                     f"RMSE: {metrics.get('RMSE', 'N/A'):.4f}",
                     fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_model_comparison(
    results_dict,
    metrics=['MAE', 'MSE', 'RMSE'],
    save_path=None,
    title="Model Comparison",
    figsize=(12, 5)
):
    """
    Create bar chart comparing model performance.

    Args:
        results_dict: Dict mapping model names to metric dicts
                      e.g., {'iTransformer': {'MAE': 0.1, 'MSE': 0.02}, ...}
        metrics: List of metrics to plot
        save_path: Path to save the figure (optional)
        title: Figure title
        figsize: Figure size tuple

    Returns:
        matplotlib figure object
    """
    models = list(results_dict.keys())
    n_models = len(models)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_models)
    width = 0.6

    for idx, (ax, metric) in enumerate(zip(axes, metrics)):
        values = [results_dict[m].get(metric, 0) for m in models]
        colors = [MODEL_COLORS.get(m, '#808080') for m in models]

        bars = ax.bar(x, values, width, color=colors, edgecolor='white', linewidth=1.2)

        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_ylabel(metric, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def create_metrics_table(
    results_dict,
    metrics=['MAE', 'MSE', 'RMSE', 'MAPE', 'CORR'],
    save_path=None,
    highlight_best=True
):
    """
    Create a formatted metrics table as an image.

    Args:
        results_dict: Dict mapping model names to metric dicts
        metrics: List of metrics to include
        save_path: Path to save the figure (optional)
        highlight_best: Whether to highlight best values

    Returns:
        pandas DataFrame and matplotlib figure
    """
    # Build DataFrame
    data = {}
    for model, metric_dict in results_dict.items():
        data[model] = {m: metric_dict.get(m, np.nan) for m in metrics}

    df = pd.DataFrame(data).T
    df.index.name = 'Model'

    # Create figure with table
    fig, ax = plt.subplots(figsize=(len(metrics) * 1.5 + 2, len(results_dict) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')

    # Format values
    cell_text = []
    for idx, row in df.iterrows():
        row_text = []
        for col in df.columns:
            val = row[col]
            if np.isnan(val):
                row_text.append('-')
            else:
                row_text.append(f'{val:.4f}')
        cell_text.append(row_text)

    # Create table
    table = ax.table(
        cellText=cell_text,
        colLabels=df.columns.tolist(),
        rowLabels=df.index.tolist(),
        cellLoc='center',
        loc='center',
        colColours=['#E8E8E8'] * len(df.columns),
        rowColours=['#E8E8E8'] * len(df.index)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Highlight best values (lower is better for most metrics, higher for CORR)
    if highlight_best:
        for col_idx, col in enumerate(df.columns):
            if col == 'CORR':
                best_idx = df[col].idxmax()
            else:
                best_idx = df[col].idxmin()
            row_idx = list(df.index).index(best_idx)
            table[(row_idx + 1, col_idx)].set_facecolor('#90EE90')  # Light green

    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Table saved to {save_path}")

        # Also save as CSV
        csv_path = save_path.replace('.png', '.csv').replace('.pdf', '.csv')
        df.to_csv(csv_path)
        print(f"CSV saved to {csv_path}")

    return df, fig


def plot_training_curves(
    train_losses,
    val_losses=None,
    model_name='Model',
    save_path=None,
    figsize=(10, 6)
):
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch (optional)
        model_name: Name of the model for title
        save_path: Path to save the figure (optional)
        figsize: Figure size tuple

    Returns:
        matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, color=COLORS['ground_truth'],
            linewidth=2, marker='o', markersize=4, label='Train Loss')

    if val_losses is not None:
        ax.plot(epochs, val_losses, color=COLORS['forecast'],
                linewidth=2, marker='s', markersize=4, label='Val Loss')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'{model_name} Training Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_multi_model_comparison(
    forecasts_dict,
    ground_truth,
    past_data=None,
    dim_idx=0,
    save_path=None,
    title="Multi-Model Forecast Comparison",
    figsize=(14, 6)
):
    """
    Plot forecasts from multiple models on the same plot.

    Args:
        forecasts_dict: Dict mapping model names to forecast arrays
        ground_truth: Ground truth values
        past_data: Historical data (optional)
        dim_idx: Which dimension to plot
        save_path: Path to save the figure (optional)
        title: Figure title
        figsize: Figure size tuple

    Returns:
        matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    gt_np = _to_numpy(ground_truth)
    past_np = _to_numpy(past_data) if past_data is not None else None

    if gt_np.ndim == 1:
        gt_np = gt_np[:, np.newaxis]
        if past_np is not None:
            past_np = past_np[:, np.newaxis]

    T_pred = gt_np.shape[0]
    T_past = past_np.shape[0] if past_np is not None else 0

    # Create time indices
    if past_np is not None:
        full_gt = np.concatenate([past_np[:, dim_idx], gt_np[:, dim_idx]])
        time_full = np.arange(T_past + T_pred)
        time_pred = np.arange(T_past, T_past + T_pred)
        pred_start = T_past
    else:
        full_gt = gt_np[:, dim_idx]
        time_full = np.arange(T_pred)
        time_pred = time_full
        pred_start = 0

    # Plot ground truth
    ax.plot(time_full, full_gt, color='black', linewidth=2,
            label='Ground Truth', alpha=0.8)

    # Highlight prediction region
    ax.axvspan(pred_start, time_full[-1] + 1,
               alpha=0.1, color=COLORS['highlight'])

    # Plot each model's forecast
    for model_name, forecast in forecasts_dict.items():
        forecast_np = _to_numpy(forecast)
        if forecast_np.ndim == 1:
            forecast_np = forecast_np[:, np.newaxis]

        color = MODEL_COLORS.get(model_name, None)
        ax.plot(time_pred, forecast_np[:, dim_idx], linewidth=1.5,
                linestyle='--', label=model_name, color=color, alpha=0.9)

    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def _to_numpy(x):
    """Convert input to numpy array."""
    if x is None:
        return None
    if hasattr(x, 'detach'):  # PyTorch tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)
