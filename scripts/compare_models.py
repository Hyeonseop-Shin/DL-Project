#!/usr/bin/env python
"""
Model Comparison Script

Loads test results from multiple models and generates comparison visualizations.

Usage:
    python scripts/compare_models.py --result_path results/
    python scripts/compare_models.py --result_path results/ --output_dir results/comparison/
"""

import os
import re
import argparse
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualization import (
    plot_model_comparison,
    create_metrics_table,
)


def parse_result_file(filepath):
    """
    Parse a test result file and extract metrics.

    Args:
        filepath: Path to the result file

    Returns:
        Dict with metric values
    """
    metrics = {}
    metric_patterns = {
        'MAE': r'MAE:\s*([\d.]+)',
        'MSE': r'MSE:\s*([\d.]+)',
        'RMSE': r'RMSE:\s*([\d.]+)',
        'MAPE': r'MAPE:\s*([\d.]+)',
        'MSPE': r'MSPE:\s*([\d.]+)',
        'CORR': r'CORR:\s*([\d.]+)',
    }

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        for metric_name, pattern in metric_patterns.items():
            match = re.search(pattern, content)
            if match:
                metrics[metric_name] = float(match.group(1))

    except Exception as e:
        print(f"Error parsing {filepath}: {e}")

    return metrics


def extract_model_name(filename):
    """Extract model name from result filename."""
    # Expected format: testing_{model_name}_e{epoch}_s{seq}_p{pred}.txt
    match = re.match(r'testing_([a-z_]+)_e\d+_s\d+_p\d+\.txt', filename, re.IGNORECASE)
    if match:
        model_name = match.group(1)
        # Format model name
        name_map = {
            'itransformer': 'iTransformer',
            'timexer': 'TimeXer',
            'wavenet': 'WaveNet',
            'timesnet': 'TimesNet',
            'waxer': 'WaXer',
            'taxer': 'TaXer',
            'timexer_hybrid': 'HybridTimeXer',
            'waveformer': 'WaveFormer',
            'timesformer': 'TimesFormer',
            'watiformer': 'WaTiFormer',
        }
        return name_map.get(model_name.lower(), model_name)
    return None


def load_all_results(result_path):
    """
    Load all test results from a directory.

    Args:
        result_path: Path to results directory

    Returns:
        Dict mapping model names to metric dicts
    """
    results = {}

    if not os.path.exists(result_path):
        print(f"Results directory not found: {result_path}")
        return results

    for filename in os.listdir(result_path):
        if filename.startswith('testing_') and filename.endswith('.txt'):
            filepath = os.path.join(result_path, filename)
            model_name = extract_model_name(filename)

            if model_name:
                metrics = parse_result_file(filepath)
                if metrics:
                    results[model_name] = metrics
                    print(f"Loaded results for {model_name}: {metrics}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Compare model performance')
    parser.add_argument('--result_path', type=str, default='results/',
                        help='Path to results directory')
    parser.add_argument('--output_dir', type=str, default='results/comparison/',
                        help='Output directory for comparison figures')
    parser.add_argument('--metrics', type=str, nargs='+',
                        default=['MAE', 'MSE', 'RMSE'],
                        help='Metrics to compare')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load all results
    print(f"\nLoading results from: {args.result_path}")
    results = load_all_results(args.result_path)

    if not results:
        print("No results found. Please run model testing first.")
        print("Example: python main.py --model itransformer --mode test --ckpt_name <checkpoint>")
        return

    print(f"\nFound results for {len(results)} models: {list(results.keys())}")

    # Generate comparison bar chart
    print("\nGenerating comparison bar chart...")
    bar_chart_path = os.path.join(args.output_dir, 'model_comparison_bar.png')
    plot_model_comparison(
        results,
        metrics=args.metrics,
        save_path=bar_chart_path,
        title='Model Performance Comparison'
    )

    # Generate metrics table
    print("\nGenerating metrics table...")
    table_path = os.path.join(args.output_dir, 'model_comparison_table.png')
    df, _ = create_metrics_table(
        results,
        metrics=['MAE', 'MSE', 'RMSE', 'MAPE', 'CORR'],
        save_path=table_path,
        highlight_best=True
    )

    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(df.to_string())
    print("=" * 60)

    # Find best model for each metric
    print("\nBest Models:")
    for metric in df.columns:
        if metric == 'CORR':
            best = df[metric].idxmax()
            print(f"  {metric}: {best} ({df.loc[best, metric]:.4f})")
        else:
            best = df[metric].idxmin()
            print(f"  {metric}: {best} ({df.loc[best, metric]:.4f})")

    print(f"\nComparison figures saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
