"""Plot convergence speed comparison for CIFAR-10 U-ViT models."""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path


def parse_fid_summary(filepath):
    """Parse FID summary file and return dict of step -> FID."""
    fid_dict = {}
    with open(filepath, 'r') as f:
        for line in f:
            match = re.match(r'(\d+)_ema: FID = ([\d.]+)', line)
            if match:
                step = int(match.group(1))
                fid = float(match.group(2))
                fid_dict[step] = fid
    return fid_dict


def find_first_reach(fid_dict, threshold):
    """Find the first step that reaches below the threshold."""
    sorted_steps = sorted(fid_dict.keys())
    for step in sorted_steps:
        if fid_dict[step] < threshold:
            return step
    return None


def main():
    parser = argparse.ArgumentParser(description='Plot convergence speed comparison')
    parser.add_argument('--eval-results', type=str, default='../eval_results',
                        help='Path to eval_results directory')
    parser.add_argument('--output', type=str, default='outputs/cifar10_convergence_speed.png',
                        help='Output path for the plot')
    args = parser.parse_args()

    eval_results = Path(args.eval_results)

    # Model configs: name -> (dir_name, timestamp)
    models = {
        'baseline': ('cifar10_uvit_small', '20260110_065028'),
        'curriculum (_c)': ('cifar10_uvit_small_c', '20260110_125959'),
        'curriculum+sparsity (_cs)': ('cifar10_uvit_small_cs', '20260110_094434'),
        'sparsity (_s)': ('cifar10_uvit_small_s', '20260110_193200'),
    }

    # Parse FID results
    results = {}
    for name, (dir_name, timestamp) in models.items():
        summary_path = eval_results / dir_name / timestamp / 'fid_summary.txt'
        if summary_path.exists():
            results[name] = parse_fid_summary(summary_path)
            print(f"Loaded {name}: {len(results[name])} checkpoints")
        else:
            print(f"Warning: {summary_path} not found")

    if not results:
        print("No results found!")
        return

    # FID thresholds to analyze
    thresholds = [80, 50, 30, 20]

    # Colors for models
    colors = {
        'baseline': '#1f77b4',
        'curriculum (_c)': '#ff7f0e',
        'curriculum+sparsity (_cs)': '#2ca02c',
        'sparsity (_s)': '#d62728',
    }

    # Find first step to reach each threshold
    convergence_data = {name: {} for name in results}
    for name, fid_dict in results.items():
        for threshold in thresholds:
            step = find_first_reach(fid_dict, threshold)
            convergence_data[name][threshold] = step

    # Print convergence data
    print("\n=== Convergence Speed Analysis ===")
    print(f"{'Model':<25} " + " ".join(f"FID<{t:3}" for t in thresholds))
    print("-" * 65)
    for name in results:
        steps_str = []
        for t in thresholds:
            step = convergence_data[name][t]
            if step:
                steps_str.append(f"{step//1000:4}k")
            else:
                steps_str.append("  N/A")
        print(f"{name:<25} " + "   ".join(steps_str))

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(thresholds))
    width = 0.2

    model_names = list(results.keys())
    for i, name in enumerate(model_names):
        values = []
        for t in thresholds:
            step = convergence_data[name][t]
            values.append(step / 1000 if step else 220)  # 220 means not reached

        bars = ax.bar(x + (i - 1.5) * width, values, width, label=name, color=colors[name])

        # Add value labels
        for bar, val, t in zip(bars, values, thresholds):
            step = convergence_data[name][t]
            if step:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                       f'{int(val)}k', ha='center', va='bottom', fontsize=8, rotation=0)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                       'N/A', ha='center', va='bottom', fontsize=8, color='red')

    ax.set_xlabel('FID Threshold', fontsize=12)
    ax.set_ylabel('Training Steps (k) to First Reach Threshold', fontsize=12)
    ax.set_title('Convergence Speed: Steps to First Reach FID Threshold\n(Lower is Better)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'FID < {t}' for t in thresholds])
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 220)

    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved: {output_path}')


if __name__ == '__main__':
    main()
