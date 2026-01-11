#!/usr/bin/env python3
"""
Run comparison across multiple models and steps.
Calls compute_single.py for each combination, collects results, plots.

Usage:
    python run_comparison.py

Modify MODELS and STEPS below to change what to compare.
"""

import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============ MODIFY THESE ============
MODELS = {
    'Baseline': '../workdir/celeba64_uvit_small/default_20260101_030900',
    'CS': '../workdir/celeba64_uvit_small_cs/default_20260101_073037',
    'C': '../workdir/celeba64_uvit_small_c/default',
}

STEPS = [40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000]

SAMPLES_PER_CLASS = 500
OUTPUT_CSV = 'outputs/comparison.csv'
OUTPUT_PNG = 'outputs/comparison.png'
# ======================================


def run_single(ckpt_dir, step, samples_per_class=500):
    """Run compute_single.py and return score."""
    cmd = [
        'python', 'compute_single.py',
        '--ckpt_dir', ckpt_dir,
        '--step', str(step),
        '--samples_per_class', str(samples_per_class),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout.strip()
        if output:
            _, score = output.split(',')
            return float(score)
    except Exception as e:
        print(f"Error: {e}")
    return float('nan')


def main():
    results = []

    print(f"Running comparison: {len(MODELS)} models x {len(STEPS)} steps")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Steps: {STEPS}")
    print()

    for step in STEPS:
        row = {'Step': step}
        for model_name, ckpt_dir in MODELS.items():
            print(f"  {model_name} @ {step}k...", end=' ', flush=True)
            score = run_single(ckpt_dir, step, SAMPLES_PER_CLASS)
            row[model_name] = score
            print(f"{score:.4f}")
        results.append(row)

    # Save CSV
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_CSV) or '.', exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'Baseline': 'blue', 'CS': 'red', 'C': 'green'}
    markers = {'Baseline': 'o', 'CS': 's', 'C': '^'}

    for model_name in MODELS.keys():
        ax.plot(df['Step'], df[model_name],
                color=colors.get(model_name, 'black'),
                marker=markers.get(model_name, 'o'),
                label=model_name, linewidth=2, markersize=6)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Silhouette Score (cosine)', fontsize=12)
    ax.set_title('Feature Clustering Quality\n(CelebA Gender×Age 4-class)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(STEPS)
    ax.set_xticklabels([f'{s//1000}k' for s in STEPS], rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_PNG}")

    # Print summary
    print("\n=== Results ===")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
