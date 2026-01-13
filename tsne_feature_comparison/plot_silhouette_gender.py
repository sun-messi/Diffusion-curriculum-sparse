#!/usr/bin/env python3
"""
Plot Silhouette Score comparison for CelebA Gender classification.
Style aligned with tsne_cifar10_comparison/plot_silhouette.py
"""

import pandas as pd
import matplotlib.pyplot as plt

# Set global style
plt.rcParams['font.size'] = 32
plt.rcParams['axes.labelsize'] = 32
plt.rcParams['axes.titlesize'] = 36
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28
plt.rcParams['legend.fontsize'] = 26

# Read data
df = pd.read_csv('outputs/silhouette_binary_all.csv')

def plot_silhouette(classification, output_name):
    df_filtered = df[df['Classification'] == classification].copy()

    print(f"\n{classification.capitalize()} Silhouette Scores:")
    print(df_filtered.to_string(index=False))

    fig, ax = plt.subplots(figsize=(14, 10))

    steps = df_filtered['Step'].values
    baseline = df_filtered['Baseline'].values
    cs_mode = df_filtered['CS_Mode'].values
    c_mode = df_filtered['C_Mode'].values

    n_markers = 10
    markevery = max(1, len(steps) // n_markers)

    ax.plot(steps, baseline, 'b-o', label='Baseline', linewidth=6, markersize=18, markevery=markevery)
    ax.plot(steps, cs_mode, 'r-s', label='CS Mode', linewidth=6, markersize=18, markevery=markevery)
    ax.plot(steps, c_mode, 'g-^', label='C Mode', linewidth=6, markersize=18, markevery=markevery)

    ax.set_xlabel('Training Step (k)', labelpad=15)
    ax.set_ylabel('Silhouette Score', labelpad=15)
    ax.legend(loc='best', handlelength=3)
    ax.grid(True, linestyle='--', alpha=0.7)

    show_steps = [20000, 60000, 100000, 140000, 200000]
    ax.set_xticks(show_steps)
    ax.set_xticklabels(['%dk' % (s//1000) for s in show_steps])

    plt.tight_layout()
    plt.savefig(f'outputs/silhouette_{output_name}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: outputs/silhouette_{output_name}.png")
    plt.close()

# Plot all classifications
classifications = df['Classification'].unique()
for cls in classifications:
    plot_silhouette(cls, cls)
