#!/usr/bin/env python3
"""
Plot Silhouette Score comparison for CIFAR-10 experiments.
Style aligned with plotting_standard.md
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
df = pd.read_csv('outputs/silhouette_cifar10.csv')

print("Silhouette Scores:")
print(df.to_string(index=False))

# Plot
fig, ax = plt.subplots(figsize=(14, 10))

steps = df['Step'].values
baseline = df['Baseline'].values
cs_mode = df['CS_Mode'].values
c_mode = df['C_Mode'].values
s_mode = df['S_Mode'].values if 'S_Mode' in df.columns else None

# Only show ~10 markers
n_markers = 10
markevery = max(1, len(steps) // n_markers)

ax.plot(steps, baseline, 'b-o', label='Baseline', linewidth=6, markersize=18, markevery=markevery)
ax.plot(steps, cs_mode, 'r-s', label='CS Mode', linewidth=6, markersize=18, markevery=markevery)
ax.plot(steps, c_mode, 'g-^', label='C Mode', linewidth=6, markersize=18, markevery=markevery)
if s_mode is not None:
    ax.plot(steps, s_mode, 'm-d', label='S Mode', linewidth=6, markersize=18, markevery=markevery)

ax.set_xlabel('Training Step (k)', labelpad=15)
ax.set_ylabel('Silhouette Score', labelpad=15)
ax.legend(loc='best', handlelength=3)
ax.grid(True, linestyle='--', alpha=0.7)

show_steps = [20000, 60000, 100000, 140000, 200000]
ax.set_xticks(show_steps)
ax.set_xticklabels(['%dk' % (s//1000) for s in show_steps])

plt.tight_layout()
plt.savefig('outputs/silhouette_cifar10.png', dpi=300, bbox_inches='tight')
print("\nSaved: outputs/silhouette_cifar10.png")
