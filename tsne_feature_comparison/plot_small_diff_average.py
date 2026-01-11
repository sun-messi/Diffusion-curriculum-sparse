#!/usr/bin/env python3
"""
Plot average Silhouette Score for classifications with smallest C-CS difference at 40k.
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('outputs/silhouette_binary_all.csv')

# Classifications with smallest C-CS difference at 40k
SMALL_DIFF = ['attractive', 'bangs']

df_selected = df[df['Classification'].isin(SMALL_DIFF)]

print(f"Selected classifications (small C-CS diff): {SMALL_DIFF}")
print(f"Number of rows: {len(df_selected)}")

avg_df = df_selected.groupby('Step')[['Baseline', 'CS_Mode', 'C_Mode']].mean().reset_index()
avg_df = avg_df.sort_values('Step')

print("\nAverage Silhouette Scores:")
print(avg_df.to_string(index=False))

# Check 40k
row_40k = avg_df[avg_df['Step'] == 40000]
print(f"\n40k: CS={row_40k['CS_Mode'].values[0]:.4f}, C={row_40k['C_Mode'].values[0]:.4f}")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

steps = avg_df['Step'].values
baseline = avg_df['Baseline'].values
cs_mode = avg_df['CS_Mode'].values
c_mode = avg_df['C_Mode'].values

ax.plot(steps, baseline, 'b-o', label='Baseline', linewidth=2, markersize=8)
ax.plot(steps, cs_mode, 'r-s', label='CS Mode (Curriculum+Sparsity)', linewidth=2, markersize=8)
ax.plot(steps, c_mode, 'g-^', label='C Mode (Curriculum only)', linewidth=2, markersize=8)

ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Average Silhouette Score (cosine)', fontsize=12)
ax.set_title('Feature Clustering Quality - Small C-CS Difference\n(Mean of: attractive, bangs)', fontsize=12)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

ax.set_xticks(steps)
ax.set_xticklabels([f'{s//1000}k' for s in steps], rotation=45)

plt.tight_layout()
plt.savefig('outputs/silhouette_small_diff_average.png', dpi=150, bbox_inches='tight')
print("\nSaved: outputs/silhouette_small_diff_average.png")
