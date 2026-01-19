#!/usr/bin/env python3
"""
Plot average Silhouette Score for classifications with middle-peak pattern.
These are high-level semantic features that show inverted-U shape.
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('outputs/silhouette_binary_all.csv')

# Classifications with middle peak (not at 20k or 200k)
MIDDLE_PEAK = ['attractive', 'bangs', 'gender', 'heavy_makeup', 'high_cheekbones', 'smiling']

df_selected = df[df['Classification'].isin(MIDDLE_PEAK)]

print(f"Selected classifications (middle-peak pattern): {MIDDLE_PEAK}")
print(f"Number of rows: {len(df_selected)}")

avg_df = df_selected.groupby('Step')[['Baseline', 'CS_Mode', 'C_Mode']].mean().reset_index()
avg_df = avg_df.sort_values('Step')

print("\nAverage Silhouette Scores:")
print(avg_df.to_string(index=False))

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

steps = avg_df['Step'].values
baseline = avg_df['Baseline'].values
cs_mode = avg_df['CS_Mode'].values
c_mode = avg_df['C_Mode'].values

ax.plot(steps, baseline, 'b-o', label='Standard training', linewidth=2, markersize=8)
ax.plot(steps, cs_mode, 'r-s', label='CS Mode (Curriculum+Sparsity)', linewidth=2, markersize=8)
ax.plot(steps, c_mode, 'g-^', label='C Mode (Curriculum only)', linewidth=2, markersize=8)

ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Average Silhouette Score (cosine)', fontsize=12)
ax.set_title('Feature Clustering Quality - High-Level Semantic Features\n(Mean of: gender, smiling, attractive, heavy_makeup, bangs, high_cheekbones)', fontsize=12)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

ax.set_xticks(steps)
ax.set_xticklabels([f'{s//1000}k' for s in steps], rotation=45)

plt.tight_layout()
plt.savefig('outputs/silhouette_middle_peak_average.png', dpi=150, bbox_inches='tight')
print("\nSaved: outputs/silhouette_middle_peak_average.png")

avg_df.to_csv('outputs/silhouette_middle_peak_average.csv', index=False)
print("Saved: outputs/silhouette_middle_peak_average.csv")
