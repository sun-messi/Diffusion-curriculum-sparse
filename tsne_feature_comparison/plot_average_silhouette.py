#!/usr/bin/env python3
"""
Plot average Silhouette Score across all binary classifications.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data
df = pd.read_csv('outputs/silhouette_binary_all.csv')

# Average across all classifications for each step
avg_df = df.groupby('Step')[['Baseline', 'CS_Mode', 'C_Mode']].mean().reset_index()

# Sort by step
avg_df = avg_df.sort_values('Step')

print("Average Silhouette Scores:")
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
ax.set_title('Average Feature Clustering Quality During Training\n(Mean of 12 Binary Classifications)', fontsize=14)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

# Format x-axis
ax.set_xticks(steps)
ax.set_xticklabels([f'{s//1000}k' for s in steps], rotation=45)

plt.tight_layout()
plt.savefig('outputs/silhouette_average_by_step.png', dpi=150, bbox_inches='tight')
print("\nSaved: outputs/silhouette_average_by_step.png")

# Save average CSV
avg_df.to_csv('outputs/silhouette_average_by_step.csv', index=False)
print("Saved: outputs/silhouette_average_by_step.csv")

plt.show()
