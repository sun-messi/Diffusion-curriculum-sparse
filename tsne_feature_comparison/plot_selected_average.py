#!/usr/bin/env python3
"""
Plot average Silhouette Score for selected classifications (similar magnitude to gender).
"""

import pandas as pd
import matplotlib.pyplot as plt

# ============== 高亮参数 (学习自 cifar10_fid_with_samples.py) ==============
IMG_GRID_ROWS = 5
IMG_GRID_COLS = 5
HIGHLIGHT_INDICES = [10, 12, 24]  # 要圈出的图片位置: (2,0), (2,2), (4,4)
HIGHLIGHT_COLOR = [255, 0, 0]     # 红色边框
HIGHLIGHT_WIDTH = 2               # 边框宽度
# =========================================================================

# Read data
df = pd.read_csv('outputs/silhouette_binary_all.csv')

# Select classifications similar to gender magnitude
SELECTED = ['gender', 'eyeglasses', 'smiling', 'heavy_makeup', 'high_cheekbones']

df_selected = df[df['Classification'].isin(SELECTED)]

print(f"Selected classifications: {SELECTED}")
print(f"Number of rows: {len(df_selected)}")

# Average across selected classifications for each step
avg_df = df_selected.groupby('Step')[['Baseline', 'CS_Mode', 'C_Mode']].mean().reset_index()
avg_df = avg_df.sort_values('Step')

print("\nAverage Silhouette Scores (selected):")
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
ax.set_title('Feature Clustering Quality During Training\n(Mean of: gender, eyeglasses, smiling, heavy_makeup, high_cheekbones)', fontsize=13)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

ax.set_xticks(steps)
ax.set_xticklabels([f'{s//1000}k' for s in steps], rotation=45)

plt.tight_layout()
plt.savefig('outputs/silhouette_selected_average.png', dpi=150, bbox_inches='tight')
print("\nSaved: outputs/silhouette_selected_average.png")

avg_df.to_csv('outputs/silhouette_selected_average.csv', index=False)
print("Saved: outputs/silhouette_selected_average.csv")
