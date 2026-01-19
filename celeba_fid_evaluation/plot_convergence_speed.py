"""Plot convergence speed comparison for CelebA U-ViT models.
Shows training steps to reach FID thresholds (50, 20, 15).
"""
import matplotlib.pyplot as plt
import numpy as np

# ============== 核心可调参数 ==============
FIGSIZE = (10, 8)
BAR_WIDTH = 0.25

FONT_SIZE = 28
LABEL_SIZE = 28
TICK_SIZE = 28
LEGEND_SIZE = 24
# =========================================

# 全局样式
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['axes.labelsize'] = LABEL_SIZE
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['xtick.labelsize'] = TICK_SIZE
plt.rcParams['ytick.labelsize'] = TICK_SIZE
plt.rcParams['legend.fontsize'] = LEGEND_SIZE

# Real data: steps (k) to first reach FID threshold
thresholds = [50, 20, 15]  # FID thresholds
threshold_labels = ['FID < 50', 'FID < 20', 'FID < 15']

# Standard training: FID<50 at 100k, FID<20 at 180k, FID<15 never (use 240)
baseline_steps = [100, 180, 240]

# Denoise curriculum: FID<50 at 100k, FID<20 at 120k, FID<15 at 160k
c_steps = [100, 120, 160]

# Joint curriculum: FID<50 at 100k, FID<20 at 120k, FID<15 at 140k
cs_steps = [100, 120, 140]

# Create bar plot
fig, ax = plt.subplots(figsize=FIGSIZE)

x = np.arange(len(thresholds))

bars1 = ax.bar(x - BAR_WIDTH, baseline_steps, BAR_WIDTH, label='Standard training', color='#1f77b4')
bars2 = ax.bar(x, c_steps, BAR_WIDTH, label='Denoise curriculum', color='#2ca02c')
bars3 = ax.bar(x + BAR_WIDTH, cs_steps, BAR_WIDTH, label='Joint curriculum', color='#d62728')

ax.set_xlabel('FID Threshold', labelpad=15)
ax.set_ylabel('Training Steps (k)', labelpad=15)
ax.set_xticks(x)
ax.set_xticklabels(threshold_labels)
ax.legend(loc='upper left', frameon=True)
ax.grid(True, linestyle='--', alpha=0.5, axis='y')
ax.tick_params(axis='both', which='major', length=8, width=3)

ax.set_ylim(0, 280)

plt.tight_layout()
plt.savefig('outputs/celeba_convergence_speed.png', dpi=300, bbox_inches='tight')
print('Saved: outputs/celeba_convergence_speed.png')
