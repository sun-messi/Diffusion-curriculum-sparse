"""Plot convergence speed comparison for CIFAR-10 U-ViT models.
Shows training steps to reach FID thresholds (50, 30, 20).
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
thresholds = [50, 30, 20]  # FID thresholds
threshold_labels = ['FID < 50', 'FID < 30', 'FID < 20']

# Baseline: FID<50 at 100k, FID<30 at 120k (21.2), FID<20 at 140k (16.7)
baseline_steps = [100, 120, 140]

# Denoise curriculum: FID<50 at 100k (32.4), FID<30 at 120k (20.7), FID<20 at 140k (17.5)
c_steps = [100, 120, 140]

# Joint curriculum: FID<50 at 100k (27.8), FID<30 at 100k (27.8), FID<20 at 120k (17.5)
cs_steps = [100, 100, 120]

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
ax.legend(loc='upper right', frameon=True)
ax.grid(True, linestyle='--', alpha=0.5, axis='y')
ax.tick_params(axis='both', which='major', length=8, width=3)

ax.set_ylim(0, 160)

plt.tight_layout()
plt.savefig('outputs/cifar10_convergence_speed.png', dpi=300, bbox_inches='tight')
print('Saved: outputs/cifar10_convergence_speed.png')
