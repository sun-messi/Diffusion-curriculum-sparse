"""Plot FID comparison for CIFAR-10 U-ViT models."""
import matplotlib.pyplot as plt

# ============== 核心可调参数 ==============
FIGSIZE = (10, 8)
LINEWIDTH = 6
MARKERSIZE = 18

FONT_SIZE = 28
LABEL_SIZE = 28
TICK_SIZE = 28
LEGEND_SIZE = 28
# =========================================

# 全局样式
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['axes.labelsize'] = LABEL_SIZE
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['xtick.labelsize'] = TICK_SIZE
plt.rcParams['ytick.labelsize'] = TICK_SIZE
plt.rcParams['legend.fontsize'] = LEGEND_SIZE

# Data from README.md
steps = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
baseline = [393.0, 219.4, 362.1, 135.6, 34.7, 21.2, 16.7, 15.1, 14.1, 13.5]
c_mode = [438.7, 215.1, 127.3, 70.6, 32.4, 20.7, 17.5, 16.0, 15.2, 14.2]
cs_mode = [415.3, 323.5, 133.0, 74.7, 27.8, 17.5, 16.3, 15.3, 14.3, 13.5]

# Plot (100k-200k range, no sparsity)
fig, ax = plt.subplots(figsize=FIGSIZE)

# Filter to 100k+
idx_start = steps.index(100)
plot_steps = steps[idx_start:]
plot_baseline = baseline[idx_start:]
plot_c = c_mode[idx_start:]
plot_cs = cs_mode[idx_start:]

ax.plot(plot_steps, plot_baseline, 'b--o', label='Standard training', linewidth=LINEWIDTH, markersize=MARKERSIZE)
ax.plot(plot_steps, plot_c, 'g-^', label='Denoise curriculum', linewidth=LINEWIDTH, markersize=MARKERSIZE)
ax.plot(plot_steps, plot_cs, 'r-s', label='Joint curriculum', linewidth=LINEWIDTH, markersize=MARKERSIZE)

ax.set_xlabel('Training Steps (k)', labelpad=15)
ax.set_ylabel('FID', labelpad=15)
ax.legend(loc='upper right', frameon=True, handlelength=3)
ax.grid(True, linestyle='--', alpha=0.7)
ax.tick_params(axis='both', which='major', length=8, width=3)

show_steps = [100, 120, 140, 160, 180, 200]
ax.set_xticks(show_steps)
ax.set_xticklabels(['%dk' % s for s in show_steps])

plt.tight_layout()
plt.savefig('outputs/cifar10_fid.png', dpi=300, bbox_inches='tight')
print('Saved: outputs/cifar10_fid.png')
