"""Plot FID comparison for CelebA U-ViT models (40k-100k range)."""
import matplotlib.pyplot as plt

# ============== 核心可调参数 ==============
FIGSIZE = (10, 8)
LINEWIDTH = 10
MARKERSIZE = 20

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

# Data from eval results
steps = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
baseline = [409.27, 99.83, 61.13, 62.91, 42.0, 21.26, 21.25, 20.01, 18.57, 16.92]
c_mode = [379.27, 122.57, 86.58, 58.73, 36.52, 18.37, 15.09, 13.65, 13.22, 12.79]
cs_mode = [401.22, 434.23, 87.71, 64.05, 40.27, 15.26, 12.99, 13.58, 13.26, 12.97]

# Plot (60k-100k range)
fig, ax = plt.subplots(figsize=FIGSIZE)

# Filter to 60k-100k
idx_start = steps.index(60)
idx_end = steps.index(100) + 1
plot_steps = steps[idx_start:idx_end]
plot_baseline = baseline[idx_start:idx_end]
plot_c = c_mode[idx_start:idx_end]
plot_cs = cs_mode[idx_start:idx_end]

ax.plot(plot_steps, plot_baseline, 'b--o', label='Standard training', linewidth=LINEWIDTH, markersize=MARKERSIZE)
ax.plot(plot_steps, plot_c, 'g-^', label='Denoise curriculum', linewidth=LINEWIDTH, markersize=MARKERSIZE)
ax.plot(plot_steps, plot_cs, 'r-s', label='Joint curriculum', linewidth=LINEWIDTH, markersize=MARKERSIZE)

ax.set_xlabel('Training Steps (k)', labelpad=15)
ax.set_ylabel('FID', labelpad=15)
ax.legend(loc='upper right', frameon=True, handlelength=3)
ax.grid(True, linestyle='--', alpha=0.7)
ax.tick_params(axis='both', which='major', length=8, width=3)

show_steps = [60, 80, 100]
ax.set_xticks(show_steps)
ax.set_xticklabels(['%d' % s for s in show_steps])

plt.tight_layout()
plt.savefig('outputs/celeba_fid_early.png', dpi=300, bbox_inches='tight')
print('Saved: outputs/celeba_fid_early.png')
