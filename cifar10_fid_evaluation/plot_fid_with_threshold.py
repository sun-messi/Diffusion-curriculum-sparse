"""Plot FID curves with threshold lines to show convergence speed advantage."""
import matplotlib.pyplot as plt
import numpy as np

# Set global style (plotting_standard.md)
plt.rcParams['font.size'] = 32
plt.rcParams['axes.labelsize'] = 32
plt.rcParams['axes.titlesize'] = 36
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28
plt.rcParams['legend.fontsize'] = 26

# Data from README.md
steps = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
baseline = [393.0, 219.4, 362.1, 135.6, 34.7, 21.2, 16.7, 15.1, 14.1, 13.5]
c_mode = [438.7, 215.1, 127.3, 70.6, 32.4, 20.7, 17.5, 16.0, 15.2, 14.2]
cs_mode = [415.3, 323.5, 133.0, 74.7, 27.8, 17.5, 16.3, 15.3, 14.3, 13.5]
s_mode = [380.9, 402.0, 119.3, 35.0, 27.0, 23.0, 20.9, 20.8, 17.7, 15.7]

# Focus on 60k-200k range where convergence happens
fig, ax = plt.subplots(figsize=(14, 10))

idx_start = steps.index(60)
plot_steps = steps[idx_start:]
plot_baseline = baseline[idx_start:]
plot_c = c_mode[idx_start:]
plot_cs = cs_mode[idx_start:]
plot_s = s_mode[idx_start:]

# Plot FID curves
ax.plot(plot_steps, plot_baseline, 'b-o', label='Standard training', linewidth=6, markersize=18)
ax.plot(plot_steps, plot_c, 'g-^', label='C Mode', linewidth=6, markersize=18)
ax.plot(plot_steps, plot_cs, 'r-s', label='CS Mode', linewidth=6, markersize=18)
ax.plot(plot_steps, plot_s, 'm-d', label='S Mode', linewidth=6, markersize=18)

# Add threshold lines
ax.axhline(y=30, color='orange', linestyle='--', linewidth=3, alpha=0.8)
ax.axhline(y=20, color='purple', linestyle='--', linewidth=3, alpha=0.8)
ax.text(202, 30, 'FID=30', fontsize=24, va='center', color='orange', fontweight='bold')
ax.text(202, 20, 'FID=20', fontsize=24, va='center', color='purple', fontweight='bold')

# Add vertical lines showing when CS Mode crosses thresholds
ax.axvline(x=100, color='red', linestyle=':', linewidth=2, alpha=0.5)
ax.axvline(x=120, color='red', linestyle=':', linewidth=2, alpha=0.5)

# Annotations
ax.annotate('CS first\nbelow 30', xy=(100, 27.8), xytext=(85, 50),
            fontsize=20, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.annotate('CS first\nbelow 20', xy=(120, 17.5), xytext=(135, 35),
            fontsize=20, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

ax.set_xlabel('Training Step (k)', labelpad=15)
ax.set_ylabel('FID', labelpad=15)
ax.legend(loc='upper right', handlelength=3)
ax.grid(True, linestyle='--', alpha=0.7)

ax.set_xticks([60, 80, 100, 120, 140, 160, 180, 200])
ax.set_xticklabels(['60k', '80k', '100k', '120k', '140k', '160k', '180k', '200k'])
ax.set_ylim(10, 150)

plt.tight_layout()
plt.savefig('outputs/cifar10_fid_threshold.png', dpi=300, bbox_inches='tight')
print('Saved: outputs/cifar10_fid_threshold.png')
