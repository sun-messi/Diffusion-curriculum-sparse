"""Plot FID comparison for CIFAR-10 U-ViT models."""
import matplotlib.pyplot as plt

# Set global style
plt.rcParams['font.size'] = 32
plt.rcParams['axes.labelsize'] = 32
plt.rcParams['axes.titlesize'] = 36
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28
plt.rcParams['legend.fontsize'] = 22

# Data from README.md
steps = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
baseline = [393.0, 219.4, 362.1, 135.6, 34.7, 21.2, 16.7, 15.1, 14.1, 13.5]
c_mode = [438.7, 215.1, 127.3, 70.6, 32.4, 20.7, 17.5, 16.0, 15.2, 14.2]
cs_mode = [415.3, 323.5, 133.0, 74.7, 27.8, 17.5, 16.3, 15.3, 14.3, 13.5]
s_mode = [380.9, 402.0, 119.3, 35.0, 27.0, 23.0, 20.9, 20.8, 17.7, 15.7]

# Plot (80k-200k range for clearer comparison)
fig, ax = plt.subplots(figsize=(14, 10))

# Filter to 80k+
idx_start = steps.index(80)
plot_steps = steps[idx_start:]
plot_baseline = baseline[idx_start:]
plot_c = c_mode[idx_start:]
plot_cs = cs_mode[idx_start:]
plot_s = s_mode[idx_start:]

ax.plot(plot_steps, plot_baseline, 'b-o', label='Baseline', linewidth=6, markersize=18)
ax.plot(plot_steps, plot_c, 'g-^', label='C Mode', linewidth=6, markersize=18)
ax.plot(plot_steps, plot_cs, 'r-s', label='CS Mode', linewidth=6, markersize=18)
ax.plot(plot_steps, plot_s, 'm-d', label='S Mode', linewidth=6, markersize=18)

ax.set_xlabel('Training Step (k)', labelpad=15)
ax.set_ylabel('FID', labelpad=15)
ax.legend(loc='upper right', handlelength=3)
ax.grid(True, linestyle='--', alpha=0.7)

show_steps = [80, 120, 160, 200]
ax.set_xticks(show_steps)
ax.set_xticklabels(['%dk' % s for s in show_steps])

plt.tight_layout()
plt.savefig('outputs/cifar10_fid.png', dpi=300, bbox_inches='tight')
print('Saved: outputs/cifar10_fid.png')
