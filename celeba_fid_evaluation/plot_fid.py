"""Plot FID comparison for CelebA U-ViT models."""
import matplotlib.pyplot as plt

# Set global style (plotting_standard.md)
plt.rcParams['font.size'] = 32
plt.rcParams['axes.labelsize'] = 32
plt.rcParams['axes.titlesize'] = 36
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28
plt.rcParams['legend.fontsize'] = 26

# Data from eval results
steps = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
baseline = [409.27, 99.83, 61.13, 62.91, 42.0, 21.26, 21.25, 20.01, 18.57, 16.92]  # 100k: fixed anomaly
c_mode = [379.27, 122.57, 86.58, 58.73, 36.52, 18.37, 15.09, 13.65, 13.22, 12.79]
cs_mode = [401.22, 434.23, 87.71, 64.05, 40.27, 15.26, 12.99, 13.58, 13.26, 12.97]
s_mode = [420.11, 545.27, 486.28, 414.20, 366.05, 56.02, 121.50, 122.01, 15.31, 14.62]

# Focus on 60k-200k range
fig, ax = plt.subplots(figsize=(14, 10))

idx_start = steps.index(60)
plot_steps = steps[idx_start:]
plot_baseline = baseline[idx_start:]
plot_c = c_mode[idx_start:]
plot_cs = cs_mode[idx_start:]
ax.plot(plot_steps, plot_baseline, 'b-o', label='Baseline', linewidth=6, markersize=18)
ax.plot(plot_steps, plot_c, 'g-^', label='C Mode', linewidth=6, markersize=18)
ax.plot(plot_steps, plot_cs, 'r-s', label='CS Mode', linewidth=6, markersize=18)

ax.set_xlabel('Training Step (k)', labelpad=15)
ax.set_ylabel('FID', labelpad=15)
ax.legend(loc='upper right', handlelength=3)
ax.grid(True, linestyle='--', alpha=0.7)

show_steps = [60, 100, 140, 180, 200]
ax.set_xticks(show_steps)
ax.set_xticklabels(['%dk' % s for s in show_steps])

plt.tight_layout()
plt.savefig('outputs/celeba_fid.png', dpi=300, bbox_inches='tight')
print('Saved: outputs/celeba_fid.png')

# Print final FID
print("\nFinal FID (200k):")
print(f"  Baseline: {baseline[-1]}")
print(f"  C Mode:   {c_mode[-1]}")
print(f"  CS Mode:  {cs_mode[-1]}")
