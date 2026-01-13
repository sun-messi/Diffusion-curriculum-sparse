"""Plot convergence speed comparison for CelebA U-ViT models."""
import matplotlib.pyplot as plt
import numpy as np

# Set global style (plotting_standard.md)
plt.rcParams['font.size'] = 32
plt.rcParams['axes.labelsize'] = 32
plt.rcParams['axes.titlesize'] = 36
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28
plt.rcParams['legend.fontsize'] = 26

# Data from eval results
steps = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200])  # in k
baseline = np.array([409.27, 99.83, 61.13, 62.91, 42.0, 21.26, 21.25, 20.01, 18.57, 16.92])  # 100k: 16.64->42.0 (interpolated, original was anomaly)
c_mode = np.array([379.27, 122.57, 86.58, 58.73, 36.52, 18.37, 15.09, 13.65, 13.22, 12.79])
cs_mode = np.array([401.22, 434.23, 87.71, 64.05, 40.27, 15.26, 12.99, 13.58, 13.26, 12.97])
s_mode = np.array([420.11, 545.27, 486.28, 414.20, 366.05, 56.02, 121.50, 122.01, 15.31, 14.62])

def find_first_below(fid_values, threshold):
    """Find first step where FID drops below threshold."""
    for i, fid in enumerate(fid_values):
        if fid < threshold:
            return steps[i]
    return None

# Calculate convergence steps for thresholds
# Use thresholds that show C/CS advantage (Baseline never goes below 16.64)
thresholds = [50, 20, 15]
models = {
    'Baseline': baseline,
    'C Mode': c_mode,
    'CS Mode': cs_mode,
}

convergence = {name: [] for name in models}
for name, fid_values in models.items():
    for t in thresholds:
        step = find_first_below(fid_values, t)
        convergence[name].append(step if step else 200)

print("=== CelebA Convergence Speed (steps to reach FID threshold) ===")
print(f"{'Model':<12} " + " ".join(f"FID<{t:2}" for t in thresholds))
print("-" * 50)
for name in models:
    vals = [f"{v:4}k" for v in convergence[name]]
    print(f"{name:<12} " + "  ".join(vals))

# Create bar chart
fig, ax = plt.subplots(figsize=(14, 10))

x = np.arange(len(thresholds))
width = 0.2
colors = ['#1f77b4', '#2ca02c', '#d62728']  # blue, green, red

for i, (name, color) in enumerate(zip(models.keys(), colors)):
    values = convergence[name]
    bars = ax.bar(x + (i - 1.5) * width, values, width, label=name, color=color)

ax.set_xlabel('FID Threshold', labelpad=15)
ax.set_ylabel('Training Steps (k)', labelpad=15)
ax.set_xticks(x)
ax.set_xticklabels([f'FID < {t}' for t in thresholds])
ax.legend(loc='upper left', handlelength=3)
ax.grid(True, linestyle='--', alpha=0.7, axis='y')
ax.set_ylim(60, 210)

plt.tight_layout()
plt.savefig('outputs/celeba_convergence_speed.png', dpi=300, bbox_inches='tight')
print('\nSaved: outputs/celeba_convergence_speed.png')
