"""Plot convergence speed comparison for CIFAR-10 U-ViT models.
Shows how quickly each model reaches FID thresholds (30, 20).
"""
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
steps = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200])  # in k
baseline = np.array([393.0, 219.4, 362.1, 135.6, 34.7, 21.2, 16.7, 15.1, 14.1, 13.5])
c_mode = np.array([438.7, 215.1, 127.3, 70.6, 32.4, 20.7, 17.5, 16.0, 15.2, 14.2])
cs_mode = np.array([415.3, 323.5, 133.0, 74.7, 27.8, 17.5, 16.3, 15.3, 14.3, 13.5])
s_mode = np.array([380.9, 402.0, 119.3, 35.0, 27.0, 23.0, 20.9, 20.8, 17.7, 15.7])

def find_first_below(fid_values, threshold):
    """Find first step where FID drops below threshold."""
    for i, fid in enumerate(fid_values):
        if fid < threshold:
            return steps[i]
    return None

# Calculate convergence steps for thresholds
thresholds = [80, 50, 30, 20]
models = {
    'Baseline': baseline,
    'C Mode': c_mode,
    'CS Mode': cs_mode,
    'S Mode': s_mode,
}

convergence = {name: [] for name in models}
for name, fid_values in models.items():
    for t in thresholds:
        step = find_first_below(fid_values, t)
        convergence[name].append(step if step else 220)  # 220 means not reached

print("=== Convergence Speed (steps to reach FID threshold) ===")
print(f"{'Model':<12} " + " ".join(f"FID<{t:2}" for t in thresholds))
print("-" * 50)
for name in models:
    vals = [f"{v:4}k" if v < 220 else " N/A " for v in convergence[name]]
    print(f"{name:<12} " + "  ".join(vals))

# Create bar chart
fig, ax = plt.subplots(figsize=(14, 10))

x = np.arange(len(thresholds))
width = 0.2
colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']  # blue, green, red, purple

for i, (name, color) in enumerate(zip(models.keys(), colors)):
    values = convergence[name]
    bars = ax.bar(x + (i - 1.5) * width, values, width, label=name, color=color)

ax.set_xlabel('FID Threshold', labelpad=15)
ax.set_ylabel('Training Steps (k)', labelpad=15)
ax.set_xticks(x)
ax.set_xticklabels([f'FID < {t}' for t in thresholds])
ax.legend(loc='upper left', handlelength=3)
ax.grid(True, linestyle='--', alpha=0.7, axis='y')
ax.set_ylim(60, 190)

plt.tight_layout()
plt.savefig('outputs/cifar10_convergence_speed.png', dpi=300, bbox_inches='tight')
print('\nSaved: outputs/cifar10_convergence_speed.png')
