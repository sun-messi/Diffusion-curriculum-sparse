"""Plot TOPIQ-NR-Face score and win rate curves across checkpoints."""

import json
import os
import matplotlib.pyplot as plt

# ============== 核心可调参数 (与 precision_recall 一致) ==============
FIGSIZE = (10, 8)
LINEWIDTH = 10
MARKERSIZE = 20

FONT_SIZE = 28
LABEL_SIZE = 28
TICK_SIZE = 28
LEGEND_SIZE = 24
# =========================================

plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['axes.labelsize'] = LABEL_SIZE
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['xtick.labelsize'] = TICK_SIZE
plt.rcParams['ytick.labelsize'] = TICK_SIZE
plt.rcParams['legend.fontsize'] = LEGEND_SIZE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(SCRIPT_DIR, "outputs/topiq_face_multi_checkpoint.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

with open(RESULTS_FILE) as f:
    data = json.load(f)

results = data["results"]
steps = [int(r['checkpoint'].replace('_ema', '')) // 1000 for r in results]
means_a = [r['a_mean'] for r in results]
means_b = [r['b_mean'] for r in results]
winrate_b = [r['b_wins'] / r['num_paired'] * 100 for r in results]

# --- Score curve ---
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(steps, means_a, 'b--o', label='Standard training',
        linewidth=LINEWIDTH, markersize=MARKERSIZE)
ax.plot(steps, means_b, 'r-s', label='Joint curriculum',
        linewidth=LINEWIDTH, markersize=MARKERSIZE)
ax.set_xlabel('Training Steps (k)', labelpad=15)
ax.set_ylabel('TOPIQ-NR-Face Score', labelpad=15)
ax.legend(loc='best', frameon=True, handlelength=3)
ax.grid(True, linestyle='--', alpha=0.7)
ax.tick_params(axis='both', which='major', length=8, width=3)
ax.set_xticks(steps)
ax.set_xticklabels([str(s) for s in steps])
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "topiq_face_score_curve.png")
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
print(f"Saved: {out}")
plt.close()

# --- Win rate curve ---
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(steps, winrate_b, 'r-s', label='Joint curriculum win %',
        linewidth=LINEWIDTH, markersize=MARKERSIZE)
ax.axhline(50, color='gray', linestyle='--', linewidth=3, alpha=0.7)
ax.set_xlabel('Training Steps (k)', labelpad=15)
ax.set_ylabel('Win Rate (%)', labelpad=15)
ax.legend(loc='best', frameon=True, handlelength=3)
ax.grid(True, linestyle='--', alpha=0.7)
ax.tick_params(axis='both', which='major', length=8, width=3)
ax.set_xticks(steps)
ax.set_xticklabels([str(s) for s in steps])
ax.set_ylim(0, 100)
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "topiq_face_winrate_curve.png")
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
print(f"Saved: {out}")
plt.close()
