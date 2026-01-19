#!/usr/bin/env python3
"""
Plot CIFAR-10 FID comparison with sample images.
Similar to CelebA64 uvit_final_three_way_comparison.png
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# ============== 核心可调参数 ==============
FIGSIZE = (24, 24)
BG_COLOR = '#E3F2FD'

LINEWIDTH = 12
MARKERSIZE = 25

FONT_SIZE = 30
LABEL_SIZE = 30
TICK_SIZE = 30
LEGEND_SIZE = 28
TITLE_SIZE = 28

# 图片网格参数
IMG_GRID_ROWS = 5             # 每个样本的行数 (5x4网格)
IMG_GRID_COLS = 5             # 每个样本的列数
IMG_GAP = 2                   # 图片间隙
IMG_SCALE = 1                 # 每张小图放大倍数
HIGHLIGHT_INDICES = [1, 16, 19]  # 要圈出的图片位置: img228, img2464, img2653
HIGHLIGHT_COLOR = [255, 0, 0]     # 红色边框
HIGHLIGHT_WIDTH = 2               # 边框宽度
# =========================================

# 全局样式
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['axes.labelsize'] = LABEL_SIZE
plt.rcParams['axes.titlesize'] = TITLE_SIZE
plt.rcParams['xtick.labelsize'] = TICK_SIZE
plt.rcParams['ytick.labelsize'] = TICK_SIZE
plt.rcParams['legend.fontsize'] = LEGEND_SIZE

# 颜色
DARK_GREY = '#1F2937'
GREY = '#9CA3AF'

# FID 数据
steps_k = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
baseline = [393.0, 219.4, 362.1, 135.6, 34.7, 21.2, 16.7, 15.1, 14.1, 13.5]
c_mode = [438.7, 215.1, 127.3, 70.6, 32.4, 20.7, 17.5, 16.0, 15.2, 14.2]
cs_mode = [415.3, 323.5, 133.0, 74.7, 27.8, 17.5, 16.3, 15.3, 14.3, 13.5]

# FID curve: 60k-200k
curve_steps = [60, 80, 100, 120, 140, 160, 180, 200]
idx_start = steps_k.index(60)
baseline_curve = baseline[idx_start:]
c_curve = c_mode[idx_start:]
cs_curve = cs_mode[idx_start:]

# 样本图片步数
sample_steps = [60000, 80000, 100000, 200000]

# 样本图片路径
base_dirs = {
    'Standard': '/home/sunj11/Documents/U-ViT-fresh/eval_samples/cifar10_uvit_small/20260111_002937',
    'Joint': '/home/sunj11/Documents/U-ViT-fresh/eval_samples/cifar10_uvit_small_cs/20260110_094434',
    'Denoise': '/home/sunj11/Documents/U-ViT-fresh/eval_samples/cifar10_uvit_small_c/20260110_125959'
}

# 随机选择25个图片 (固定种子保证可复现, 5x5网格)
np.random.seed(42)
SELECTED_INDICES = sorted(np.random.choice(range(5000), IMG_GRID_ROWS * IMG_GRID_COLS, replace=False).tolist())
print(f"\n=== Grid Position Mapping ===")
print("Position (row, col) -> Image Index:")
for pos, img_idx in enumerate(SELECTED_INDICES):
    row, col = pos // IMG_GRID_COLS, pos % IMG_GRID_COLS
    print(f"  [{pos:2d}] ({row},{col}) -> image {img_idx}")
print("=" * 30)

def load_sample_grid(checkpoint_dir, highlight=False):
    """Load and create a grid of sample images"""
    if not os.path.exists(checkpoint_dir):
        return None

    selected_indices = SELECTED_INDICES  # 使用随机选择的25个索引
    imgs = []
    for idx in selected_indices:
        img_path = os.path.join(checkpoint_dir, f'{idx}.png')
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                if IMG_SCALE != 1:
                    new_size = (img.width * IMG_SCALE, img.height * IMG_SCALE)
                    img = img.resize(new_size, Image.LANCZOS)
                imgs.append(np.array(img))
            except Exception:
                pass

    if not imgs:
        return None

    rows, cols = IMG_GRID_ROWS, IMG_GRID_COLS
    gap = IMG_GAP
    h, w = imgs[0].shape[:2]
    grid_h = rows * h + (rows - 1) * gap
    grid_w = cols * w + (cols - 1) * gap
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

    for idx, img in enumerate(imgs[:rows*cols]):
        row = idx // cols
        col = idx % cols
        y = row * (h + gap)
        x = col * (w + gap)
        grid[y:y+h, x:x+w] = img

        # 在高亮索引位置画红色边框 (只在200k时)
        if highlight and idx in HIGHLIGHT_INDICES:
            bw = HIGHLIGHT_WIDTH
            # 上边框
            grid[y:y+bw, x:x+w] = HIGHLIGHT_COLOR
            # 下边框
            grid[y+h-bw:y+h, x:x+w] = HIGHLIGHT_COLOR
            # 左边框
            grid[y:y+h, x:x+bw] = HIGHLIGHT_COLOR
            # 右边框
            grid[y:y+h, x+w-bw:x+w] = HIGHLIGHT_COLOR

    return grid

# 创建图形
fig = plt.figure(figsize=FIGSIZE, facecolor=BG_COLOR)

# 外层 GridSpec: FID图 和 图片区域
gs_outer = GridSpec(2, 1, height_ratios=[0.8, 2.0], hspace=0.10)

# 上部分成两个子图: 左边宏观(80k-200k), 右边细节(100k-200k)
gs_fid = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[0], wspace=0.15)

# 内层 GridSpec: 3行(方法) x 3列(步数)
gs_inner = GridSpecFromSubplotSpec(3, 4, subplot_spec=gs_outer[1], hspace=0.08, wspace=0.08)

# 左图: 宏观视图 (80k-200k)
ax_fid_left = fig.add_subplot(gs_fid[0])
ax_fid_left.plot(curve_steps, baseline_curve, 'b--o', label='Standard training',
            linewidth=LINEWIDTH, markersize=MARKERSIZE)
ax_fid_left.plot(curve_steps, cs_curve, 'r-s', label='Joint curriculum',
            linewidth=LINEWIDTH, markersize=MARKERSIZE)
ax_fid_left.plot(curve_steps, c_curve, 'g-^', label='Denoise curriculum',
            linewidth=LINEWIDTH, markersize=MARKERSIZE)

ax_fid_left.set_ylabel('FID', labelpad=10, color=DARK_GREY)
ax_fid_left.grid(True, linestyle='--', alpha=0.4, color=GREY)
ax_fid_left.set_xticks([60, 100, 140, 180])
ax_fid_left.set_xticklabels(['60k', '100k', '140k', '180k'])
ax_fid_left.set_xlim([55, 205])
ax_fid_left.legend(loc='upper right', framealpha=0.9, edgecolor=GREY)
ax_fid_left.spines['top'].set_visible(False)
ax_fid_left.spines['right'].set_visible(False)

# 右图: 细节视图 (120k-200k)
detail_steps = [120, 140, 160, 180, 200]
idx_detail = steps_k.index(120)
baseline_detail = baseline[idx_detail:]
cs_detail = cs_mode[idx_detail:]
c_detail = c_mode[idx_detail:]

ax_fid_right = fig.add_subplot(gs_fid[1])
ax_fid_right.plot(detail_steps, baseline_detail, 'b--o', label='Standard training',
            linewidth=LINEWIDTH, markersize=MARKERSIZE)
ax_fid_right.plot(detail_steps, cs_detail, 'r-s', label='Joint curriculum',
            linewidth=LINEWIDTH, markersize=MARKERSIZE)
ax_fid_right.plot(detail_steps, c_detail, 'g-^', label='Denoise curriculum',
            linewidth=LINEWIDTH, markersize=MARKERSIZE)

ax_fid_right.set_ylabel('FID', labelpad=10, color=DARK_GREY)
ax_fid_right.grid(True, linestyle='--', alpha=0.4, color=GREY)
ax_fid_right.set_xticks([120, 140, 160, 180, 200])
ax_fid_right.set_xticklabels(['120k', '140k', '160k', '180k', '200k'])
ax_fid_right.set_xlim([115, 205])
ax_fid_right.spines['top'].set_visible(False)
ax_fid_right.spines['right'].set_visible(False)

# 下部: 样本图片
methods = ['Standard', 'Joint', 'Denoise']
colors = ['#2563EB', '#DC2626', '#2ca02c']  # Blue, Red, Green

for row_idx, method in enumerate(methods):
    for col_idx, step in enumerate(sample_steps):
        ax_img = fig.add_subplot(gs_inner[row_idx, col_idx])
        ax_img.axis('off')

        # 行标签
        if col_idx == 0:
            ax_img.text(-0.15, 0.5, method, transform=ax_img.transAxes,
                       fontsize=TITLE_SIZE, fontweight='bold', va='center', ha='right',
                       rotation=90, color=colors[row_idx])

        # 列标签
        if row_idx == 0:
            ax_img.set_title(f'{step//1000}k steps', fontsize=TITLE_SIZE, fontweight='bold',
                           pad=8, color=DARK_GREY)

        # 加载并显示图片 (只在200k时高亮)
        checkpoint_dir = os.path.join(base_dirs[method], f'{step}_ema')
        grid = load_sample_grid(checkpoint_dir, highlight=(step == 200000))

        if grid is not None:
            ax_img.imshow(grid, interpolation='bilinear')

plt.subplots_adjust(left=0.12, right=0.96, top=0.95, bottom=0.02)

# 保存
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/cifar10_fid_with_samples.png', dpi=300,
            bbox_inches='tight', pad_inches=0.3, facecolor=BG_COLOR)
print("Saved: outputs/cifar10_fid_with_samples.png")
