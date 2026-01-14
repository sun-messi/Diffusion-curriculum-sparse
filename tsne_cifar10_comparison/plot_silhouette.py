#!/usr/bin/env python3
"""
Plot Silhouette Score comparison for CIFAR-10 experiments.
With sample images at 40k, 60k, 200k below the plot.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from matplotlib.gridspec import GridSpec

# ============== 核心可调参数 ==============
FIGSIZE = (16, 14)            # 图形大小
BG_COLOR = '#E3F2FD'          # 背景颜色 (浅蓝色)
LINEWIDTH = 15                 # 线条粗细
MARKERSIZE = 33               # 标记大小

FONT_SIZE = 28                # 全局字号
LABEL_SIZE = 28               # 坐标轴标签字号
TICK_SIZE = 28                # 刻度字号
LEGEND_SIZE = 24              # 图例字号
TITLE_SIZE = 24               # 子图标题字号
YLABEL_SIZE = 32              # Y轴标签字号

VLINE_WIDTH = 8               # 竖线粗细
VLINE_ALPHA = 0.7             # 竖线透明度

GRID_ALPHA = 0.7              # 网格透明度
TICK_LENGTH = 8               # 刻度线长度
TICK_WIDTH = 3                # 刻度线宽度

# 图片网格参数
IMG_GRID_ROWS = 2             # 每个样本的行数
IMG_GRID_COLS = 4             # 每个样本的列数
IMG_GAP = 2                   # 图片间隙

# 布局参数
GS_TOP_BOTTOM = 0.58         # 上图底部位置
GS_INNER_TOP = 0.50          # 下图顶部位置
GS_INNER_BOTTOM = 0.00      # 下图底部位置
GS_INNER_HSPACE = 0.1        # 图片行间距
GS_INNER_WSPACE = 0.12        # 图片列间距
# =========================================

# 全局样式设置
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['axes.labelsize'] = LABEL_SIZE
plt.rcParams['axes.titlesize'] = TITLE_SIZE
plt.rcParams['xtick.labelsize'] = TICK_SIZE
plt.rcParams['ytick.labelsize'] = TICK_SIZE
plt.rcParams['legend.fontsize'] = LEGEND_SIZE

# 数据路径
base_dirs = {
    'Baseline': '/home/sunj11/Documents/U-ViT-fresh/eval_samples/cifar10_uvit_small/20260111_002937',
    'Joint curriculum': '/home/sunj11/Documents/U-ViT-fresh/eval_samples/cifar10_uvit_small_cs/20260110_094434',
    'Denoise curriculum': '/home/sunj11/Documents/U-ViT-fresh/eval_samples/cifar10_uvit_small_c/20260110_125959'
}
sample_steps = [40000, 60000, 200000]

# 读取数据
df = pd.read_csv('outputs/silhouette_cifar10.csv')
print("Silhouette Scores:")
print(df.to_string(index=False))

def load_sample_grid(checkpoint_dir):
    """Load and create a grid of sample images"""
    if not os.path.exists(checkpoint_dir):
        return None

    selected_indices = [0, 1, 2, 4, 10, 11, 12, 13]
    imgs = []
    for idx in selected_indices:
        img_path = os.path.join(checkpoint_dir, f'{idx}.png')
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                imgs.append(np.array(img))
            except Exception:
                pass

    if not imgs:
        return None

    rows, cols = IMG_GRID_ROWS, IMG_GRID_COLS
    h, w = imgs[0].shape[:2]
    grid_h = rows * h + (rows - 1) * IMG_GAP
    grid_w = cols * w + (cols - 1) * IMG_GAP
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

    for idx, img in enumerate(imgs[:rows*cols]):
        row = idx // cols
        col = idx % cols
        y = row * (h + IMG_GAP)
        x = col * (w + IMG_GAP)
        grid[y:y+h, x:x+w] = img

    return grid

# 创建图形
fig = plt.figure(figsize=FIGSIZE, facecolor=BG_COLOR)

# 上部: Silhouette score 图
gs_top = GridSpec(1, 1, left=0.1, right=0.98, top=0.98, bottom=GS_TOP_BOTTOM)
ax = fig.add_subplot(gs_top[0])

steps = df['Step'].values
baseline = df['Baseline'].values
cs_mode = df['CS_Mode'].values
c_mode = df['C_Mode'].values

ax.plot(steps, baseline, 'b--o', label='Baseline', linewidth=LINEWIDTH, markersize=MARKERSIZE)
ax.plot(steps, cs_mode, 'r-s', label='Joint curriculum', linewidth=LINEWIDTH, markersize=MARKERSIZE)
ax.plot(steps, c_mode, 'g-^', label='Denoise curriculum', linewidth=LINEWIDTH, markersize=MARKERSIZE)

# 竖向虚线
vline_style = {'color': 'black', 'linestyle': '--', 'linewidth': VLINE_WIDTH, 'alpha': VLINE_ALPHA}
ax.axvline(x=60000, **vline_style)
ax.axvline(x=120000, **vline_style)

ax.set_ylabel('Silhouette Score', labelpad=15, fontsize=YLABEL_SIZE)
ax.legend(loc='lower right', frameon=True, handlelength=3)
ax.grid(True, linestyle='--', alpha=GRID_ALPHA)
ax.tick_params(axis='both', which='major', length=TICK_LENGTH, width=TICK_WIDTH)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))

show_steps = [20000, 60000, 100000, 140000, 200000]
ax.set_xticks(show_steps)
ax.set_xticklabels(['%dk' % (s//1000) for s in show_steps])

# 下部: 样本图片
methods = ['Baseline', 'Joint curriculum', 'Denoise curriculum']
method_short = ['Baseline', 'Joint', 'Denoise']
colors = ['#1f77b4', '#d62728', '#2ca02c']

gs_inner = GridSpec(3, 3, left=0.0, right=0.98, top=GS_INNER_TOP, bottom=GS_INNER_BOTTOM,
                    hspace=GS_INNER_HSPACE, wspace=GS_INNER_WSPACE)

for row_idx, method in enumerate(methods):
    for col_idx, step in enumerate(sample_steps):
        ax_img = fig.add_subplot(gs_inner[row_idx, col_idx])
        ax_img.axis('off')

        if col_idx == 0:
            ax_img.text(-0.12, 0.5, method_short[row_idx], transform=ax_img.transAxes,
                       fontsize=TITLE_SIZE, fontweight='bold', va='center', ha='right',
                       rotation=90, color=colors[row_idx])

        if row_idx == 0:
            ax_img.set_title(f'{step//1000}k steps', fontsize=TITLE_SIZE, fontweight='bold', pad=12)

        checkpoint_dir = os.path.join(base_dirs[method], f'{step}_ema')
        grid = load_sample_grid(checkpoint_dir)

        if grid is not None:
            ax_img.imshow(grid, interpolation='bilinear')

plt.savefig('outputs/silhouette_cifar10.png', dpi=300, bbox_inches='tight', facecolor=BG_COLOR)
print("\nSaved: outputs/silhouette_cifar10.png")
