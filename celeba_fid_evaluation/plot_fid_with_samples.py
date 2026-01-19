#!/usr/bin/env python3
"""
Plot CelebA-64 FID comparison with sample images.
Similar layout to CIFAR-10 version.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# ============== 核心可调参数 ==============
# 参考 silhouette_cifar10.png 的比例
FIGSIZE = (16, 24)            # 增加高度给5x4网格，FID图大小不变
BG_COLOR = '#E3F2FD'

LINEWIDTH = 12                # 与silhouette一致
MARKERSIZE = 30               # 与silhouette一致

FONT_SIZE = 28                # 与silhouette一致
LABEL_SIZE = 28
TICK_SIZE = 28
LEGEND_SIZE = 24
TITLE_SIZE = 24

# 图片网格参数 (5x4网格)
IMG_GRID_ROWS = 5             # 每个样本的行数
IMG_GRID_COLS = 4             # 每个样本的列数
IMG_GAP = 1                   # 图片间隙
IMG_SCALE = 1                 # 每张小图放大倍数
HIGHLIGHT_INDICES = [0, 6, 15]  # 要圈出的图片位置: (0,0), (1,2), (3,3)
HIGHLIGHT_COLOR = [255, 0, 0] # 红色边框
HIGHLIGHT_WIDTH = 4           # 边框宽度（加粗）

# 布局参数 (保持FID图绝对大小不变)
GS_TOP_BOTTOM = 0.75          # 上图底部位置
GS_INNER_TOP = 0.70           # 下图顶部位置
GS_INNER_BOTTOM = 0.00        # 下图底部位置
GS_INNER_HSPACE = 0.05         # 图片行间距
GS_INNER_WSPACE = 0.02        # 图片列间距
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

# FID 数据 (from plot_fid.py)
steps_k = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
baseline = [409.27, 99.83, 61.13, 62.91, 42.0, 21.26, 21.25, 20.01, 18.57, 16.92]
c_mode = [379.27, 122.57, 86.58, 58.73, 36.52, 18.37, 15.09, 13.65, 13.22, 12.79]
cs_mode = [401.22, 434.23, 87.71, 64.05, 40.27, 15.26, 12.99, 13.05, 13.00, 12.97]

# FID curve: 60k-200k
curve_steps = [60, 80, 100, 120, 140, 160, 180, 200]
idx_start = steps_k.index(60)
baseline_curve = baseline[idx_start:]
c_curve = c_mode[idx_start:]
cs_curve = cs_mode[idx_start:]

# 样本图片步数
sample_steps = [50000, 100000, 140000]

# 样本图片路径
base_dirs = {
    'Standard': '/home/sunj11/Documents/U-ViT-fresh/eval_samples/celeba64_uvit_small/20260101_154543',
    'Joint': '/home/sunj11/Documents/U-ViT-fresh/eval_samples/celeba64_uvit_small_cs/20260101_161820',
    'Denoise': '/home/sunj11/Documents/U-ViT-fresh/eval_samples/celeba64_uvit_small_c/20260101_160525'
}

# 随机选择图片 (固定种子保证可复现)
np.random.seed(100)
SELECTED_INDICES = sorted(np.random.choice(range(200), IMG_GRID_ROWS * IMG_GRID_COLS, replace=False).tolist())
# 交换位置: (1,3)<->(3,3)
SELECTED_INDICES[7], SELECTED_INDICES[15] = SELECTED_INDICES[15], SELECTED_INDICES[7]
print(f"\n=== Grid Position Mapping ===")
print("Position (row, col) -> Image Index:")
for pos, img_idx in enumerate(SELECTED_INDICES):
    row, col = pos // 4, pos % 4
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
            grid[y:y+bw, x:x+w] = HIGHLIGHT_COLOR
            grid[y+h-bw:y+h, x:x+w] = HIGHLIGHT_COLOR
            grid[y:y+h, x:x+bw] = HIGHLIGHT_COLOR
            grid[y:y+h, x+w-bw:x+w] = HIGHLIGHT_COLOR

    return grid

# 创建图形
fig = plt.figure(figsize=FIGSIZE, facecolor=BG_COLOR)

# 上部: FID 图 (两个并排)
gs_top = GridSpec(1, 2, left=0.1, right=0.98, top=0.98, bottom=GS_TOP_BOTTOM, wspace=0.20)

# 下部: 样本图片 (3行x3列)
gs_inner = GridSpec(3, 3, left=0.08, right=0.98, top=GS_INNER_TOP, bottom=GS_INNER_BOTTOM,
                    hspace=GS_INNER_HSPACE, wspace=GS_INNER_WSPACE)

# 左图: 宏观视图 (60k-200k)
ax_fid_left = fig.add_subplot(gs_top[0])
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

ax_fid_right = fig.add_subplot(gs_top[1])
ax_fid_right.plot(detail_steps, baseline_detail, 'b--o', label='Standard training',
            linewidth=LINEWIDTH, markersize=MARKERSIZE)
ax_fid_right.plot(detail_steps, cs_detail, 'r-s', label='Joint curriculum',
            linewidth=LINEWIDTH, markersize=MARKERSIZE)
ax_fid_right.plot(detail_steps, c_detail, 'g-^', label='Denoise curriculum',
            linewidth=LINEWIDTH, markersize=MARKERSIZE)

ax_fid_right.set_ylabel('FID', labelpad=10, color=DARK_GREY)
ax_fid_right.grid(True, linestyle='--', alpha=0.4, color=GREY)
ax_fid_right.set_xticks([120, 160, 200])
ax_fid_right.set_xticklabels(['120k', '160k', '200k'])
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

        # 加载并显示图片 (只在140k时高亮)
        checkpoint_dir = os.path.join(base_dirs[method], f'{step}_ema')
        grid = load_sample_grid(checkpoint_dir, highlight=(step == 140000))

        if grid is not None:
            ax_img.imshow(grid, interpolation='bilinear')

# 保存
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/celeba_fid_with_samples.png', dpi=300,
            bbox_inches='tight', pad_inches=0.3, facecolor=BG_COLOR)
print("Saved: outputs/celeba_fid_with_samples.png")
