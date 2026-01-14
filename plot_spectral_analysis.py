"""
生成特征值和图像投影的对比分析图。
显示 <x, e_k> 和 λ_k(y) 的关系。
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 配置
IMAGE_IDS = [0, 50, 100]
METHODS = ['baseline', 'C', 'CS']
EPOCHS = [10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180]  # 10k-180k
WORK_SIZE = 32

CACHE_DIR = 'analysis_outputs/jacobian_cache'
IMAGE_DIR = '/home/sunj11/Documents/U-ViT-fresh/eval_samples/celeba64_uvit_small/20260101_154543/100000_ema'
OUTPUT_DIR = 'analysis_outputs/spectral_analysis'


def load_image(img_id):
    """加载图片并转为tensor。"""
    img_path = os.path.join(IMAGE_DIR, f'{img_id}.png')
    img = Image.open(img_path).convert('L')
    img = img.resize((WORK_SIZE, WORK_SIZE))
    img = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
    return img.flatten()


def plot_single_spectral(img_id, method, epoch, ax1, title):
    """绘制单个方法/epoch的频谱分析。"""
    img_str = f"img_{img_id:03d}"
    epoch_str = f"{epoch:03d}k"
    cache_path = os.path.join(CACHE_DIR, img_str, method, epoch_str)

    U_path = os.path.join(cache_path, 'U.pt')
    S_path = os.path.join(cache_path, 'S.pt')

    if not os.path.exists(U_path) or not os.path.exists(S_path):
        ax1.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title(title)
        return

    U = torch.load(U_path, weights_only=True)
    S = torch.load(S_path, weights_only=True)
    im = load_image(img_id)

    K = WORK_SIZE
    cors = []
    for j in range(K * K):
        cors.append(torch.dot(U[:, j], im).cpu())
    cors = torch.stack(cors)

    # 绘制
    color = 'tab:blue'
    ax1.plot(abs(cors.cpu()), '*', alpha=0.6, markersize=2, color=color)
    ax1.set_ylabel(r'$\langle x,e_k\rangle$', color=color, fontsize=10)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=8)
    ax1.set_xlabel(r'$k$', fontsize=10)
    ax1.tick_params(axis='x', labelsize=8)

    color = 'tab:red'
    ax2 = ax1.twinx()
    ax2.plot(S.cpu(), '.', alpha=0.6, markersize=2, color=color)
    ax2.set_ylabel(r'$\lambda_k$', color=color, fontsize=10)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=8)

    ax1.set_title(title, fontsize=10)


def generate_comparison_grid(img_id):
    """为单个图片生成方法×epoch的对比网格。"""
    img_str = f"img_{img_id:03d}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(3, 12, figsize=(28, 9))
    fig.suptitle(f'Spectral Analysis - {img_str} (10k-180k)\nBlue: $\\langle x, e_k \\rangle$, Red: $\\lambda_k$', fontsize=14)

    for row, method in enumerate(METHODS):
        for col, epoch in enumerate(EPOCHS):
            ax = axes[row, col]
            title = f'{method} - {epoch}k' if row == 0 else f'{epoch}k'
            if col == 0:
                title = f'{method}\n{epoch}k'
            plot_single_spectral(img_id, method, epoch, ax, title)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f'{img_str}_spectral_grid.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_method_comparison(img_id, epoch):
    """生成单个epoch下三种方法的对比图。"""
    img_str = f"img_{img_id:03d}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Spectral Analysis - {img_str} @ {epoch}k\nBlue: $\\langle x, e_k \\rangle$, Red: $\\lambda_k$', fontsize=14)

    for col, method in enumerate(METHODS):
        plot_single_spectral(img_id, method, epoch, axes[col], method)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f'{img_str}_epoch{epoch}k_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 为每个图片生成网格对比
    for img_id in IMAGE_IDS:
        generate_comparison_grid(img_id)

    # 为关键epoch生成方法对比
    key_epochs = [100, 180]
    for img_id in IMAGE_IDS:
        for epoch in key_epochs:
            generate_method_comparison(img_id, epoch)

    print(f"\nAll done! Output: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
