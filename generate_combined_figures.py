"""
为每个图片生成 lambda 组合对比图。
格式：3行(baseline, C, CS) × 7列(60k-180k)
"""

import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 配置
IMAGE_IDS = [1]  # 64×64 原图
METHODS = ['baseline', 'C', 'CS']
EPOCHS = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]  # 20k-180k
LAMBDA_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80]

INPUT_DIR = 'analysis_outputs/jacobian_single_64'
OUTPUT_DIR = 'analysis_outputs/jacobian_combined_64'


def generate_combined_for_image(img_id):
    """为单个图片生成所有 lambda 的组合图。"""
    img_str = f"img_{img_id:03d}"
    output_subdir = os.path.join(OUTPUT_DIR, img_str)
    os.makedirs(output_subdir, exist_ok=True)

    print(f"Generating combined figures for {img_str}...")

    for lambda_idx in LAMBDA_INDICES:
        fig, axes = plt.subplots(3, 9, figsize=(18, 6))
        fig.suptitle(f'λ_{lambda_idx} Eigenvector Evolution (20k-180k) - {img_str}', fontsize=14)

        for row, method in enumerate(METHODS):
            for col, epoch in enumerate(EPOCHS):
                epoch_str = f"{epoch:03d}k"
                img_path = os.path.join(INPUT_DIR, img_str, method, epoch_str, f'lambda_{lambda_idx:02d}.png')

                ax = axes[row, col]

                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)

                ax.axis('off')

                # 列标题（只在第一行）
                if row == 0:
                    ax.set_title(f'{epoch}k', fontsize=10)

                # 行标签（只在第一列）
                if col == 0:
                    ax.set_ylabel(method, fontsize=12, fontweight='bold')
                    ax.yaxis.set_label_position('left')

        # 手动添加行标签
        for row, method in enumerate(METHODS):
            fig.text(0.02, 0.78 - row * 0.27, method, fontsize=12, fontweight='bold',
                    va='center', ha='left')

        plt.tight_layout()
        plt.subplots_adjust(left=0.08, top=0.92)

        output_path = os.path.join(output_subdir, f'lambda_{lambda_idx:02d}_combined.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {output_path}")

    print(f"  {img_str} done: {len(LAMBDA_INDICES)} combined figures")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for img_id in IMAGE_IDS:
        generate_combined_for_image(img_id)

    print(f"\nAll done! Output: {OUTPUT_DIR}/img_XXX/")


if __name__ == '__main__':
    main()
