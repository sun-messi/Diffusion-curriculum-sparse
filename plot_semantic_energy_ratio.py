"""
计算并可视化语义能量占比曲线 (多图片平均)。
使用 lag-1 autocorrelation 自动检测语义边界。

X轴: iteration (60k → 180k)
Y轴: Σ(λ_1...λ_boundary) / Σ(λ_all) (语义能量比例)
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# ========== 配置 ==========
CACHE_BASE = 'analysis_outputs/jacobian_cache'
OUTPUT_PATH = 'analysis_outputs/semantic_boundary_analysis/semantic_energy_ratio_auto.png'

# 所有要处理的图片
IMAGE_IDS = ['img_000', 'img_001', 'img_002', 'img_003', 'img_004', 'img_005', 'img_050', 'img_100']

EPOCHS = [60, 80, 100, 120, 140, 160, 180]  # in k

# 用于检测边界的阈值
# lag-1 autocorrelation < THRESHOLD 表示高频/棋盘格
AUTOCORR_THRESHOLD = 0.0  # 负相关表示棋盘格

# 要检查的 eigenvector 范围
MAX_EIGENVECTOR_CHECK = 20


def load_svd(img_id, method, epoch_k):
    """Load U and S from cache."""
    epoch_str = f"{epoch_k:03d}k"
    cache_path = os.path.join(CACHE_BASE, img_id, method, epoch_str)
    u_path = os.path.join(cache_path, 'U.pt')
    s_path = os.path.join(cache_path, 'S.pt')

    if not os.path.exists(u_path) or not os.path.exists(s_path):
        return None, None

    U = torch.load(u_path, weights_only=True).numpy()
    S = torch.load(s_path, weights_only=True).numpy()
    return U, S


def compute_lag1_autocorr(eigvec, size=32):
    """Compute lag-1 autocorrelation of eigenvector reshaped as 2D image.

    Positive autocorr = smooth/semantic
    Negative autocorr = checkerboard/high-freq
    """
    img = eigvec.reshape(size, size)

    # Horizontal lag-1
    h_corr = np.corrcoef(img[:, :-1].flatten(), img[:, 1:].flatten())[0, 1]
    # Vertical lag-1
    v_corr = np.corrcoef(img[:-1, :].flatten(), img[1:, :].flatten())[0, 1]

    return (h_corr + v_corr) / 2


def find_semantic_boundary(U, threshold=AUTOCORR_THRESHOLD, max_check=MAX_EIGENVECTOR_CHECK):
    """Find the boundary where eigenvectors transition from semantic to high-freq.

    Returns the index of the last semantic eigenvector (1-indexed).
    """
    size = int(np.sqrt(U.shape[0]))

    for i in range(min(max_check, U.shape[1])):
        eigvec = U[:, i]
        autocorr = compute_lag1_autocorr(eigvec, size)

        if autocorr < threshold:
            # This eigenvector is high-freq, boundary is at i (0-indexed)
            # Return i as the count of semantic eigenvectors
            return i

    # All checked eigenvectors are semantic
    return max_check


def compute_energy_ratio(S, boundary):
    """Compute semantic energy ratio: sum(S[:boundary]) / sum(S)."""
    if boundary == 0:
        return 0.0
    semantic_energy = np.sum(S[:boundary])
    total_energy = np.sum(S)
    return semantic_energy / total_energy


def main():
    plt.figure(figsize=(10, 6))

    colors = {
        'baseline': 'blue',
        'CS': 'red',
    }

    # 收集边界统计信息
    boundary_stats = {method: {epoch: [] for epoch in EPOCHS} for method in ['baseline', 'CS']}

    for method in ['baseline', 'CS']:
        epochs_plot = []
        mean_ratios = []
        std_ratios = []

        for epoch_k in EPOCHS:
            ratios_this_epoch = []
            boundaries_this_epoch = []

            for img_id in IMAGE_IDS:
                U, S = load_svd(img_id, method, epoch_k)
                if U is None:
                    continue

                # 自动检测边界
                boundary = find_semantic_boundary(U)
                boundaries_this_epoch.append(boundary)
                boundary_stats[method][epoch_k].append(boundary)

                ratio = compute_energy_ratio(S, boundary)
                ratios_this_epoch.append(ratio)

            if len(ratios_this_epoch) > 0:
                epochs_plot.append(epoch_k)
                mean_ratios.append(np.mean(ratios_this_epoch))
                std_ratios.append(np.std(ratios_this_epoch))

                mean_boundary = np.mean(boundaries_this_epoch)
                std_boundary = np.std(boundaries_this_epoch)
                print(f"{method} @ {epoch_k}k: boundary={mean_boundary:.1f}±{std_boundary:.1f}, "
                      f"ratio={np.mean(ratios_this_epoch):.4f}±{np.std(ratios_this_epoch):.4f}, n={len(ratios_this_epoch)}")

        # 绘制带误差棒的曲线
        plt.errorbar(epochs_plot, mean_ratios, yerr=std_ratios,
                     fmt='o-', color=colors[method], label=method,
                     linewidth=2, markersize=8, capsize=4)

    plt.xlabel('Iteration (k)', fontsize=12)
    plt.ylabel('Semantic Energy Ratio', fontsize=12)
    plt.title(f'Semantic Energy Ratio (Auto-detected boundary, {len(IMAGE_IDS)} images)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(EPOCHS)
    plt.ylim(0, None)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved to: {OUTPUT_PATH}")

    # 打印边界统计
    print("\n=== Boundary Statistics ===")
    for method in ['baseline', 'CS']:
        print(f"\n{method}:")
        for epoch_k in EPOCHS:
            boundaries = boundary_stats[method][epoch_k]
            if boundaries:
                print(f"  {epoch_k}k: {np.mean(boundaries):.1f} ± {np.std(boundaries):.1f} (range: {min(boundaries)}-{max(boundaries)})")


if __name__ == '__main__':
    main()
