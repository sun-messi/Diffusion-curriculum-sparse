"""
批量生成单独的 Jacobian 特征向量图片。
支持缓存机制，避免重复计算 SVD。
支持多图片处理。

输出结构：
- jacobian_cache/img_XXX/{method}/{epoch}/U.pt, S.pt
- jacobian_single/img_XXX/{method}/{epoch}/lambda_XX.png
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, '/home/sunj11/Documents/U-ViT-fresh')
from libs.uvit import UViT


# ========== 配置区 ==========
METHODS = {
    'baseline': 'workdir/celeba64_uvit_small/default_20260101_030900/ckpts',
    'C': 'workdir/celeba64_uvit_small_c/default/ckpts',
    'CS': 'workdir/celeba64_uvit_small_cs/default_20260101_073037/ckpts',
}

EPOCHS = [20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000]  # 测试用
LAMBDA_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80]

# 多图片支持 - 使用生成样本
IMAGE_IDS = [0]  # 使用生成样本 0.png (与32×32版本相同)
IMAGE_DIR = '/home/sunj11/Documents/U-ViT-fresh/eval_samples/celeba64_uvit_small/20260101_154543/100000_ema'
USE_ORIGINAL_CELEBA = False  # 使用生成样本 (png)

CACHE_DIR = 'analysis_outputs/jacobian_cache_64'  # 64×64 版本
OUTPUT_DIR = 'analysis_outputs/jacobian_single_64'

# Model config
MODEL_CONFIG = {
    'img_size': 64,
    'patch_size': 4,
    'embed_dim': 256,
    'depth': 12,
    'num_heads': 8,
    'mlp_ratio': 4,
    'qkv_bias': False,
    'mlp_time_embed': False,
    'num_classes': -1,
}

WORK_SIZE = 64  # 与训练一致，使用 64×64
TIMESTEP = 500

# CelebA 预处理参数 (与 datasets.py 一致)
CELEBA_CX = 89
CELEBA_CY = 121
CELEBA_CROP_SIZE = 128  # 裁剪 128×128 区域


# ========== 工具函数 ==========
def rgb_to_gray(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def gray_to_rgb(x):
    return x.repeat(1, 3, 1, 1)


def load_image(img_id, device):
    """Load and preprocess image by ID (与训练预处理一致)."""
    if USE_ORIGINAL_CELEBA:
        # 原始 CelebA: 000001.jpg, 000002.jpg, ...
        img_path = os.path.join(IMAGE_DIR, f'{img_id:06d}.jpg')
    else:
        # 生成样本: 0.png, 1.png, ...
        img_path = os.path.join(IMAGE_DIR, f'{img_id}.png')

    if not os.path.exists(img_path):
        print(f"Warning: Image {img_path} not found, using random noise")
        return torch.randn(1, 1, WORK_SIZE, WORK_SIZE, device=device)

    img = Image.open(img_path).convert('L')

    if USE_ORIGINAL_CELEBA:
        # 使用与训练相同的 CelebA 预处理
        # Crop: 以 (cy, cx) = (121, 89) 为中心，裁剪 128×128
        x1 = CELEBA_CY - CELEBA_CROP_SIZE // 2  # 57
        x2 = CELEBA_CY + CELEBA_CROP_SIZE // 2  # 185
        y1 = CELEBA_CX - CELEBA_CROP_SIZE // 2  # 25
        y2 = CELEBA_CX + CELEBA_CROP_SIZE // 2  # 153
        img = img.crop((x1, y1, x2, y2))  # PIL crop: (left, upper, right, lower)
        img = img.resize((WORK_SIZE, WORK_SIZE))
    else:
        # 生成样本已经是 64×64
        if img.size != (WORK_SIZE, WORK_SIZE):
            img = img.resize((WORK_SIZE, WORK_SIZE))

    img = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
    img = img.unsqueeze(0).unsqueeze(0).to(device)
    return img


def load_uvit_checkpoint(ckpt_path, config, device):
    """Load U-ViT model from checkpoint."""
    nnet = UViT(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
        qkv_bias=config['qkv_bias'],
        mlp_time_embed=config['mlp_time_embed'],
        num_classes=config['num_classes'],
    )

    ema_path = os.path.join(ckpt_path, 'nnet_ema.pth')
    if os.path.exists(ema_path):
        state_dict = torch.load(ema_path, map_location='cpu', weights_only=True)
        nnet.load_state_dict(state_dict)
    else:
        nnet_path = os.path.join(ckpt_path, 'nnet.pth')
        state_dict = torch.load(nnet_path, map_location='cpu', weights_only=True)
        nnet.load_state_dict(state_dict)

    nnet = nnet.to(device)
    nnet.eval()
    return nnet


def compute_jacobian(img, nnet, timestep, device, work_size=64):
    """Compute Jacobian of denoiser using sequential gradient computation.

    Note: torch.func.jacrev doesn't work with UViT due to pos_embed shape issues under vmap.
    Using optimized sequential computation with retained graph for speed.

    Args:
        img: Input image tensor [1, 1, H, W]
        nnet: UViT model
        timestep: Diffusion timestep
        device: torch device
        work_size: Size for Jacobian computation (64 = full resolution, 32 = faster)
    """
    import torch.nn.functional as F

    N = work_size * work_size
    t = torch.tensor([timestep], device=device, dtype=torch.long)

    # 确保输入是 work_size
    if img.shape[2] != work_size or img.shape[3] != work_size:
        img_work = F.interpolate(img, size=(work_size, work_size), mode='bilinear')
    else:
        img_work = img

    J = torch.zeros(N, N, device=device)

    # Compute with retained graph for slight speedup (reuse forward pass)
    input_img = img_work.clone().detach().requires_grad_(True)

    # 如果 work_size != 64，需要上采样到 64×64 给模型
    if work_size != 64:
        input_64 = F.interpolate(input_img, size=(64, 64), mode='bilinear')
    else:
        input_64 = input_img

    rgb_input = gray_to_rgb(input_64)
    noise_pred = nnet(rgb_input, t)
    gray_output = rgb_to_gray(noise_pred)

    # 如果 work_size != 64，需要下采样输出
    if work_size != 64:
        output_work = F.interpolate(gray_output, size=(work_size, work_size), mode='bilinear')
    else:
        output_work = gray_output

    output_flat = output_work.view(-1)

    for i in tqdm(range(N), desc="Jacobian rows", leave=False):
        grad = torch.autograd.grad(output_flat[i], input_img, retain_graph=(i < N-1))[0]
        J[i, :] = grad.view(-1)

    return J


def cache_exists(cache_path):
    """Check if cache exists."""
    return os.path.exists(os.path.join(cache_path, 'U.pt')) and \
           os.path.exists(os.path.join(cache_path, 'S.pt'))


def load_cache(cache_path):
    """Load cached U and S."""
    U = torch.load(os.path.join(cache_path, 'U.pt'), weights_only=True)
    S = torch.load(os.path.join(cache_path, 'S.pt'), weights_only=True)
    return U, S


def save_cache(cache_path, U, S):
    """Save U and S to cache."""
    os.makedirs(cache_path, exist_ok=True)
    torch.save(U.cpu(), os.path.join(cache_path, 'U.pt'))
    torch.save(S.cpu(), os.path.join(cache_path, 'S.pt'))


def save_eigenvector_image(eigvec, work_size, output_path):
    """Save eigenvector as work_size x work_size image without title."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    eigvec_2d = eigvec.reshape(work_size, work_size)

    # Create figure with exact pixel size matching work_size
    fig, ax = plt.subplots(figsize=(1, 1), dpi=work_size)
    ax.imshow(-eigvec_2d, cmap='RdBu', norm=colors.CenteredNorm())
    ax.axis('off')

    # Remove all margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.savefig(output_path, dpi=work_size, bbox_inches='tight', pad_inches=0)
    plt.close()


def process_single_image(img_id, device):
    """Process a single image across all methods and epochs."""
    img_str = f"img_{img_id:03d}"
    print(f"\n{'='*60}")
    print(f"Processing {img_str}")
    print(f"{'='*60}")

    # Load image
    img = load_image(img_id, device)
    N = WORK_SIZE * WORK_SIZE

    # Count checkpoints
    total_models = 0
    for method, ckpt_base in METHODS.items():
        for epoch in EPOCHS:
            ckpt_path = os.path.join(ckpt_base, f'{epoch}.ckpt')
            if os.path.exists(ckpt_path):
                total_models += 1

    processed = 0
    for method, ckpt_base in METHODS.items():
        print(f"\n  Method: {method}")

        for epoch in EPOCHS:
            ckpt_path = os.path.join(ckpt_base, f'{epoch}.ckpt')
            if not os.path.exists(ckpt_path):
                continue

            epoch_str = f"{epoch // 1000:03d}k"
            cache_path = os.path.join(CACHE_DIR, img_str, method, epoch_str)
            output_base = os.path.join(OUTPUT_DIR, img_str, method, epoch_str)

            # Check/load cache
            if cache_exists(cache_path):
                print(f"    [{epoch_str}] Loading from cache...")
                U, S = load_cache(cache_path)
            else:
                print(f"    [{epoch_str}] Computing Jacobian + SVD...")
                nnet = load_uvit_checkpoint(ckpt_path, MODEL_CONFIG, device)

                J = compute_jacobian(img, nnet, TIMESTEP, device, WORK_SIZE)

                I = torch.eye(N, device=device)
                I_minus_J = I - J
                U, S, _ = torch.linalg.svd(I_minus_J)

                # Save cache
                save_cache(cache_path, U, S)

                # Cleanup
                del nnet, J, I_minus_J
                torch.cuda.empty_cache()

                U = U.cpu()
                S = S.cpu()

            # Generate images
            for lambda_idx in LAMBDA_INDICES:
                if lambda_idx < U.shape[1]:
                    output_path = os.path.join(output_base, f'lambda_{lambda_idx:02d}.png')
                    eigvec = U[:, lambda_idx].numpy()
                    save_eigenvector_image(eigvec, WORK_SIZE, output_path)

            processed += 1

    print(f"\n  {img_str} done: {processed} models processed")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Images to process: {IMAGE_IDS}")
    print(f"Methods: {list(METHODS.keys())}")
    print(f"Epochs: {len(EPOCHS)} checkpoints per method")
    print(f"Lambda indices: {LAMBDA_INDICES}")

    for img_id in IMAGE_IDS:
        process_single_image(img_id, device)

    print(f"\n{'='*60}")
    print(f"All done!")
    print(f"Cache: {CACHE_DIR}/img_XXX/{{method}}/{{epoch}}/")
    print(f"Images: {OUTPUT_DIR}/img_XXX/{{method}}/{{epoch}}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
