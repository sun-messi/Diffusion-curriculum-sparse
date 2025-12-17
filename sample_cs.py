"""
从训练好的 Curriculum + Sparsity 模型生成图片

Usage:
    python sample_cs.py
    python sample_cs.py --ckpt checkpoints_cs/ddpm_cs_stage10.pth --n_sample 16 --n_batch 4
"""

import argparse
import os

import torch
from torchvision.utils import save_image, make_grid

from mindiffusion.sparse_unet import SparseNaiveUnet
from mindiffusion.curriculum_ddpm import CurriculumDDPM
from config_cs import ConfigCS


def sample(
    ckpt_path: str,
    n_sample: int = 8,
    n_batch: int = 10,
    output_dir: str = "./generated_samples",
    device: str = "cuda:0"
):
    """
    生成多组图片

    Args:
        ckpt_path: 模型 checkpoint 路径
        n_sample: 每组生成的图片数量
        n_batch: 生成多少组
        output_dir: 输出目录
        device: 设备
    """
    cfg = ConfigCS()

    # 创建模型
    sparse_unet = SparseNaiveUnet(
        in_channels=cfg.in_channels,
        out_channels=cfg.in_channels,
        n_feat=cfg.n_feat,
        initial_sparsity=0.0  # 采样时不需要稀疏
    )

    ddpm = CurriculumDDPM(
        eps_model=sparse_unet,
        betas=(cfg.beta1, cfg.beta2),
        n_T=cfg.n_T
    )

    # 加载权重
    state_dict = torch.load(ckpt_path, map_location=device)
    ddpm.load_state_dict(state_dict)
    ddpm.to(device)
    ddpm.eval()

    print(f"Loaded model from {ckpt_path}")
    print(f"Generating {n_batch} batches of {n_sample} images each...")
    print(f"Output directory: {output_dir}")

    # 打印稀疏度信息
    print(f"\nModel sparsity info:")
    sparse_unet.print_sparsity_info()

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成多组图片
    with torch.no_grad():
        for batch_idx in range(n_batch):
            print(f"\nGenerating batch {batch_idx + 1}/{n_batch}...")

            # 生成图片
            samples = ddpm.sample(
                n_sample,
                (cfg.in_channels, cfg.image_size, cfg.image_size),
                device
            )

            # 保存为 grid
            grid = make_grid(samples, normalize=True, value_range=(-1, 1), nrow=4)
            grid_path = f"{output_dir}/samples_batch{batch_idx + 1}.png"
            save_image(grid, grid_path)
            print(f"  Saved grid: {grid_path}")

            # 也保存单独的图片
            for i, img in enumerate(samples):
                img_path = f"{output_dir}/sample_b{batch_idx + 1}_{i + 1}.png"
                save_image(img, img_path, normalize=True, value_range=(-1, 1))

    print(f"\nDone! Generated {n_batch * n_sample} images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from trained CS model")
    parser.add_argument("--ckpt", type=str, default="./ddpm_cifar_cs_latest.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--n_sample", type=int, default=25,
                        help="Number of samples per batch")
    parser.add_argument("--n_batch", type=int, default=10,
                        help="Number of batches to generate")
    parser.add_argument("--output_dir", type=str, default="./generated_samples",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")

    args = parser.parse_args()

    sample(
        ckpt_path=args.ckpt,
        n_sample=args.n_sample,
        n_batch=args.n_batch,
        output_dir=args.output_dir,
        device=args.device
    )
