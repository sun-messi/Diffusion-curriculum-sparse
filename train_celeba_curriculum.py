"""
CelebA DDPM Training with Curriculum Learning (单GPU版本)

只加入 Curriculum Learning，不加入 Sparsity Curriculum
从高噪声到低噪声逐步训练

Usage:
    python train_celeba_curriculum.py
"""
from typing import Optional
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from mindiffusion.unet import NaiveUnet
from mindiffusion.curriculum_ddpm import CurriculumDDPM

from dotenv import load_dotenv

load_dotenv("./.env")
CELEBA_PATH = os.getenv("CELEBA_PATH")


def train_celeba_curriculum(
    device: str = "cuda:0",
    load_pth: Optional[str] = None
) -> None:
    """
    Curriculum Learning 训练 CelebA

    Curriculum 策略：
    - Stage 1: t ∈ [0.8, 1.0] - 高噪声，学习粗糙结构
    - Stage 2: t ∈ [0.6, 1.0] - 扩展范围
    - Stage 3: t ∈ [0.4, 1.0] - 继续扩展
    - Stage 4: t ∈ [0.2, 1.0] - 接近完整
    - Stage 5: t ∈ [0.0, 1.0] - 完整范围，精细化
    """

    # ==================== 配置 ====================
    num_stages = 5
    epochs_per_stage = [5, 5, 15, 15, 50]  # 总共 90 epochs
    # 时间范围：从高噪声逐步扩展到全范围
    curriculum_stages = [
        (0.8, 1.0, "stage1_high_noise"),
        (0.6, 1.0, "stage2_expand"),
        (0.4, 1.0, "stage3_expand"),
        (0.2, 1.0, "stage4_expand"),
        (0.0, 1.0, "stage5_full_range"),
    ]

    # ==================== 模型 ====================
    ddpm = CurriculumDDPM(
        eps_model=NaiveUnet(3, 3, n_feat=128),
        betas=(1e-4, 0.02),
        n_T=1000
    )

    if load_pth is not None:
        ddpm.load_state_dict(torch.load(load_pth))
        print(f"Loaded checkpoint from {load_pth}")

    ddpm.to(device)

    # ==================== 数据 ====================
    tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = ImageFolder(root=CELEBA_PATH, transform=tf)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=20)

    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")

    # ==================== 优化器 ====================
    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-5)

    # ==================== 输出目录 ====================
    os.makedirs("./contents_curriculum", exist_ok=True)
    os.makedirs("./checkpoints_curriculum", exist_ok=True)

    # ==================== Curriculum 训练 ====================
    global_epoch = 0

    for stage_idx, (t_start, t_end, stage_name) in enumerate(curriculum_stages):
        print(f"\n{'='*60}")
        print(f"CURRICULUM STAGE {stage_idx + 1}/{num_stages}")
        print(f"Stage: {stage_name}")
        print(f"Time range: t ∈ [{t_start:.1f}, {t_end:.1f}]")
        print(f"Epochs: {epochs_per_stage[stage_idx]}")
        print(f"{'='*60}")

        # 设置时间范围
        ddpm.set_time_range(t_start, t_end)

        # 训练这个 stage
        for epoch in range(epochs_per_stage[stage_idx]):
            ddpm.train()

            pbar = tqdm(dataloader, desc=f"Stage {stage_idx+1} Epoch {epoch+1}")
            loss_ema = None

            for x, _ in pbar:
                optim.zero_grad()
                x = x.to(device)
                loss = ddpm(x)
                loss.backward()

                if loss_ema is None:
                    loss_ema = loss.item()
                else:
                    loss_ema = 0.9 * loss_ema + 0.1 * loss.item()

                pbar.set_description(f"Stage {stage_idx+1} Epoch {epoch+1} | loss: {loss_ema:.4f}")
                optim.step()

            global_epoch += 1

            # 每个 epoch 生成样本
            ddpm.eval()
            with torch.no_grad():
                xh = ddpm.sample(8, (3, 128, 128), device)
                xset = torch.cat([xh, x[:8]], dim=0)
                grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
                save_image(grid, f"./contents_curriculum/ddpm_celeba_s{stage_idx+1}_e{epoch+1}.png")

            # 保存最新模型
            torch.save(ddpm.state_dict(), "./ddpm_celeba_curriculum.pth")

        # Stage 结束，保存 checkpoint
        ckpt_path = f"./checkpoints_curriculum/ddpm_celeba_stage{stage_idx+1}.pth"
        torch.save(ddpm.state_dict(), ckpt_path)
        print(f"Stage {stage_idx + 1} completed. Checkpoint saved: {ckpt_path}")

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Total epochs: {global_epoch}")
    print(f"Final model: ./ddpm_celeba_curriculum.pth")
    print(f"{'='*60}")


if __name__ == "__main__":
    train_celeba_curriculum()
