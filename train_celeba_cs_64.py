"""
CelebA DDPM Training with Curriculum Learning + Sparsity Curriculum - 64x64 版本 (单GPU)

CS = Curriculum + Sparsity
从高噪声到低噪声逐步训练，同时逐步增加模型容量

Usage:
    python train_celeba_cs_64.py
"""
from typing import Optional
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from mindiffusion.sparse_unet import SparseNaiveUnet
from mindiffusion.curriculum_ddpm import CurriculumDDPM

from dotenv import load_dotenv

load_dotenv("./.env")
CELEBA_PATH = os.getenv("CELEBA_PATH")


def train_celeba_cs_64(
    device: str = "cuda:3",
    load_pth: Optional[str] = None
) -> None:
    """
    Curriculum Learning + Sparsity Curriculum 训练 CelebA (64x64)

    Curriculum 策略：
    - Stage 1: t ∈ [0.8, 1.0], sparsity=80% - 高噪声，少量channels学习粗糙结构
    - Stage 2: t ∈ [0.6, 1.0], sparsity=60% - 扩展范围，regrow channels
    - Stage 3: t ∈ [0.4, 1.0], sparsity=40% - 继续扩展
    - Stage 4: t ∈ [0.2, 1.0], sparsity=20% - 接近完整
    - Stage 5: t ∈ [0.0, 1.0], sparsity=0%  - 完整范围，全部channels
    """

    # ==================== 配置 ====================
    num_stages = 5
    epochs_per_stage = [3, 5, 5, 5, 22]  # 总共 90 epochs

    # 时间范围：从高噪声逐步扩展到全范围
    curriculum_stages = [
        (0.8, 1.0, "stage1_high_noise"),
        (0.6, 1.0, "stage2_expand"),
        (0.4, 1.0, "stage3_expand"),
        (0.2, 1.0, "stage4_expand"),
        (0.0, 1.0, "stage5_full_range"),
    ]

    # 稀疏度 schedule：从高稀疏度逐步降到0
    # [0.80, 0.60, 0.40, 0.20, 0.00] - 最后一个 stage 直接到 0%
    sparsity_schedule = [0.80, 0.60, 0.40, 0.20, 0.00]
    initial_sparsity = sparsity_schedule[0]  # 0.80

    regrowth_method = "random"  # "gradient" or "random"

    # ==================== 模型 (使用 SparseNaiveUnet) ====================
    sparse_unet = SparseNaiveUnet(
        in_channels=3,
        out_channels=3,
        n_feat=64,
        initial_sparsity=initial_sparsity
    )

    ddpm = CurriculumDDPM(
        eps_model=sparse_unet,
        betas=(1e-4, 0.02),
        n_T=1000
    )

    if load_pth is not None:
        ddpm.load_state_dict(torch.load(load_pth))
        print(f"Loaded checkpoint from {load_pth}")

    ddpm.to(device)

    print(f"Created SparseNaiveUnet:")
    print(f"  Bottleneck channels: {sparse_unet.bottleneck_channels}")
    print(f"  Initial sparsity: {sparse_unet.get_current_sparsity():.1%}")
    print(f"  Active channels: {sparse_unet.get_active_channel_count()}")

    # ==================== 数据 ====================
    tf = transforms.Compose([
        transforms.Resize((64, 64)),
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
    os.makedirs("./contents_cs_64", exist_ok=True)
    os.makedirs("./checkpoints_cs_64", exist_ok=True)

    # ==================== Curriculum + Sparsity 训练 ====================
    global_epoch = 0

    for stage_idx, (t_start, t_end, stage_name) in enumerate(curriculum_stages):
        target_sparsity = sparsity_schedule[stage_idx]

        print(f"\n{'='*60}")
        print(f"CURRICULUM STAGE {stage_idx + 1}/{num_stages}")
        print(f"Stage: {stage_name}")
        print(f"Time range: t ∈ [{t_start:.1f}, {t_end:.1f}]")
        print(f"Target sparsity: {target_sparsity:.0%}")
        print(f"Epochs: {epochs_per_stage[stage_idx]}")
        print(f"{'='*60}")

        # 设置时间范围
        ddpm.set_time_range(t_start, t_end)

        # ========== Regrowth (stage > 0) ==========
        if stage_idx > 0:
            current_sparsity = sparse_unet.get_current_sparsity()
            if target_sparsity < current_sparsity:
                # 计算需要激活多少 channels
                total_channels = sparse_unet.bottleneck_channels
                current_active = int(total_channels * (1 - current_sparsity))
                target_active = int(total_channels * (1 - target_sparsity))
                num_to_grow = target_active - current_active

                if num_to_grow > 0:
                    print(f"\n    [Regrowth] Growing {num_to_grow} channels")
                    print(f"    {current_sparsity:.1%} -> {target_sparsity:.1%} sparsity")

                    actually_grown = sparse_unet.regrow_channels(
                        num_to_grow,
                        current_stage=stage_idx,
                        method=regrowth_method
                    )
                    print(f"    Regrew {actually_grown} channels")
                    print(f"    New sparsity: {sparse_unet.get_current_sparsity():.1%}")

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

                # 累积梯度用于 regrowth
                sparse_unet.accumulate_gradients()

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
                xh = ddpm.sample(8, (3, 64, 64), device)
                xset = torch.cat([xh, x[:8]], dim=0)
                grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
                save_image(grid, f"./contents_cs_64/ddpm_celeba_s{stage_idx+1}_e{epoch+1}.png")

            # 保存最新模型
            torch.save(ddpm.state_dict(), "./ddpm_celeba_cs_64.pth")

        # Stage 结束，保存 checkpoint 并打印稀疏度信息
        ckpt_path = f"./checkpoints_cs_64/ddpm_celeba_stage{stage_idx+1}.pth"
        torch.save(ddpm.state_dict(), ckpt_path)

        print(f"\nStage {stage_idx + 1} completed.")
        sparse_unet.print_sparsity_info()
        print(f"Checkpoint saved: {ckpt_path}")

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Total epochs: {global_epoch}")
    print(f"Final model: ./ddpm_celeba_cs_64.pth")
    print("\nFinal sparsity info:")
    sparse_unet.print_sparsity_info()
    print(f"{'='*60}")


if __name__ == "__main__":
    train_celeba_cs_64()
