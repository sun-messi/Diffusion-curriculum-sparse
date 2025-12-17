"""
CelebA DDPM Training with Curriculum Learning + Sparsity Curriculum - 32x32 版本 (DDP多GPU)

CS = Curriculum + Sparsity
从高噪声到低噪声逐步训练，同时逐步增加模型容量

Usage:
    # 使用所有可用GPU
    torchrun --nproc_per_node=6 train_celeba_cs_32.py

    # 指定GPU数量
    torchrun --nproc_per_node=4 train_celeba_cs_32.py

    # 指定特定GPU
    CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 train_celeba_cs_32.py
"""
from typing import Optional
import os
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from mindiffusion.sparse_unet import SparseNaiveUnet
from mindiffusion.curriculum_ddpm import CurriculumDDPM

from dotenv import load_dotenv

load_dotenv("./.env")
CELEBA_PATH = os.getenv("CELEBA_PATH")


def setup_ddp():
    """初始化DDP环境"""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    return rank, world_size, local_rank, device


def cleanup_ddp():
    """清理DDP环境"""
    dist.destroy_process_group()


def is_main_process(rank):
    """判断是否为主进程"""
    return rank == 0


def train_celeba_cs_32() -> None:
    """
    Curriculum Learning + Sparsity Curriculum 训练 CelebA (32x32) with DDP

    Curriculum 策略：
    - Stage 1: t ∈ [0.8, 1.0], sparsity=80% - 高噪声，少量channels学习粗糙结构
    - Stage 2: t ∈ [0.6, 1.0], sparsity=60% - 扩展范围，regrow channels
    - Stage 3: t ∈ [0.4, 1.0], sparsity=40% - 继续扩展
    - Stage 4: t ∈ [0.2, 1.0], sparsity=20% - 接近完整
    - Stage 5: t ∈ [0.0, 1.0], sparsity=0%  - 完整范围，全部channels
    """

    # ==================== 初始化 DDP ====================
    rank, world_size, local_rank, device = setup_ddp()

    # ==================== 配置 ====================
    num_stages = 5
    epochs_per_stage = [5, 10, 15, 20, 30]  # 总共 80 epochs
    global_batch_size = 256  # 总 batch size
    batch_size = global_batch_size // world_size  # 每个 GPU 的 batch size
    lr = 2e-5

    # 时间范围：从高噪声逐步扩展到全范围
    curriculum_stages = [
        (0.8, 1.0, "stage1_high_noise"),
        (0.6, 1.0, "stage2_expand"),
        (0.4, 1.0, "stage3_expand"),
        (0.2, 1.0, "stage4_expand"),
        (0.0, 1.0, "stage5_full_range"),
    ]

    # 稀疏度 schedule：从高稀疏度逐步降到0
    sparsity_schedule = [0.80, 0.60, 0.40, 0.20, 0.00]
    initial_sparsity = sparsity_schedule[0]  # 0.80

    regrowth_method = "random"  # "gradient" or "random"

    if is_main_process(rank):
        print(f"{'='*60}")
        print("CelebA CS Training (32x32) with DDP")
        print(f"{'='*60}")
        print(f"World size: {world_size} GPUs")
        print(f"Global batch size: {global_batch_size}")
        print(f"Per-GPU batch size: {batch_size}")
        print(f"Learning rate: {lr}")
        print(f"Sparsity schedule: {sparsity_schedule}")
        print(f"Epochs per stage: {epochs_per_stage}")
        print(f"Total epochs: {sum(epochs_per_stage)}")
        print(f"{'='*60}")

    # ==================== 模型 (使用 SparseNaiveUnet) ====================
    sparse_unet = SparseNaiveUnet(
        in_channels=3,
        out_channels=3,
        n_feat=32,
        initial_sparsity=initial_sparsity
    )

    ddpm = CurriculumDDPM(
        eps_model=sparse_unet,
        betas=(1e-4, 0.02),
        n_T=1000
    )

    ddpm.to(device)

    # DDP 包装
    ddpm = DDP(ddpm, device_ids=[local_rank])

    if is_main_process(rank):
        print(f"\nSparseNaiveUnet created:")
        print(f"  Bottleneck channels: {sparse_unet.bottleneck_channels}")
        print(f"  Initial sparsity: {sparse_unet.get_current_sparsity():.1%}")
        print(f"  Active channels: {sparse_unet.get_active_channel_count()}")

    # ==================== 数据 ====================
    tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = ImageFolder(root=CELEBA_PATH, transform=tf)

    # 分布式采样器
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    if is_main_process(rank):
        print(f"\nDataset size: {len(dataset)}")
        print(f"Batches per epoch: {len(dataloader)}")

    # ==================== 优化器 ====================
    optim = torch.optim.Adam(ddpm.parameters(), lr=lr)

    # ==================== 输出目录 ====================
    if is_main_process(rank):
        os.makedirs("./contents_cs_32", exist_ok=True)
        os.makedirs("./checkpoints_cs_32", exist_ok=True)

    # 等待主进程创建目录
    dist.barrier()

    # ==================== Curriculum + Sparsity 训练 ====================
    global_epoch = 0

    for stage_idx, (t_start, t_end, stage_name) in enumerate(curriculum_stages):
        target_sparsity = sparsity_schedule[stage_idx]

        if is_main_process(rank):
            print(f"\n{'='*60}")
            print(f"CURRICULUM STAGE {stage_idx + 1}/{num_stages}")
            print(f"Stage: {stage_name}")
            print(f"Time range: t ∈ [{t_start:.1f}, {t_end:.1f}]")
            print(f"Target sparsity: {target_sparsity:.0%}")
            print(f"Epochs: {epochs_per_stage[stage_idx]}")
            print(f"{'='*60}")

        # 设置时间范围 (访问 DDP 内部模型)
        ddpm.module.set_time_range(t_start, t_end)

        # ========== Regrowth (stage > 0) ==========
        if stage_idx > 0:
            current_sparsity = sparse_unet.get_current_sparsity()
            if target_sparsity < current_sparsity:
                # 计算需要激活多少 channels
                total_channels = sparse_unet.bottleneck_channels
                current_active = int(total_channels * (1 - current_sparsity))
                target_active = int(total_channels * (1 - target_sparsity))
                num_to_grow = target_active - current_active

                if num_to_grow > 0 and is_main_process(rank):
                    print(f"\n    [Regrowth] Growing {num_to_grow} channels")
                    print(f"    {current_sparsity:.1%} -> {target_sparsity:.1%} sparsity")

                if num_to_grow > 0:
                    actually_grown = sparse_unet.regrow_channels(
                        num_to_grow,
                        current_stage=stage_idx,
                        method=regrowth_method
                    )
                    if is_main_process(rank):
                        print(f"    Regrew {actually_grown} channels")
                        print(f"    New sparsity: {sparse_unet.get_current_sparsity():.1%}")

        # 训练这个 stage
        for epoch in range(epochs_per_stage[stage_idx]):
            # 设置 epoch 确保每个 epoch 的 shuffle 不同
            sampler.set_epoch(global_epoch)

            ddpm.train()

            if is_main_process(rank):
                pbar = tqdm(dataloader, desc=f"Stage {stage_idx+1} Epoch {epoch+1}")
            else:
                pbar = dataloader

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

                if is_main_process(rank):
                    pbar.set_description(f"Stage {stage_idx+1} Epoch {epoch+1} | loss: {loss_ema:.4f}")

                optim.step()

            global_epoch += 1

            # 同步所有进程
            dist.barrier()

            # 每个 epoch 生成样本 (只在主进程)
            if is_main_process(rank):
                ddpm.eval()
                with torch.no_grad():
                    xh = ddpm.module.sample(8, (3, 32, 32), device)
                    xset = torch.cat([xh, x[:8]], dim=0)
                    grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
                    save_image(grid, f"./contents_cs_32/ddpm_celeba_s{stage_idx+1}_e{epoch+1}.png")

                # 保存最新模型
                torch.save(ddpm.module.state_dict(), "./ddpm_celeba_cs_32.pth")

            dist.barrier()

        # Stage 结束，保存 checkpoint 并打印稀疏度信息
        if is_main_process(rank):
            ckpt_path = f"./checkpoints_cs_32/ddpm_celeba_stage{stage_idx+1}.pth"
            torch.save(ddpm.module.state_dict(), ckpt_path)

            print(f"\nStage {stage_idx + 1} completed.")
            sparse_unet.print_sparsity_info()
            print(f"Checkpoint saved: {ckpt_path}")

        dist.barrier()

    if is_main_process(rank):
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Total epochs: {global_epoch}")
        print(f"Final model: ./ddpm_celeba_cs_32.pth")
        print("\nFinal sparsity info:")
        sparse_unet.print_sparsity_info()
        print(f"{'='*60}")

    cleanup_ddp()


if __name__ == "__main__":
    train_celeba_cs_32()
