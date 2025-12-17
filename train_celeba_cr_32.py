"""
CelebA DDPM Training with Curriculum Learning + Group L1 Regularization - 32x32 版本 (DDP多GPU)

CR = Curriculum + Regularization
从高噪声到低噪声逐步训练，同时使用 Group L1 正则化驱动软稀疏

Group L1 正则化：
- 对所有 Conv 层的 output channels 施加 L2 范数约束
- λ 随 stage 递减 (cosine schedule)，从 lambda_max 递减到 0
- 不使用硬掩码，通过正则化惩罚驱动稀疏

Usage:
    # 使用所有可用GPU
    torchrun --nproc_per_node=6 train_celeba_cr_32.py

    # 指定GPU数量
    torchrun --nproc_per_node=4 train_celeba_cr_32.py

    # 指定特定GPU
    CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 train_celeba_cr_32.py
    CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29502 train_celeba_cr_32.py

"""
from typing import Optional
import os
import math
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from mindiffusion.reg_unet import RegNaiveUnet  # 使用带正则化的 UNet
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


def generate_cosine_lambda_schedule(lambda_max: float, num_stages: int) -> list:
    """
    生成 cosine λ schedule

    λ(i) = lambda_max * 0.5 * (1 + cos(π * i / (num_stages - 1)))

    Args:
        lambda_max: 最大 λ 值
        num_stages: 阶段数

    Returns:
        list of λ values, 从 lambda_max 递减到 0
    """
    if num_stages == 1:
        return [0.0]

    schedule = []
    for i in range(num_stages):
        lambda_val = lambda_max * 0.5 * (1 + math.cos(math.pi * i / (num_stages - 1)))
        schedule.append(lambda_val)
    return schedule


def train_celeba_cr_32() -> None:
    """
    Curriculum Learning + Group L1 Regularization 训练 CelebA (32x32) with DDP

    Curriculum 策略：
    - Stage 1: t ∈ [0.8, 1.0], λ = lambda_max - 高噪声，强正则化
    - Stage 2: t ∈ [0.6, 1.0], λ 递减
    - Stage 3: t ∈ [0.4, 1.0], λ 递减
    - Stage 4: t ∈ [0.2, 1.0], λ 递减
    - Stage 5: t ∈ [0.0, 1.0], λ ≈ 0 - 完整范围，无正则化
    """

    # ==================== 初始化 DDP ====================
    rank, world_size, local_rank, device = setup_ddp()

    # ==================== 配置 ====================
    num_stages = 5
    epochs_per_stage = [5, 10, 15, 20, 30]  # 总共 80 epochs
    global_batch_size = 256  # 总 batch size
    batch_size = global_batch_size // world_size  # 每个 GPU 的 batch size
    lr = 2e-5

    # Group L1 正则化配置
    lambda_max = 0.00005  # 最大正则化系数
    lambda_schedule = generate_cosine_lambda_schedule(lambda_max, num_stages)
    reg_threshold = 0.01  # 观测用阈值

    # 时间范围：从高噪声逐步扩展到全范围
    curriculum_stages = [
        (0.8, 1.0, "stage1_high_noise"),
        (0.6, 1.0, "stage2_expand"),
        (0.4, 1.0, "stage3_expand"),
        (0.2, 1.0, "stage4_expand"),
        (0.0, 1.0, "stage5_full_range"),
    ]

    if is_main_process(rank):
        print(f"{'='*60}")
        print("CelebA Curriculum + Regularization Training (32x32) with DDP")
        print("Group L1 Regularization for Soft Sparsity")
        print(f"{'='*60}")
        print(f"World size: {world_size} GPUs")
        print(f"Global batch size: {global_batch_size}")
        print(f"Per-GPU batch size: {batch_size}")
        print(f"Learning rate: {lr}")
        print(f"Epochs per stage: {epochs_per_stage}")
        print(f"Total epochs: {sum(epochs_per_stage)}")
        print(f"Lambda max: {lambda_max}")
        print(f"Lambda schedule (cosine): {[f'{l:.5f}' for l in lambda_schedule]}")
        print(f"{'='*60}")

    # ==================== 模型 (使用 RegNaiveUnet) ====================
    reg_unet = RegNaiveUnet(
        in_channels=3,
        out_channels=3,
        n_feat=32
    )

    ddpm = CurriculumDDPM(
        eps_model=reg_unet,
        betas=(1e-4, 0.02),
        n_T=1000
    )

    ddpm.to(device)

    # DDP 包装
    ddpm = DDP(ddpm, device_ids=[local_rank])

    if is_main_process(rank):
        print(f"\nRegNaiveUnet created (Group L1 Regularization)")
        conv_layers = reg_unet.get_conv_layers()
        print(f"  Total Conv layers: {len(conv_layers)}")
        reg_unet.print_reg_info(reg_threshold)

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
        os.makedirs("./contents_cr_32", exist_ok=True)
        os.makedirs("./checkpoints_cr_32", exist_ok=True)

    # 等待主进程创建目录
    dist.barrier()

    # ==================== Curriculum + Regularization 训练 ====================
    global_epoch = 0

    for stage_idx, (t_start, t_end, stage_name) in enumerate(curriculum_stages):
        lambda_curr = lambda_schedule[stage_idx]

        if is_main_process(rank):
            print(f"\n{'='*60}")
            print(f"CURRICULUM STAGE {stage_idx + 1}/{num_stages}")
            print(f"Stage: {stage_name}")
            print(f"Time range: t ∈ [{t_start:.1f}, {t_end:.1f}]")
            print(f"Lambda: {lambda_curr:.5f}")
            print(f"Epochs: {epochs_per_stage[stage_idx]}")
            print(f"{'='*60}")

        # 设置时间范围 (访问 DDP 内部模型)
        ddpm.module.set_time_range(t_start, t_end)

        # 训练这个 stage
        for epoch in range(epochs_per_stage[stage_idx]):
            # 设置 epoch 确保每个 epoch 的 shuffle 不同
            sampler.set_epoch(global_epoch)

            ddpm.train()

            if is_main_process(rank):
                pbar = tqdm(dataloader, desc=f"Stage {stage_idx+1} Epoch {epoch+1}")
            else:
                pbar = dataloader

            loss_mse_ema = None
            loss_reg_ema = None
            loss_total_ema = None

            for x, _ in pbar:
                optim.zero_grad()
                x = x.to(device)

                # MSE Loss (diffusion loss)
                loss_mse = ddpm(x)

                # Group L1 Regularization
                loss_reg = reg_unet.get_group_l1_penalty(lambda_curr)

                # Total Loss
                loss_total = loss_mse + loss_reg

                loss_total.backward()

                # EMA for logging
                if loss_mse_ema is None:
                    loss_mse_ema = loss_mse.item()
                    loss_reg_ema = loss_reg.item()
                    loss_total_ema = loss_total.item()
                else:
                    loss_mse_ema = 0.9 * loss_mse_ema + 0.1 * loss_mse.item()
                    loss_reg_ema = 0.9 * loss_reg_ema + 0.1 * loss_reg.item()
                    loss_total_ema = 0.9 * loss_total_ema + 0.1 * loss_total.item()

                if is_main_process(rank):
                    pbar.set_description(
                        f"Stage {stage_idx+1} Epoch {epoch+1} | "
                        f"total: {loss_total_ema:.4f} (mse: {loss_mse_ema:.4f}, reg: {loss_reg_ema:.4f})"
                    )

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
                    save_image(grid, f"./contents_cr_32/ddpm_celeba_s{stage_idx+1}_e{epoch+1}.png")

                # 保存最新模型
                torch.save(ddpm.module.state_dict(), "./ddpm_celeba_cr_32.pth")

            dist.barrier()

        # Stage 结束，保存 checkpoint 并打印正则化信息
        if is_main_process(rank):
            ckpt_path = f"./checkpoints_cr_32/ddpm_celeba_stage{stage_idx+1}.pth"
            torch.save(ddpm.module.state_dict(), ckpt_path)

            print(f"\nStage {stage_idx + 1} completed.")
            print(f"  Lambda used: {lambda_curr:.5f}")
            reg_unet.print_reg_info(reg_threshold)
            print(f"  Checkpoint saved: {ckpt_path}")

        dist.barrier()

    if is_main_process(rank):
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Total epochs: {global_epoch}")
        print(f"Final model: ./ddpm_celeba_cr_32.pth")
        print("\nFinal regularization info:")
        reg_unet.print_reg_info(reg_threshold)
        print(f"{'='*60}")

    cleanup_ddp()


if __name__ == "__main__":
    train_celeba_cr_32()
