"""
CelebA DDPM Training - 32x32 版本 (DDP多GPU, 无 Curriculum/Sparsity)

标准训练，用于与 CS 版本对比

Usage:
    # 使用所有可用GPU
    torchrun --nproc_per_node=6 train_celeba_32.py

    # 指定GPU数量
    torchrun --nproc_per_node=4 train_celeba_32.py

    # 指定特定GPU
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_celeba_32.py
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

from mindiffusion.unet import NaiveUnet
from mindiffusion.ddpm import DDPM

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


def train_celeba_32() -> None:
    """
    标准训练 CelebA (32x32) with DDP
    无 Curriculum Learning，无 Sparsity Curriculum
    """

    # ==================== 初始化 DDP ====================
    rank, world_size, local_rank, device = setup_ddp()

    # ==================== 配置 ====================
    n_epoch = 66  # 与 CS 版本对齐
    global_batch_size = 128  # 总 batch size
    batch_size = global_batch_size // world_size  # 每个 GPU 的 batch size
    lr = 2e-5

    if is_main_process(rank):
        print(f"{'='*60}")
        print("CelebA Standard Training (32x32) with DDP")
        print(f"{'='*60}")
        print(f"World size: {world_size} GPUs")
        print(f"Global batch size: {global_batch_size}")
        print(f"Per-GPU batch size: {batch_size}")
        print(f"Learning rate: {lr}")
        print(f"Total epochs: {n_epoch}")
        print(f"{'='*60}")

    # ==================== 模型 ====================
    ddpm = DDPM(
        eps_model=NaiveUnet(3, 3, n_feat=32),
        betas=(1e-4, 0.02),
        n_T=1000
    )

    ddpm.to(device)

    # DDP 包装
    ddpm = DDP(ddpm, device_ids=[local_rank])

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
        os.makedirs("./contents_32", exist_ok=True)
        os.makedirs("./checkpoints_32", exist_ok=True)

    # 等待主进程创建目录
    dist.barrier()

    # ==================== 训练 ====================
    for epoch in range(n_epoch):
        # 设置 epoch 确保每个 epoch 的 shuffle 不同
        sampler.set_epoch(epoch)

        ddpm.train()

        if is_main_process(rank):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epoch}")
        else:
            pbar = dataloader

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

            if is_main_process(rank):
                pbar.set_description(f"Epoch {epoch+1}/{n_epoch} | loss: {loss_ema:.4f}")

            optim.step()

        # 同步所有进程
        dist.barrier()

        # 每个 epoch 生成样本 (只在主进程)
        if is_main_process(rank):
            ddpm.eval()
            with torch.no_grad():
                xh = ddpm.module.sample(8, (3, 32, 32), device)
                xset = torch.cat([xh, x[:8]], dim=0)
                grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
                save_image(grid, f"./contents_32/ddpm_celeba_e{epoch+1:03d}.png")

            # 保存最新模型
            torch.save(ddpm.module.state_dict(), "./ddpm_celeba_32.pth")

            # 每 10 个 epoch 保存 checkpoint
            if (epoch + 1) % 10 == 0:
                ckpt_path = f"./checkpoints_32/ddpm_celeba_epoch{epoch+1}.pth"
                torch.save(ddpm.module.state_dict(), ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")

        dist.barrier()

    if is_main_process(rank):
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Total epochs: {n_epoch}")
        print(f"Final model: ./ddpm_celeba_32.pth")
        print(f"{'='*60}")

    cleanup_ddp()


if __name__ == "__main__":
    train_celeba_32()
