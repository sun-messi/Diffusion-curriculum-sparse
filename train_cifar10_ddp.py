"""
CIFAR10 DDPM Training with DDP (DistributedDataParallel)
支持多GPU并行训练

Usage:
    # 使用所有可用GPU
    torchrun --nproc_per_node=6 train_cifar10_ddp.py

    # 指定GPU数量
    torchrun --nproc_per_node=4 train_cifar10_ddp.py

    # 指定特定GPU
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_cifar10_ddp.py
"""

import os
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from mindiffusion.unet import NaiveUnet
from mindiffusion.ddpm import DDPM
from config import Config


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


def train_cifar10_ddp(cfg: Config = None) -> None:
    """
    DDP 训练主函数

    Args:
        cfg: Config 对象，如果为 None 则使用默认配置
    """
    if cfg is None:
        cfg = Config()

    # 初始化DDP
    rank, world_size, local_rank, device = setup_ddp()

    # 计算实际的 batch size 和 learning rate
    batch_size = cfg.get_batch_size(world_size)  # global_batch_size // world_size
    lr = cfg.get_lr(world_size)

    if is_main_process(rank):
        cfg.print_config(world_size)
        print(f"Training with {world_size} GPUs")
        print(f"Batch size per GPU: {batch_size}")
        print(f"Total batch size: {batch_size * world_size}")
        print(f"Learning rate: {lr}")

    # 创建模型
    ddpm = DDPM(
        eps_model=NaiveUnet(cfg.in_channels, cfg.in_channels, n_feat=cfg.n_feat),
        betas=(cfg.beta1, cfg.beta2),
        n_T=cfg.n_T
    )

    # 加载checkpoint (如果存在)
    if os.path.exists(cfg.model_path):
        ddpm.load_state_dict(torch.load(cfg.model_path, map_location=device))
        if is_main_process(rank):
            print(f"Loaded checkpoint from {cfg.model_path}")

    ddpm.to(device)

    # 用DDP包装模型
    ddpm = DDP(ddpm, device_ids=[local_rank])

    # 数据预处理
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 数据集
    dataset = CIFAR10(
        cfg.data_dir,
        train=True,
        download=is_main_process(rank),  # 只在主进程下载
        transform=tf,
    )

    # 等待主进程下载完成
    dist.barrier()

    # 分布式采样器
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    # DataLoader - 注意shuffle=False，因为sampler会处理
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 使用sampler时必须为False
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )

    # 优化器
    optim = torch.optim.Adam(ddpm.parameters(), lr=lr)

    if is_main_process(rank):
        print(f"Batches per epoch: {len(dataloader)}")
        print(f"Total iterations: {len(dataloader) * cfg.n_epoch}")

    # 确保保存目录存在
    if is_main_process(rank):
        os.makedirs(cfg.save_dir, exist_ok=True)
        os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # 训练循环
    for epoch in range(cfg.n_epoch):
        # 设置epoch确保每个epoch的shuffle不同 (重要!)
        sampler.set_epoch(epoch)

        ddpm.train()

        # 只在主进程显示进度条
        if is_main_process(rank):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        else:
            pbar = dataloader

        loss_ema = None
        real_images = None  # 保存第一个batch的真实图片用于可视化
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            # 保存第一个batch的前n_sample张真实图片
            if real_images is None:
                real_images = x[:cfg.n_sample].clone()
            loss = ddpm(x)
            loss.backward()
            optim.step()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()

            if is_main_process(rank):
                pbar.set_description(f"Epoch {epoch} | loss: {loss_ema:.4f}")

        # 只在主进程保存和采样
        if is_main_process(rank):
            # 采样
            if (epoch + 1) % cfg.sample_every == 0:
                ddpm.eval()
                with torch.no_grad():
                    # 采样时使用module获取原始模型
                    xh = ddpm.module.sample(cfg.n_sample, (cfg.in_channels, cfg.image_size, cfg.image_size), device)
                    # 使用保存的第一个batch的真实图片，避免最后batch不足n_sample张
                    xset = torch.cat([xh, real_images], dim=0)
                    grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
                    save_image(grid, f"{cfg.save_dir}/ddpm_sample_cifar{epoch}.png")

            # 保存模型
            # 1. 每epoch保存最新的 (覆盖)
            torch.save(ddpm.module.state_dict(), cfg.model_path)
            # 2. 每save_every个epoch保存一个checkpoint (带编号)
            if (epoch + 1) % cfg.save_every == 0:
                ckpt_path = f"{cfg.ckpt_dir}/ddpm_cifar_epoch{epoch}.pth"
                torch.save(ddpm.module.state_dict(), ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")

            print(f"Epoch {epoch} done | loss_ema: {loss_ema:.4f}")

        # 同步所有进程
        dist.barrier()

    cleanup_ddp()


if __name__ == "__main__":
    # 使用默认配置
    cfg = Config()

    # 可以在这里覆盖配置
    # cfg = Config(n_epoch=200, lr=2e-5)

    train_cifar10_ddp(cfg)
