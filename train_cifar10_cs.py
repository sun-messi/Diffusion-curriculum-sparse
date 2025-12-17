"""
CIFAR10 DDPM Training with Curriculum Learning + Sparsity Curriculum + DDP

CS = Curriculum + Sparsity
支持多GPU并行训练

Usage:
    # 使用所有可用GPU
    torchrun --nproc_per_node=6 train_cifar10_cs.py

    # 指定GPU数量
    torchrun --nproc_per_node=4 train_cifar10_cs.py

    # 指定特定GPU
    CUDA_VISIBLE_DEVICES=0,2,3,4,5 torchrun --nproc_per_node=5 train_cifar10_cs.py
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

from mindiffusion.sparse_unet import SparseNaiveUnet
from mindiffusion.curriculum_ddpm import CurriculumDDPM
from config_cs import ConfigCS


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


def perform_channel_regrowth(model, target_sparsity: float, stage_idx: int, cfg: ConfigCS, rank: int) -> int:
    """
    激活 channels 直到达到目标稀疏度

    Args:
        model: SparseNaiveUnet 模型 (可能被 DDP 包装)
        target_sparsity: 目标稀疏度
        stage_idx: 当前 curriculum stage
        cfg: 配置对象
        rank: DDP rank

    Returns:
        实际激活的 channel 数量
    """
    # 获取原始模型 (unwrap DDP -> CurriculumDDPM -> SparseNaiveUnet)
    ddpm = model.module if hasattr(model, 'module') else model
    sparse_unet = ddpm.eps_model  # SparseNaiveUnet

    current_sparsity = sparse_unet.get_current_sparsity()

    # 只有在需要降低稀疏度时才 regrow
    if target_sparsity >= current_sparsity:
        return 0

    total_channels = sparse_unet.bottleneck_channels
    current_active = int(total_channels * (1 - current_sparsity))
    target_active = int(total_channels * (1 - target_sparsity))
    num_to_grow = target_active - current_active

    if num_to_grow <= 0:
        return 0

    if is_main_process(rank):
        print(f"\n    [Regrowth] Stage {stage_idx + 1}")
        print(f"    Target: {current_sparsity:.1%} -> {target_sparsity:.1%} sparsity")
        print(f"    Growing {num_to_grow} channels ({current_active} -> {target_active})")

    actually_grown = sparse_unet.regrow_channels(
        num_to_grow,
        current_stage=stage_idx,
        method=cfg.regrowth_method
    )

    if is_main_process(rank):
        print(f"    Regrew {actually_grown} channels")
        print(f"    New sparsity: {sparse_unet.get_current_sparsity():.1%}")

    return actually_grown


def train_one_epoch(
    ddpm,
    dataloader,
    optimizer,
    device,
    rank,
    epoch,
    stage_idx,
    cfg
):
    """训练一个 epoch"""
    ddpm.train()

    # 获取原始模型
    net = ddpm.module if hasattr(ddpm, 'module') else ddpm

    if is_main_process(rank):
        pbar = tqdm(dataloader, desc=f"Stage {stage_idx+1} Epoch {epoch+1}")
    else:
        pbar = dataloader

    loss_ema = None
    real_images = None

    for x, _ in pbar:
        optimizer.zero_grad()
        x = x.to(device)

        # 保存第一个 batch 的真实图片
        if real_images is None:
            real_images = x[:cfg.n_sample].clone()

        loss = ddpm(x)
        loss.backward()

        # 累积梯度用于 regrowth
        if cfg.sparsity_enabled:
            sparse_unet = net.eps_model
            sparse_unet.accumulate_gradients()

        optimizer.step()

        if loss_ema is None:
            loss_ema = loss.item()
        else:
            loss_ema = 0.9 * loss_ema + 0.1 * loss.item()

        if is_main_process(rank):
            pbar.set_description(f"Stage {stage_idx+1} Epoch {epoch+1} | loss: {loss_ema:.4f}")

    return loss_ema, real_images


def train_cs_ddp(cfg: ConfigCS = None) -> None:
    """
    Curriculum + Sparsity DDP 训练主函数

    Args:
        cfg: ConfigCS 对象，如果为 None 则使用默认配置
    """
    if cfg is None:
        cfg = ConfigCS()

    # 初始化 DDP
    rank, world_size, local_rank, device = setup_ddp()

    # 计算实际的 batch size 和 learning rate
    batch_size = cfg.get_batch_size(world_size)
    lr = cfg.get_lr(world_size)

    if is_main_process(rank):
        cfg.print_config(world_size)
        print(f"\nTraining with {world_size} GPUs")
        print(f"Batch size per GPU: {batch_size}")
        print(f"Total batch size: {batch_size * world_size}")
        print(f"Learning rate: {lr}")

    # ==================== 创建带稀疏的 UNet ====================
    sparse_unet = SparseNaiveUnet(
        in_channels=cfg.in_channels,
        out_channels=cfg.in_channels,
        n_feat=cfg.n_feat,
        initial_sparsity=cfg.initial_sparsity if cfg.sparsity_enabled else 0.0
    )

    if is_main_process(rank):
        print(f"\nCreated SparseNaiveUnet:")
        print(f"  Bottleneck channels: {sparse_unet.bottleneck_channels}")
        print(f"  Initial sparsity: {sparse_unet.get_current_sparsity():.1%}")
        print(f"  Active channels: {sparse_unet.get_active_channel_count()}")

    # ==================== 创建 Curriculum DDPM ====================
    ddpm = CurriculumDDPM(
        eps_model=sparse_unet,
        betas=(cfg.beta1, cfg.beta2),
        n_T=cfg.n_T
    )

    # 加载 checkpoint (如果存在)
    if os.path.exists(cfg.model_path):
        ddpm.load_state_dict(torch.load(cfg.model_path, map_location=device))
        if is_main_process(rank):
            print(f"Loaded checkpoint from {cfg.model_path}")

    ddpm.to(device)

    # DDP 包装
    ddpm = DDP(ddpm, device_ids=[local_rank])

    # ==================== 数据集 ====================
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CIFAR10(
        cfg.data_dir,
        train=True,
        download=is_main_process(rank),
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

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )

    # 优化器
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=lr)

    if is_main_process(rank):
        print(f"\nBatches per epoch: {len(dataloader)}")
        print(f"Total stages: {cfg.num_curriculum_stages}")
        print(f"Epochs schedule: {cfg.epochs_schedule}")
        print(f"Total epochs: {cfg.get_total_epochs()}")

    # 确保保存目录存在
    if is_main_process(rank):
        os.makedirs(cfg.save_dir, exist_ok=True)
        os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # ==================== Curriculum 训练循环 ====================
    global_epoch = 0

    for stage_idx, (t_start, t_end, stage_name) in enumerate(cfg.curriculum_stages):
        if is_main_process(rank):
            print(f"\n{'='*60}")
            print(f"CURRICULUM STAGE {stage_idx + 1}/{cfg.num_curriculum_stages}")
            print(f"Stage: {stage_name}")
            print(f"Time range: t ∈ [{t_start:.1f}, {t_end:.1f}]")
            if cfg.sparsity_enabled:
                target_sparsity = cfg.sparsity_schedule[stage_idx]
                print(f"Target sparsity: {target_sparsity:.0%}")
            print(f"{'='*60}")

        # 设置时间范围
        ddpm.module.set_time_range(t_start, t_end)

        # ========== Regrowth (stage > 0) ==========
        if stage_idx > 0 and cfg.sparsity_enabled:
            target_sparsity = cfg.sparsity_schedule[stage_idx]
            perform_channel_regrowth(ddpm, target_sparsity, stage_idx, cfg, rank)

        # ========== 应用 max_accumulated_stages 限制时间范围 (滑动窗口) ==========
        # 如果累积的 stage 数超过限制，调整 t_end 形成滑动窗口
        # 例如: max_accumulated_stages=6, num_stages=10, stage_idx=7 (第8个stage) 时
        #       原本 t ∈ [0.2, 1.0]，但只保留最近6个stage的范围
        #       丢掉 stage 1 的范围 [0.9, 1.0]，所以 t_end = 0.9
        #       结果: t ∈ [0.2, 0.8]
        if stage_idx >= cfg.max_accumulated_stages:
            # 计算需要丢掉多少个早期 stage
            stages_to_drop = stage_idx - cfg.max_accumulated_stages + 1
            # t_end 对应被丢掉的最后一个 stage 的 t_start
            effective_t_end = 1.0 - stages_to_drop * (1.0 / cfg.num_curriculum_stages)

            if is_main_process(rank):
                print(f"    [Curriculum] Sliding window: t ∈ [{t_start:.2f}, {effective_t_end:.2f}] (dropped {stages_to_drop} early stages)")

            ddpm.module.set_time_range(t_start, effective_t_end)

        # ========== 获取当前 stage 的 epoch 数 ==========
        stage_epochs = cfg.epochs_schedule[stage_idx]

        # ========== 训练 N epochs ==========
        is_final_stage = (stage_idx == cfg.num_curriculum_stages - 1)

        for epoch in range(stage_epochs):
            # 设置 epoch 确保每个 epoch 的 shuffle 不同
            sampler.set_epoch(global_epoch)

            loss_ema, real_images = train_one_epoch(
                ddpm, dataloader, optimizer, device,
                rank, epoch, stage_idx, cfg
            )

            global_epoch += 1

            # 同步所有进程
            dist.barrier()

            # 最后 stage 每 20 epochs 采样一次
            if is_final_stage and (epoch + 1) % 20 == 0 and is_main_process(rank):
                ddpm.eval()
                with torch.no_grad():
                    xh = ddpm.module.sample(
                        cfg.n_sample,
                        (cfg.in_channels, cfg.image_size, cfg.image_size),
                        device
                    )
                    grid = make_grid(xh, normalize=True, value_range=(-1, 1), nrow=4)
                    save_path = f"{cfg.save_dir}/ddpm_cs_stage{stage_idx+1}_epoch{epoch+1}.png"
                    save_image(grid, save_path)
                    print(f"Sample saved: {save_path}")
                ddpm.train()

        # ========== Stage 结束后的操作 ==========
        if is_main_process(rank):
            # 采样
            if (stage_idx + 1) % cfg.sample_every_stage == 0:
                ddpm.eval()
                with torch.no_grad():
                    xh = ddpm.module.sample(
                        cfg.n_sample,
                        (cfg.in_channels, cfg.image_size, cfg.image_size),
                        device
                    )
                    if real_images is not None:
                        xset = torch.cat([xh, real_images], dim=0)
                    else:
                        xset = xh
                    grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
                    save_path = f"{cfg.save_dir}/ddpm_cs_stage{stage_idx+1}.png"
                    save_image(grid, save_path)
                    print(f"Sample saved: {save_path}")

            # 保存最新模型
            torch.save(ddpm.module.state_dict(), cfg.model_path)

            # 定期保存 checkpoint
            if (stage_idx + 1) % cfg.save_every_stage == 0:
                ckpt_path = f"{cfg.ckpt_dir}/ddpm_cs_stage{stage_idx+1}.pth"
                torch.save(ddpm.module.state_dict(), ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")

            # 打印稀疏度信息
            if cfg.sparsity_enabled:
                sparse_unet = ddpm.module.eps_model
                sparse_unet.print_sparsity_info()

            print(f"Stage {stage_idx + 1} completed | loss_ema: {loss_ema:.4f}")

        # 同步所有进程
        dist.barrier()

    # ==================== 训练完成 ====================
    if is_main_process(rank):
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Total epochs: {global_epoch}")
        print(f"Final model saved: {cfg.model_path}")

        if cfg.sparsity_enabled:
            print("\nFinal sparsity info:")
            sparse_unet = ddpm.module.eps_model
            sparse_unet.print_sparsity_info()
        print(f"{'='*60}")

    cleanup_ddp()


if __name__ == "__main__":
    # 使用默认配置
    cfg = ConfigCS()

    # 可以在这里覆盖配置
    # cfg = ConfigCS(
    #     num_curriculum_stages=5,
    #     curriculum_epochs_per_stage=10,
    #     initial_sparsity=0.70,
    # )

    train_cs_ddp(cfg)
