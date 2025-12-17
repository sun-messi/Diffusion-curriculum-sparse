"""
SparseNaiveUnet: NaiveUnet with Channel-level Sparsity at Bottleneck

稀疏策略: mask bottleneck (thro) 的部分 channels
- channel_mask: (2*n_feat,) 控制哪些 channel 活跃
- 在 to_vec 之后、up0 之前应用 mask
- 支持基于梯度的 regrowth

参考: curriculum_sparse_ablation/sparse_model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import Conv3, UnetDown, UnetUp, TimeSiren


class SparseNaiveUnet(nn.Module):
    """
    NaiveUnet with channel-level sparsity at bottleneck

    原版 forward 流程:
        x → init_conv → down1 → down2 → down3 → to_vec
                                                   ↓
                                           thro (B, 2*n_feat, 1, 1)
                                                   ↓
                                           thro + temb
                                                   ↓
                                           up0 → up1 → up2 → up3 → out

    修改后:
        thro = to_vec(down3)
        thro = thro * channel_mask    ← 在这里应用 mask
        thro = up0(thro + temb)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_feat: int = 128,
        initial_sparsity: float = 0.0
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_feat = n_feat
        self.bottleneck_channels = 2 * n_feat  # 256 if n_feat=128

        # ==================== 复用原版 UNet 的所有层 ====================
        self.init_conv = Conv3(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(4), nn.ReLU())

        self.timeembed = TimeSiren(2 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 4, 4),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, 2 * n_feat)
        self.up2 = UnetUp(4 * n_feat, n_feat)
        self.up3 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Conv2d(2 * n_feat, self.out_channels, 3, 1, 1)

        # ==================== Channel-level Sparsity ====================
        # channel_mask: 1 = active, 0 = masked
        self.register_buffer('channel_mask', torch.ones(self.bottleneck_channels))

        # 记录每个 channel 是在哪个 stage 被激活的
        # 0 = 初始活跃, >0 = 在第 N 个 stage 通过 regrowth 激活
        self.register_buffer('channel_birth_stage', torch.zeros(self.bottleneck_channels, dtype=torch.long))

        # 用于存储梯度信息 (regrowth 时使用)
        self.register_buffer('channel_grad_accum', torch.zeros(self.bottleneck_channels))
        self.grad_accum_count = 0

        # 初始化稀疏 mask
        if initial_sparsity > 0:
            self._initialize_sparse_mask(initial_sparsity)

    def _initialize_sparse_mask(self, sparsity: float):
        """
        随机 mask 部分 channels

        Args:
            sparsity: 稀疏度 (0 = 全部活跃, 1 = 全部 mask)
        """
        num_to_mask = int(self.bottleneck_channels * sparsity)
        mask_indices = torch.randperm(self.bottleneck_channels, device=self.channel_mask.device)[:num_to_mask]
        self.channel_mask[mask_indices] = 0

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        前向传播，在 bottleneck 处应用 channel mask
        """
        x = self.init_conv(x)

        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)

        thro = self.to_vec(down3)  # (B, 2*n_feat, 1, 1)

        # ========== 应用 channel mask ==========
        thro = thro * self.channel_mask.view(1, -1, 1, 1)

        temb = self.timeembed(t).view(-1, self.n_feat * 2, 1, 1)

        thro = self.up0(thro + temb)

        up1 = self.up1(thro, down3) + temb
        up2 = self.up2(up1, down2)
        up3 = self.up3(up2, down1)

        out = self.out(torch.cat((up3, x), 1))

        return out

    def accumulate_gradients(self):
        """
        累积 bottleneck 层的梯度信息

        在每次 backward 后调用，用于 regrowth 决策
        梯度来源: up0 的第一个 ConvTranspose2d 层的输入
        """
        # 获取 up0 第一层 (ConvTranspose2d) 的权重梯度
        conv_transpose = self.up0[0]
        if conv_transpose.weight.grad is not None:
            # weight shape: (in_channels, out_channels, kH, kW) = (256, 256, 4, 4)
            # 对每个输入 channel 计算梯度幅度
            grad_per_channel = conv_transpose.weight.grad.abs().sum(dim=(1, 2, 3))  # (256,)
            self.channel_grad_accum += grad_per_channel
            self.grad_accum_count += 1

    def get_channel_gradients(self) -> torch.Tensor:
        """
        获取累积的 channel 梯度

        Returns:
            (bottleneck_channels,) 每个 channel 的平均梯度幅度
        """
        if self.grad_accum_count == 0:
            return None
        return self.channel_grad_accum / self.grad_accum_count

    def reset_gradient_accumulation(self):
        """重置梯度累积"""
        self.channel_grad_accum.zero_()
        self.grad_accum_count = 0

    def regrow_channels(
        self,
        num_to_grow: int,
        current_stage: int,
        method: str = "gradient"
    ) -> int:
        """
        基于梯度或随机激活新的 channels

        Args:
            num_to_grow: 要激活的 channel 数量
            current_stage: 当前 curriculum stage (用于记录 birth_stage)
            method: "gradient" 或 "random"

        Returns:
            实际激活的 channel 数量
        """
        inactive_mask = (self.channel_mask == 0)
        num_inactive = inactive_mask.sum().item()

        if num_inactive == 0:
            print(f"    [Regrowth] No inactive channels to regrow")
            return 0

        num_to_grow = min(num_to_grow, num_inactive)

        if num_to_grow == 0:
            return 0

        if method == "gradient":
            grad_magnitudes = self.get_channel_gradients()

            if grad_magnitudes is None:
                print(f"    [Regrowth] No gradients available, falling back to random")
                method = "random"
            else:
                # 只考虑 inactive channels
                grad_for_selection = grad_magnitudes.clone()
                grad_for_selection[~inactive_mask] = -float('inf')

                # 选择梯度最大的 N 个 inactive channels
                _, topk_indices = torch.topk(grad_for_selection, num_to_grow)

        if method == "random":
            inactive_indices = inactive_mask.nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(inactive_indices), device=self.channel_mask.device)
            topk_indices = inactive_indices[perm[:num_to_grow]]

        # 激活选中的 channels
        self.channel_mask[topk_indices] = 1

        # 记录 birth stage
        self.channel_birth_stage[topk_indices] = current_stage

        # 重置累积的梯度
        self.reset_gradient_accumulation()

        return num_to_grow

    def get_current_sparsity(self) -> float:
        """
        获取当前稀疏度

        Returns:
            稀疏度 (0 = 全部活跃, 1 = 全部 mask)
        """
        return (self.channel_mask == 0).sum().item() / self.bottleneck_channels

    def get_active_channel_count(self) -> int:
        """获取活跃 channel 数量"""
        return (self.channel_mask == 1).sum().item()

    def get_channel_stats_by_stage(self) -> dict:
        """
        获取按 birth stage 分组的 channel 统计

        Returns:
            {stage: {'total_born': N, 'currently_active': M}}
        """
        stats = {}
        max_stage = self.channel_birth_stage.max().item()

        for stage in range(max_stage + 1):
            stage_mask = (self.channel_birth_stage == stage)
            total_born = stage_mask.sum().item()
            currently_active = (stage_mask & (self.channel_mask == 1)).sum().item()

            if total_born > 0:
                stats[stage] = {
                    'total_born': total_born,
                    'currently_active': currently_active
                }

        return stats

    def print_sparsity_info(self):
        """打印当前稀疏度信息"""
        sparsity = self.get_current_sparsity()
        active = self.get_active_channel_count()
        total = self.bottleneck_channels
        stats = self.get_channel_stats_by_stage()

        print(f"    Sparsity: {sparsity:.1%} ({active}/{total} channels active)")
        print(f"    Channels by birth stage: {stats}")
