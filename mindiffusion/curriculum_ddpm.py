"""
CurriculumDDPM: DDPM with Curriculum Learning Support

修改 forward() 函数，限制采样的 t 范围，实现课程学习

参考: mindiffusion/ddpm.py
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .ddpm import DDPM, ddpm_schedules


class CurriculumDDPM(DDPM):
    """
    DDPM with curriculum learning support

    课程学习策略:
    - 初始阶段只训练高噪声 (t ≈ n_T)
    - 逐步扩展到低噪声 (t → 0)
    - 通过 set_time_range() 动态调整训练范围
    """

    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__(eps_model, betas, n_T, criterion)

        # 时间范围 (归一化到 [0, 1])
        self.t_min = 0.0  # 默认全范围
        self.t_max = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Curriculum learning forward pass

        只从 [t_min*n_T, t_max*n_T] 范围采样时间步
        """
        # 计算时间步范围
        t_min_idx = max(1, int(self.t_min * self.n_T))
        t_max_idx = int(self.t_max * self.n_T)

        # 确保范围有效
        if t_min_idx >= t_max_idx:
            t_min_idx = t_max_idx - 1
        if t_min_idx < 1:
            t_min_idx = 1

        # 在范围内随机采样时间步
        _ts = torch.randint(t_min_idx, t_max_idx + 1, (x.shape[0],)).to(x.device)

        # 生成噪声
        eps = torch.randn_like(x)

        # 加噪
        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * eps
        )

        # 预测噪声并计算损失
        return self.criterion(eps, self.eps_model(x_t, _ts / self.n_T))

    def set_time_range(self, t_min: float, t_max: float):
        """
        设置训练的时间范围

        Args:
            t_min: 最小时间 (0 = 清晰图像, 1 = 纯噪声)
            t_max: 最大时间

        Examples:
            set_time_range(0.9, 1.0)  # Stage 1: 高噪声
            set_time_range(0.5, 1.0)  # Stage 5: 中等范围
            set_time_range(0.0, 1.0)  # Final: 全范围
        """
        self.t_min = max(0.0, min(t_min, 1.0))
        self.t_max = max(0.0, min(t_max, 1.0))

        if self.t_min >= self.t_max:
            self.t_min = self.t_max - 0.01

    def get_time_range(self) -> Tuple[float, float]:
        """获取当前时间范围"""
        return (self.t_min, self.t_max)

    def get_time_range_indices(self) -> Tuple[int, int]:
        """获取当前时间范围的索引"""
        t_min_idx = max(1, int(self.t_min * self.n_T))
        t_max_idx = int(self.t_max * self.n_T)
        return (t_min_idx, t_max_idx)

    def sample(self, n_sample: int, size, device) -> torch.Tensor:
        """
        采样 (使用完整的 reverse process)

        注意: 采样时始终使用完整的时间范围，不受 curriculum 限制
        """
        x_i = torch.randn(n_sample, *size).to(device)

        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.eps_model(
                x_i, torch.tensor(i / self.n_T).to(device).repeat(n_sample, 1)
            )
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        return x_i
