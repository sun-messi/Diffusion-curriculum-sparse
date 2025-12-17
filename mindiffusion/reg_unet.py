"""
RegNaiveUnet: NaiveUnet with Group L1 Regularization

使用 Group L1 正则化实现软稀疏，无硬掩码
对所有 Conv2d 和 ConvTranspose2d 层的 output channels 施加正则化

Group L1 正则化形式：
    L_reg = λ · Σ_(所有Conv层) Σ_c ||W[c,:,:,:]||_2

其中：
- W shape: (out_channels, in_channels, kernel_h, kernel_w)
- 对每个 output channel 计算 L2 范数，然后求和

参考: curriculum_sparse_reg/reg_model.py
"""
import torch
import torch.nn as nn

from .unet import NaiveUnet


class RegNaiveUnet(NaiveUnet):
    """
    NaiveUnet with Group L1 regularization for soft sparsity

    与硬掩码版本 (SparseNaiveUnet) 的区别：
    - 无 channel_mask，不使用硬掩码
    - 通过 Group L1 正则化驱动稀疏
    - 对所有 Conv 层施加约束，不仅仅是 bottleneck
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_feat: int = 128
    ) -> None:
        super().__init__(in_channels, out_channels, n_feat)

    # ========== Group L1 正则化方法 ==========

    def get_conv_layers(self) -> list:
        """
        获取所有 Conv2d 和 ConvTranspose2d 层

        Returns:
            list of (name, module) tuples
        """
        conv_layers = []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                conv_layers.append((name, module))
        return conv_layers

    def get_channel_norms(self) -> dict:
        """
        计算每层每个 output channel 的 L2 范数

        Returns:
            dict: {layer_name: tensor of shape (out_channels,)}
        """
        channel_norms = {}
        for name, module in self.get_conv_layers():
            weight = module.weight  # (out_ch, in_ch, kH, kW)
            # reshape to (out_ch, -1), compute L2 norm along dim=1
            weight_2d = weight.view(weight.shape[0], -1)
            norms = torch.norm(weight_2d, p=2, dim=1)  # (out_ch,)
            channel_norms[name] = norms
        return channel_norms

    def get_group_l1_penalty(self, lambda_val: float) -> torch.Tensor:
        """
        计算 Group L1 正则化惩罚项

        L_reg = λ · Σ_(所有Conv层) Σ_c ||W[c,:,:,:]||_2

        Args:
            lambda_val: 正则化系数

        Returns:
            标量张量 - Group L1 惩罚值
        """
        if lambda_val == 0:
            # 获取设备
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)

        total_penalty = 0.0
        for name, module in self.get_conv_layers():
            weight = module.weight  # (out_ch, in_ch, kH, kW)
            weight_2d = weight.view(weight.shape[0], -1)
            channel_norms = torch.norm(weight_2d, p=2, dim=1)  # (out_ch,)
            total_penalty = total_penalty + channel_norms.sum()

        return lambda_val * total_penalty

    # ========== 稀疏性分析方法 (仅用于观测) ==========

    def get_sparsity_stats(self, threshold: float = 0.01) -> dict:
        """
        统计各层的 channel 活跃度

        Args:
            threshold: 判断 channel 是否活跃的阈值
                      channel_norm > threshold 视为活跃

        Returns:
            dict: {
                'total_channels': 总 channel 数,
                'active_channels': 活跃 channel 数,
                'sparsity': 稀疏度 (不活跃比例),
                'per_layer': {layer_name: {'total': N, 'active': M, 'ratio': M/N}}
            }
        """
        channel_norms = self.get_channel_norms()

        total_channels = 0
        active_channels = 0
        per_layer = {}

        for name, norms in channel_norms.items():
            layer_total = norms.shape[0]
            layer_active = (norms > threshold).sum().item()

            total_channels += layer_total
            active_channels += layer_active

            per_layer[name] = {
                'total': layer_total,
                'active': layer_active,
                'ratio': layer_active / layer_total if layer_total > 0 else 0
            }

        sparsity = 1.0 - (active_channels / total_channels) if total_channels > 0 else 0

        return {
            'total_channels': total_channels,
            'active_channels': active_channels,
            'sparsity': sparsity,
            'per_layer': per_layer
        }

    def get_norm_statistics(self) -> dict:
        """
        获取 channel norm 的统计信息

        Returns:
            dict: {
                'mean': 所有 channel norm 的平均值,
                'std': 标准差,
                'min': 最小值,
                'max': 最大值,
                'per_layer': {layer_name: {'mean': ..., 'std': ..., 'min': ..., 'max': ...}}
            }
        """
        channel_norms = self.get_channel_norms()

        all_norms = []
        per_layer = {}

        for name, norms in channel_norms.items():
            all_norms.append(norms)
            per_layer[name] = {
                'mean': norms.mean().item(),
                'std': norms.std().item(),
                'min': norms.min().item(),
                'max': norms.max().item()
            }

        all_norms_tensor = torch.cat(all_norms)

        return {
            'mean': all_norms_tensor.mean().item(),
            'std': all_norms_tensor.std().item(),
            'min': all_norms_tensor.min().item(),
            'max': all_norms_tensor.max().item(),
            'per_layer': per_layer
        }

    def print_reg_info(self, threshold: float = 0.01):
        """打印当前正则化相关信息"""
        stats = self.get_sparsity_stats(threshold)
        norm_stats = self.get_norm_statistics()

        print(f"    Channel Sparsity (threshold={threshold}):")
        print(f"      Total: {stats['active_channels']}/{stats['total_channels']} active ({1-stats['sparsity']:.1%})")
        print(f"      Sparsity: {stats['sparsity']:.1%}")
        print(f"    Norm Statistics:")
        print(f"      Mean: {norm_stats['mean']:.4f}, Std: {norm_stats['std']:.4f}")
        print(f"      Range: [{norm_stats['min']:.4f}, {norm_stats['max']:.4f}]")

    def print_per_layer_info(self, threshold: float = 0.01):
        """打印每层的详细信息"""
        stats = self.get_sparsity_stats(threshold)
        norm_stats = self.get_norm_statistics()

        print(f"    Per-layer Channel Activity (threshold={threshold}):")
        for name in stats['per_layer']:
            layer_stats = stats['per_layer'][name]
            layer_norms = norm_stats['per_layer'][name]
            print(f"      {name}: {layer_stats['active']}/{layer_stats['total']} "
                  f"({layer_stats['ratio']:.1%}) | "
                  f"norm: {layer_norms['mean']:.4f} ± {layer_norms['std']:.4f}")
