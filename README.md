# Curriculum Learning + Sparsity for Diffusion Models

<p align="center">
<img src="figure1_timeline.png" width="800">
</p>

本项目实现了 **Curriculum Learning** 和两种稀疏化方法用于改进扩散模型训练。

---

## 目录

- [核心思想](#核心思想)
- [理论基础](#理论基础)
  - [为什么需要 Curriculum Learning](#为什么需要-curriculum-learning)
  - [为什么需要 Sparsity Curriculum](#为什么需要-sparsity-curriculum)
  - [两者的协同作用](#两者的协同作用)
- [三种训练变体](#三种训练变体)
- [架构设计](#架构设计)
  - [SparseNaiveUnet (CS)](#sparsenaiveunet-cs)
  - [RegNaiveUnet (CR)](#regnaiveunet-cr)
  - [CurriculumDDPM](#curriculumddpm)
- [实现细节](#实现细节)
  - [Channel-level Sparsity (CS)](#channel-level-sparsity-cs)
  - [Group L1 Regularization (CR)](#group-l1-regularization-cr)
  - [Gradient-based Regrowth](#gradient-based-regrowth)
- [使用方法](#使用方法)
- [配置说明](#配置说明)
- [项目结构](#项目结构)

---

## 核心思想

标准扩散模型训练对所有时间步一视同仁，但这忽略了一个重要结构：

| 时间步 | 噪声水平 | 可学习的内容 |
|--------|----------|--------------|
| t ≈ 1000 | 非常高 | 只有 **主要特征**（粗糙结构） |
| t ≈ 500 | 中等 | 中等特征 |
| t ≈ 1 | 非常低 | **精细细节**（次要特征） |

**核心洞察**：在高噪声水平下，精细细节完全被噪声掩盖。在这些样本上训练会迫使网络首先关注主要特征。

---

## 理论基础

### 为什么需要 Curriculum Learning

**核心思想**：从简单（主要特征）到困难（次要特征）逐步学习。

考虑一个简化的数据生成模型：
```
x = α₁·M₁·z₁ + α₂·M₂·z₂
```
其中 `α₁ >> α₂`（M₁ 是主要特征，M₂ 是次要特征）。

当我们添加噪声时：
```
x_noisy = √(1-t)·x + √t·ε
```

| 训练阶段 | 发生了什么 |
|----------|------------|
| **高噪声（t 大）** | 信号被严重掩盖 → 只有 **M₁** 可识别 |
| **低噪声（t 小）** | 信号更清晰 → **M₂** 开始显现 |

**没有 Curriculum**：网络在所有噪声级别同时学习所有特征 → 混乱，特征分离效果差。

**有 Curriculum**：
1. 首先在高噪声样本上训练 → 掌握 M₁
2. 然后扩展到低噪声样本 → 精细化 M₂
3. 自然的从粗到细学习

### 为什么需要 Sparsity Curriculum

**问题**：即使有 Curriculum Learning，如果所有神经元/通道从一开始就活跃：

```
Stage 1（高噪声）：
    所有 256 个 channels 都在更新 → 全部被 M₁ 吸引
    ↓
    Channels 被 M₁ "污染"
    ↓
Stage N（低噪声）：
    这些被污染的 channels 难以转向学习 M₂
    ↓
    结果：M₂ 表示效果差
```

**解决方案 - Sparsity Curriculum**：

```
Stage 1（高噪声，80% 稀疏）：
    只有 51 个 channels 活跃 → 专门学习 M₁
    其他 205 个 channels 保持干净（初始化状态）
    ↓
Stage 5（中等噪声，40% 稀疏）：
    Regrow 102 个新 channels → 干净的 channels 学习新特征
    ↓
Stage 10（低噪声，0% 稀疏）：
    所有 256 个 channels 活跃
    后期激活的 channels 直接学习 M₂，不受 M₁ 干扰
```

### 两者的协同作用

| 组件 | 控制 | 目的 |
|------|------|------|
| **Curriculum** | 数据难度（噪声：高 → 低） | 学**什么**（主要 → 次要特征） |
| **Sparsity** | 模型容量（稀疏 → 稠密） | **谁**来学（为次要特征预留容量） |

**一句话总结**：Curriculum 控制学习顺序；Sparsity 为后期特征预留干净的容量。

---

## 三种训练变体

| Script | Curriculum | Sparsity | Method | 说明 |
|--------|:----------:|:--------:|--------|------|
| `train_celeba_c_32.py` | ✓ | ✗ | Curriculum only | 只有时间步 curriculum |
| `train_celeba_cs_32.py` | ✓ | ✓ | Hard mask + regrowth | Bottleneck 硬掩码 |
| `train_celeba_cr_32.py` | ✓ | ✓ | Group L1 regularization | 软稀疏正则化 |

### CS vs CR 对比

| 特性 | CS (硬掩码) | CR (Group L1) |
|-----|------------|---------------|
| 稀疏机制 | `channel_mask` 直接掩盖 | 正则化惩罚驱动 |
| 稀疏位置 | 仅 bottleneck | 所有 Conv 层 |
| 控制方式 | 显式 regrowth | λ 递减自动释放 |
| 灵活性 | 离散 (0/1) | 连续 (软稀疏) |

---

## 架构设计

### SparseNaiveUnet (CS)

在 **bottleneck 处实现 channel-level sparsity** 的 UNet。

```
原始 UNet 流程：
    x → init_conv → down1 → down2 → down3 → to_vec
                                              ↓
                                      thro (B, 256, 1, 1)  ← BOTTLENECK
                                              ↓
                                      thro + time_embed
                                              ↓
                                      up0 → up1 → up2 → up3 → out

加入 Sparsity：
    thro = to_vec(down3)
    thro = thro * channel_mask    ← 在这里应用 MASK
    thro = up0(thro + temb)
```

**为什么选择 bottleneck？**
- 所有信息必须通过这个瓶颈
- Mask channels 直接限制模型容量
- 类似于在 MLP 中 mask hidden neurons

### RegNaiveUnet (CR)

对 **所有 Conv 层施加 Group L1 正则化** 的 UNet。

```python
# Group L1 正则化形式：
L_reg = λ · Σ_(所有Conv层) Σ_c ||W[c,:,:,:]||_2

# 其中：
# - W shape: (out_channels, in_channels, kernel_h, kernel_w)
# - 对每个 output channel 计算 L2 范数，然后求和
```

**总 Loss**：
```
L_total = MSE(noise_pred, noise) + λ(stage) × Σ_c ||W[c,:,:,:]||_2
```

**λ Schedule (Cosine)**：
```
λ(i) = λ_max × 0.5 × (1 + cos(π × i / (num_stages - 1)))

Stage 1: λ = λ_max (强正则化)
Stage 5: λ ≈ 0 (无正则化)
```

### CurriculumDDPM

支持动态时间步范围控制的 DDPM。

```python
# 标准 DDPM: t ~ Uniform(1, 1000)
# Curriculum DDPM: t ~ Uniform(t_min, t_max)

# Stage 1: t ∈ [0.8, 1.0] - 只有高噪声
# Stage 3: t ∈ [0.4, 1.0] - 扩展范围
# Stage 5: t ∈ [0.0, 1.0] - 完整范围
```

---

## 实现细节

### Channel-level Sparsity (CS)

位置：`mindiffusion/sparse_unet.py`

```python
class SparseNaiveUnet(nn.Module):
    def __init__(self, ..., initial_sparsity=0.0):
        # Channel mask: 1 = 活跃, 0 = 被 mask
        self.register_buffer('channel_mask', torch.ones(256))

        # 记录每个 channel 是在哪个 stage 被激活的
        self.register_buffer('channel_birth_stage', torch.zeros(256))

    def forward(self, x, t):
        ...
        thro = self.to_vec(down3)  # (B, 256, 1, 1)

        # 在 bottleneck 应用 channel mask
        thro = thro * self.channel_mask.view(1, -1, 1, 1)

        thro = self.up0(thro + temb)
        ...
```

### Group L1 Regularization (CR)

位置：`mindiffusion/reg_unet.py`

```python
class RegNaiveUnet(NaiveUnet):
    def get_group_l1_penalty(self, lambda_val: float) -> torch.Tensor:
        """计算 Group L1 正则化惩罚项"""
        if lambda_val == 0:
            return torch.tensor(0.0, device=self.device)

        total_penalty = 0.0
        for name, module in self.get_conv_layers():
            weight = module.weight  # (out_ch, in_ch, kH, kW)
            weight_2d = weight.view(weight.shape[0], -1)
            channel_norms = torch.norm(weight_2d, p=2, dim=1)  # (out_ch,)
            total_penalty = total_penalty + channel_norms.sum()

        return lambda_val * total_penalty
```

### Gradient-based Regrowth

**关键洞察**：即使被 mask 的 channels 在 backprop 时也会收到梯度！

```python
def regrow_channels(self, num_to_grow, current_stage, method="gradient"):
    if method == "gradient":
        # 选择梯度幅度最大的 inactive channels
        grad_for_selection = self.get_channel_gradients()
        grad_for_selection[active_channels] = -inf  # 排除已活跃的
        _, topk_indices = torch.topk(grad_for_selection, num_to_grow)

    # 激活选中的 channels
    self.channel_mask[topk_indices] = 1
    self.channel_birth_stage[topk_indices] = current_stage
```

---

## 使用方法

### CelebA 32x32 训练

```bash
# Curriculum Only (C)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train_celeba_c_32.py

# Curriculum + Sparsity (CS) - Hard Mask
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train_celeba_cs_32.py

# Curriculum + Regularization (CR) - Group L1
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train_celeba_cr_32.py

# 多 GPU
torchrun --nproc_per_node=4 train_celeba_cr_32.py

# 指定不同端口（避免冲突）
torchrun --nproc_per_node=1 --master_port=29501 train_celeba_cr_32.py
```

### 环境配置

设置 `.env` 文件：
```
CELEBA_PATH=/path/to/celeba/dataset
```

---

## 配置说明

### 当前配置 (32x32)

```python
# 共同配置
n_feat = 32
num_stages = 5
epochs_per_stage = [5, 10, 15, 20, 30]  # 总共 80 epochs
global_batch_size = 256
lr = 2e-5

# Curriculum stages
curriculum_stages = [
    (0.8, 1.0, "stage1_high_noise"),
    (0.6, 1.0, "stage2_expand"),
    (0.4, 1.0, "stage3_expand"),
    (0.2, 1.0, "stage4_expand"),
    (0.0, 1.0, "stage5_full_range"),
]

# CS 特有配置
sparsity_schedule = [0.80, 0.60, 0.40, 0.20, 0.00]

# CR 特有配置
lambda_max = 0.00005
lambda_schedule = cosine_schedule(lambda_max, num_stages)
# → [0.00005, 0.0000427, 0.000025, 0.0000073, 0.0]
```

---

## 项目结构

```
minDiffusion_curriculum/
├── mindiffusion/
│   ├── __init__.py
│   ├── unet.py              # 原始 NaiveUnet
│   ├── sparse_unet.py       # SparseNaiveUnet (CS - 硬掩码)
│   ├── reg_unet.py          # RegNaiveUnet (CR - Group L1)
│   ├── ddpm.py              # 原始 DDPM
│   ├── curriculum_ddpm.py   # CurriculumDDPM（时间范围控制）
│   └── ddim.py              # DDIM 采样器
│
├── train_celeba_c_32.py     # Curriculum Only
├── train_celeba_cs_32.py    # Curriculum + Sparsity (Hard Mask)
├── train_celeba_cr_32.py    # Curriculum + Regularization (Group L1)
│
├── contents_c_32/           # C 生成样本
├── contents_cs_32/          # CS 生成样本
├── contents_cr_32/          # CR 生成样本
│
├── checkpoints_c_32/        # C checkpoints
├── checkpoints_cs_32/       # CS checkpoints
└── checkpoints_cr_32/       # CR checkpoints
```

---

## 核心要点

1. **Curriculum Learning** 提供自然的从粗到细学习轨迹
2. **CS (Hard Mask)** 在 bottleneck 直接限制容量，显式 regrowth
3. **CR (Group L1)** 对所有 Conv 层软约束，λ 递减自动释放
4. **两种稀疏方法** 都为精细特征预留干净的模型容量

---

## 参考

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Original minDiffusion](https://github.com/cloneofsimo/minDiffusion)
