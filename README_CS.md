# Curriculum Learning + Sparsity Curriculum for Diffusion Models

本项目在 minDiffusion 基础上实现了 **Curriculum Learning** 和 **Sparsity Curriculum**，用于改进扩散模型训练。

---

## 目录

- [核心思想](#核心思想)
- [理论基础](#理论基础)
  - [为什么需要 Curriculum Learning](#为什么需要-curriculum-learning)
  - [为什么需要 Sparsity Curriculum](#为什么需要-sparsity-curriculum)
  - [两者的协同作用](#两者的协同作用)
- [架构设计](#架构设计)
  - [SparseNaiveUnet](#sparsenaiveunet)
  - [CurriculumDDPM](#curriculumddpm)
- [实现细节](#实现细节)
  - [Channel-level Sparsity](#channel-level-sparsity)
  - [Gradient-based Regrowth](#gradient-based-regrowth)
  - [滑动窗口机制](#滑动窗口机制)
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

## 架构设计

### SparseNaiveUnet

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
- 所有信息必须通过这个 256 通道的瓶颈
- Mask channels 直接限制模型容量
- 类似于在 MLP 中 mask hidden neurons

### CurriculumDDPM

支持动态时间步范围控制的 DDPM。

```python
# 标准 DDPM: t ~ Uniform(1, 1000)
# Curriculum DDPM: t ~ Uniform(t_min, t_max)

# Stage 1: t ∈ [900, 1000] - 只有高噪声
# Stage 5: t ∈ [500, 1000] - 扩展范围
# Stage 10: t ∈ [0, 1000] - 完整范围
```

---

## 实现细节

### Channel-level Sparsity

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

### Gradient-based Regrowth

**关键洞察**：即使被 mask 的 channels 在 backprop 时也会收到梯度！

```python
def accumulate_gradients(self):
    """
    累积 up0 的 ConvTranspose2d 层的梯度。
    梯度幅度表示："如果这个 channel 活跃，能多大程度降低 loss？"
    """
    conv_transpose = self.up0[0]  # ConvTranspose2d(256, 256, 4, 4)
    if conv_transpose.weight.grad is not None:
        # 对每个输入 channel 求梯度幅度之和
        grad_per_channel = conv_transpose.weight.grad.abs().sum(dim=(1, 2, 3))
        self.channel_grad_accum += grad_per_channel

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

**为什么梯度选择有效**：
- 在 Stage N（低噪声），M₂ 特征变得重要
- 能帮助学习 M₂ 的 channels 会有高梯度
- 选择高梯度 channels = 选择会学习 M₂ 的 channels

### 滑动窗口机制

防止累积所有时间步范围导致内存爆炸：

```python
# 没有滑动窗口：
# Stage 7: t ∈ [0.3, 1.0] - 范围太大，数据太多

# 有滑动窗口 (max_accumulated_stages=6)：
# Stage 7: t ∈ [0.3, 0.9] - 丢弃最早期 stage 的范围
# 只保留最近 6 个 stages 的时间步范围
```

---

## 使用方法

### Curriculum + Sparsity 训练

```bash
# 多 GPU（推荐）
torchrun --nproc_per_node=6 train_cifar10_cs.py

# 指定 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_cifar10_cs.py
```

### 标准训练（对比用）

```bash
# 单 GPU
python train_cifar10.py

# 多 GPU
torchrun --nproc_per_node=4 train_cifar10_ddp.py
```

---

## 配置说明

编辑 `config_cs.py` 自定义训练：

```python
class ConfigCS:
    # ==================== 模型 ====================
    n_feat: int = 128           # UNet hidden dimension
    n_T: int = 1000             # 扩散步数

    # ==================== Curriculum Learning ====================
    curriculum_enabled: bool = True
    num_curriculum_stages: int = 10

    # 每个 stage 的 epochs 数: [早期 stages ... 后期 stages]
    # 在低噪声（更难）的 stages 训练更久
    epochs_schedule: list = [3, 3, 3, 5, 5, 5, 8, 8, 8, 20]  # 总计: 68 epochs

    # ==================== Sparsity Curriculum ====================
    sparsity_enabled: bool = True
    initial_sparsity: float = 0.80    # 初始 80% channels 被 mask
    final_sparsity: float = 0.00      # 最终所有 channels 活跃
    regrowth_method: str = "random"   # "gradient" 或 "random"

    # ==================== 滑动窗口 ====================
    max_accumulated_stages: int = 7   # 限制时间步范围累积
```

### 自动生成的 Schedule

当 `num_curriculum_stages=10` 时：

| Stage | 时间范围 | 稀疏度 | Epochs | 活跃 Channels |
|-------|----------|--------|--------|---------------|
| 1 | t ∈ [0.9, 1.0] | 80% | 3 | 51/256 |
| 2 | t ∈ [0.8, 1.0] | 72% | 3 | 72/256 |
| 3 | t ∈ [0.7, 1.0] | 64% | 3 | 92/256 |
| 4 | t ∈ [0.6, 1.0] | 56% | 5 | 113/256 |
| 5 | t ∈ [0.5, 1.0] | 48% | 5 | 133/256 |
| 6 | t ∈ [0.4, 1.0] | 40% | 5 | 154/256 |
| 7 | t ∈ [0.3, 1.0] | 32% | 8 | 174/256 |
| 8 | t ∈ [0.2, 1.0] | 24% | 8 | 194/256 |
| 9 | t ∈ [0.1, 1.0] | 16% | 8 | 215/256 |
| 10 | t ∈ [0.0, 1.0] | 0% | 20 | 256/256 |

---

## 项目结构

```
minDiffusion_curriculum/
├── mindiffusion/
│   ├── __init__.py
│   ├── unet.py              # 原始 NaiveUnet
│   ├── sparse_unet.py       # SparseNaiveUnet（channel sparsity）
│   ├── ddpm.py              # 原始 DDPM
│   ├── curriculum_ddpm.py   # CurriculumDDPM（时间范围控制）
│   └── ddim.py              # DDIM 采样器
│
├── config.py                # 标准训练配置
├── config_cs.py             # Curriculum + Sparsity 配置
│
├── train_cifar10.py         # 标准单 GPU 训练
├── train_cifar10_ddp.py     # 标准多 GPU 训练
├── train_cifar10_cs.py      # Curriculum + Sparsity 训练（DDP）
│
├── contents/                # 生成样本（标准）
├── contents_cs/             # 生成样本（curriculum+sparsity）
├── checkpoints/             # 模型 checkpoints
└── checkpoints_cs/          # Curriculum+Sparsity checkpoints
```

---

## 核心要点

1. **Curriculum Learning** 提供自然的从粗到细学习轨迹
2. **Sparsity Curriculum** 为精细特征预留干净的模型容量
3. **两者结合** 实现更好的特征分离和表示
4. **Gradient-based regrowth** 智能选择应该激活哪些 channels

---

## 参考

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Original minDiffusion](https://github.com/cloneofsimo/minDiffusion)
- Curriculum Learning 概念来自 [curriculum_sparse_ablation](../diffusion/curriculum_sparse_ablation)
