# Curriculum Learning + Sparsity for Diffusion Models

<p align="center">
<img src="figure1_timeline.png" width="800">
</p>

This project implements **Curriculum Learning** and two sparsification methods to improve diffusion model training.

---

## Table of Contents

- [Core Idea](#core-idea)
- [Theoretical Background](#theoretical-background)
  - [Why Curriculum Learning](#why-curriculum-learning)
  - [Why Sparsity Curriculum](#why-sparsity-curriculum)
  - [Synergy Between Both](#synergy-between-both)
- [Three Training Variants](#three-training-variants)
- [Architecture Design](#architecture-design)
  - [SparseNaiveUnet (CS)](#sparsenaiveunet-cs)
  - [RegNaiveUnet (CR)](#regnaiveunet-cr)
  - [CurriculumDDPM](#curriculumddpm)
- [Implementation Details](#implementation-details)
  - [Channel-level Sparsity (CS)](#channel-level-sparsity-cs)
  - [Group L1 Regularization (CR)](#group-l1-regularization-cr)
  - [Gradient-based Regrowth](#gradient-based-regrowth)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)

---

## Core Idea

Standard diffusion model training treats all timesteps equally, but this ignores an important structure:

| Timestep | Noise Level | Learnable Content |
|----------|-------------|-------------------|
| t ≈ 1000 | Very high | Only **major features** (coarse structure) |
| t ≈ 500 | Medium | Medium features |
| t ≈ 1 | Very low | **Fine details** (minor features) |

**Key Insight**: At high noise levels, fine details are completely masked by noise. Training on these samples forces the network to focus on major features first.

---

## Theoretical Background

### Why Curriculum Learning

**Core idea**: Learn progressively from easy (major features) to hard (minor features).

Consider a simplified data generation model:
```
x = α₁·M₁·z₁ + α₂·M₂·z₂
```
where `α₁ >> α₂` (M₁ is the major feature, M₂ is the minor feature).

When we add noise:
```
x_noisy = √(1-t)·x + √t·ε
```

| Training Phase | What Happens |
|----------------|--------------|
| **High noise (large t)** | Signal heavily masked → Only **M₁** is identifiable |
| **Low noise (small t)** | Signal clearer → **M₂** starts to emerge |

**Without Curriculum**: Network learns all features at all noise levels simultaneously → Confusion, poor feature separation.

**With Curriculum**:
1. First train on high-noise samples → Master M₁
2. Then expand to low-noise samples → Refine M₂
3. Natural coarse-to-fine learning

### Why Sparsity Curriculum

**Problem**: Even with Curriculum Learning, if all neurons/channels are active from the start:

```
Stage 1 (high noise):
    All 256 channels updating → All attracted to M₁
    ↓
    Channels "contaminated" by M₁
    ↓
Stage N (low noise):
    These contaminated channels struggle to learn M₂
    ↓
    Result: Poor M₂ representation
```

**Solution - Sparsity Curriculum**:

```
Stage 1 (high noise, 80% sparse):
    Only 51 channels active → Dedicated to learning M₁
    Other 205 channels stay clean (initialized state)
    ↓
Stage 5 (medium noise, 40% sparse):
    Regrow 102 new channels → Clean channels learn new features
    ↓
Stage 10 (low noise, 0% sparse):
    All 256 channels active
    Late-activated channels directly learn M₂, uncontaminated by M₁
```

### Synergy Between Both

| Component | Controls | Purpose |
|-----------|----------|---------|
| **Curriculum** | Data difficulty (noise: high → low) | **What** to learn (major → minor features) |
| **Sparsity** | Model capacity (sparse → dense) | **Who** learns (reserve capacity for minor features) |

**One-liner**: Curriculum controls learning order; Sparsity reserves clean capacity for later features.

---

## Three Training Variants

| Script | Curriculum | Sparsity | Method | Description |
|--------|:----------:|:--------:|--------|-------------|
| `train_celeba_c_32.py` | ✓ | ✗ | Curriculum only | Only timestep curriculum |
| `train_celeba_cs_32.py` | ✓ | ✓ | Hard mask + regrowth | Bottleneck hard mask |
| `train_celeba_cr_32.py` | ✓ | ✓ | Group L1 regularization | Soft sparsity regularization |

### CS vs CR Comparison

| Feature | CS (Hard Mask) | CR (Group L1) |
|---------|----------------|---------------|
| Sparsity mechanism | `channel_mask` directly masks | Regularization penalty drives sparsity |
| Sparsity location | Bottleneck only | All Conv layers |
| Control method | Explicit regrowth | λ decay auto-releases |
| Flexibility | Discrete (0/1) | Continuous (soft sparsity) |

---

## Architecture Design

### SparseNaiveUnet (CS)

UNet with **channel-level sparsity at the bottleneck**.

```
Original UNet flow:
    x → init_conv → down1 → down2 → down3 → to_vec
                                              ↓
                                      thro (B, 256, 1, 1)  ← BOTTLENECK
                                              ↓
                                      thro + time_embed
                                              ↓
                                      up0 → up1 → up2 → up3 → out

With Sparsity:
    thro = to_vec(down3)
    thro = thro * channel_mask    ← Apply MASK here
    thro = up0(thro + temb)
```

**Why bottleneck?**
- All information must pass through this bottleneck
- Masking channels directly limits model capacity
- Similar to masking hidden neurons in MLP

### RegNaiveUnet (CR)

UNet with **Group L1 regularization on all Conv layers**.

```python
# Group L1 regularization form:
L_reg = λ · Σ_(all Conv layers) Σ_c ||W[c,:,:,:]||_2

# Where:
# - W shape: (out_channels, in_channels, kernel_h, kernel_w)
# - Compute L2 norm for each output channel, then sum
```

**Total Loss**:
```
L_total = MSE(noise_pred, noise) + λ(stage) × Σ_c ||W[c,:,:,:]||_2
```

**λ Schedule (Cosine)**:
```
λ(i) = λ_max × 0.5 × (1 + cos(π × i / (num_stages - 1)))

Stage 1: λ = λ_max (strong regularization)
Stage 5: λ ≈ 0 (no regularization)
```

### CurriculumDDPM

DDPM with dynamic timestep range control.

```python
# Standard DDPM: t ~ Uniform(1, 1000)
# Curriculum DDPM: t ~ Uniform(t_min, t_max)

# Stage 1: t ∈ [0.8, 1.0] - High noise only
# Stage 3: t ∈ [0.4, 1.0] - Expanded range
# Stage 5: t ∈ [0.0, 1.0] - Full range
```

---

## Implementation Details

### Channel-level Sparsity (CS)

Location: `mindiffusion/sparse_unet.py`

```python
class SparseNaiveUnet(nn.Module):
    def __init__(self, ..., initial_sparsity=0.0):
        # Channel mask: 1 = active, 0 = masked
        self.register_buffer('channel_mask', torch.ones(256))

        # Record which stage each channel was activated
        self.register_buffer('channel_birth_stage', torch.zeros(256))

    def forward(self, x, t):
        ...
        thro = self.to_vec(down3)  # (B, 256, 1, 1)

        # Apply channel mask at bottleneck
        thro = thro * self.channel_mask.view(1, -1, 1, 1)

        thro = self.up0(thro + temb)
        ...
```

### Group L1 Regularization (CR)

Location: `mindiffusion/reg_unet.py`

```python
class RegNaiveUnet(NaiveUnet):
    def get_group_l1_penalty(self, lambda_val: float) -> torch.Tensor:
        """Compute Group L1 regularization penalty"""
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

**Key insight**: Even masked channels receive gradients during backprop!

```python
def regrow_channels(self, num_to_grow, current_stage, method="gradient"):
    if method == "gradient":
        # Select inactive channels with highest gradient magnitude
        grad_for_selection = self.get_channel_gradients()
        grad_for_selection[active_channels] = -inf  # Exclude already active
        _, topk_indices = torch.topk(grad_for_selection, num_to_grow)

    # Activate selected channels
    self.channel_mask[topk_indices] = 1
    self.channel_birth_stage[topk_indices] = current_stage
```

---

## Usage

### CelebA 32x32 Training

```bash
# Curriculum Only (C)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train_celeba_c_32.py

# Curriculum + Sparsity (CS) - Hard Mask
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train_celeba_cs_32.py

# Curriculum + Regularization (CR) - Group L1
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train_celeba_cr_32.py

# Multi-GPU
torchrun --nproc_per_node=4 train_celeba_cr_32.py

# Specify different port (avoid conflicts)
torchrun --nproc_per_node=1 --master_port=29501 train_celeba_cr_32.py
```

### Environment Setup

Set up `.env` file:
```
CELEBA_PATH=/path/to/celeba/dataset
```

---

## Configuration

### Current Config (32x32)

```python
# Common config
n_feat = 32
num_stages = 5
epochs_per_stage = [5, 10, 15, 20, 30]  # Total 80 epochs
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

# CS-specific config
sparsity_schedule = [0.80, 0.60, 0.40, 0.20, 0.00]

# CR-specific config
lambda_max = 0.00005
lambda_schedule = cosine_schedule(lambda_max, num_stages)
# → [0.00005, 0.0000427, 0.000025, 0.0000073, 0.0]
```

---

## Project Structure

```
minDiffusion_curriculum/
├── mindiffusion/
│   ├── __init__.py
│   ├── unet.py              # Original NaiveUnet
│   ├── sparse_unet.py       # SparseNaiveUnet (CS - hard mask)
│   ├── reg_unet.py          # RegNaiveUnet (CR - Group L1)
│   ├── ddpm.py              # Original DDPM
│   ├── curriculum_ddpm.py   # CurriculumDDPM (timestep range control)
│   └── ddim.py              # DDIM sampler
│
├── train_celeba_c_32.py     # Curriculum Only
├── train_celeba_cs_32.py    # Curriculum + Sparsity (Hard Mask)
├── train_celeba_cr_32.py    # Curriculum + Regularization (Group L1)
│
├── contents_c_32/           # C generated samples
├── contents_cs_32/          # CS generated samples
├── contents_cr_32/          # CR generated samples
│
├── checkpoints_c_32/        # C checkpoints
├── checkpoints_cs_32/       # CS checkpoints
└── checkpoints_cr_32/       # CR checkpoints
```

---

## Key Takeaways

1. **Curriculum Learning** provides a natural coarse-to-fine learning trajectory
2. **CS (Hard Mask)** directly limits capacity at bottleneck with explicit regrowth
3. **CR (Group L1)** soft-constrains all Conv layers, λ decay auto-releases
4. **Both sparsity methods** reserve clean model capacity for fine features

---

## References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Original minDiffusion](https://github.com/cloneofsimo/minDiffusion)
