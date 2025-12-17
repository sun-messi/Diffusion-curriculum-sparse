"""
Configuration for DDPM Training with Curriculum Learning + Sparsity Curriculum
CS = Curriculum + Sparsity

基于 config.py 扩展，添加课程学习和稀疏度课程参数
"""


class ConfigCS:
    # ==================== Model ====================
    n_feat: int = 128           # 原版: n_feat=128 (NaiveUnet hidden dim)
    n_T: int = 1000             # 原版: n_T=1000 (扩散步数)
    beta1: float = 1e-4         # 原版: betas=(1e-4, 0.02)
    beta2: float = 0.02         # 原版: betas=(1e-4, 0.02)

    # ==================== Data ====================
    dataset: str = "cifar10"    # 原版: CIFAR10
    data_dir: str = "./data"    # 原版: "./data"
    image_size: int = 32        # 原版: CIFAR10 32x32
    in_channels: int = 3        # 原版: RGB 3通道

    # ==================== Training ====================
    global_batch_size: int = 512  # 原版: batch_size=512 (总batch size)
    lr: float = 1e-5            # 原版: lr=1e-5

    # Learning rate scaling for DDP
    lr_scaling: str = "none"    # "none"/"linear"/"sqrt"

    # ==================== DDP ====================
    num_workers: int = 4        # DDP 建议 4
    pin_memory: bool = True     # 加速数据传输

    # ==================== Saving ====================
    save_dir: str = "./contents_cs"       # 课程学习结果目录
    ckpt_dir: str = "./checkpoints_cs"    # checkpoint 保存目录
    model_path: str = "./ddpm_cifar_cs_latest.pth"  # 最新模型路径
    sample_every_stage: int = 1   # 每 N 个 stage 采样一次
    save_every_stage: int = 2     # 每 N 个 stage 保存 checkpoint
    n_sample: int = 8             # 采样 8 张图

    # ==================== Device (单卡模式) ====================
    device: str = "cuda:0"

    # ==================== Curriculum Learning ====================
    curriculum_enabled: bool = True
    num_curriculum_stages: int = 6       # 课程阶段数
    # 每个 stage 的 epochs 数 (长度必须等于 num_curriculum_stages)
    # 高噪声少训练，低噪声多训练
    epochs_schedule: list = None          # 默认 None，在 __init__ 中设置默认值
    # 默认值: [2, 2, 2, 5, 5, 5, 10, 10, 10, 49] = 100 epochs

    # ==================== Sparsity Curriculum ====================
    sparsity_enabled: bool = True
    initial_sparsity: float = 0.80        # 初始稀疏度 80% (只有 20% channels 活跃)
    final_sparsity: float = 0.00          # 最终稀疏度 0% (所有 channels 活跃)
    regrowth_method: str = "random"       # "gradient" or "random"
    max_accumulated_stages: int = 5       # curriculum 累积的最大 stage 数
                                          # 限制时间范围累积，只保留最近 N 个 stage 的范围
                                          # 例如: stage 7 时，t_start 基于 stage 2 而非 stage 1

    # Auto-generated (在 __init__ 中生成)
    curriculum_stages: list = None        # [(t_start, t_end, name), ...]
    sparsity_schedule: list = None        # [0.50, 0.45, 0.40, ...]

    def __init__(self, **kwargs):
        """允许通过 kwargs 覆盖默认配置"""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"Unknown config key: {k}")

        # 自动生成 curriculum_stages
        # 从高噪声 (t≈1) 逐步扩展到全范围 (t∈[0,1])
        # Stage 1: t ∈ [0.9, 1.0] - 纯噪声，最简单
        # Stage 2: t ∈ [0.8, 1.0]
        # ...
        # Stage 10: t ∈ [0.0, 1.0] - 完整范围
        self.curriculum_stages = []
        for i in range(self.num_curriculum_stages):
            # t_start 从 0.9 递减到 0.0
            t_start = 1.0 - (i + 1) * (1.0 / self.num_curriculum_stages)
            t_end = 1.0
            stage_name = f"stage_{i+1}_t[{t_start:.1f}-{t_end:.1f}]"
            self.curriculum_stages.append((t_start, t_end, stage_name))

        # 自动生成 sparsity_schedule (与 curriculum_stages 对齐)
        # 从 initial_sparsity 线性递减到 final_sparsity
        if self.sparsity_enabled:
            sparsity_step = (self.initial_sparsity - self.final_sparsity) / self.num_curriculum_stages
            self.sparsity_schedule = [
                self.initial_sparsity - i * sparsity_step
                for i in range(self.num_curriculum_stages)
            ]
        else:
            self.sparsity_schedule = [0.0] * self.num_curriculum_stages

        # epochs_schedule: 如果用户没有提供，使用默认值
        if self.epochs_schedule is None:
            # 默认: [2, 2, 2, 5, 5, 5, 10, 10, 10, 49] = 100 epochs
            self.epochs_schedule = [5, 5, 8, 8, 8, 20]

    def get_batch_size(self, world_size: int = 1) -> int:
        """计算每个 GPU 的 batch size"""
        return self.global_batch_size // world_size

    def get_lr(self, world_size: int = 1) -> float:
        """根据 lr_scaling 策略计算实际学习率"""
        if self.lr_scaling == "none":
            return self.lr
        elif self.lr_scaling == "linear":
            return self.lr * world_size
        elif self.lr_scaling == "sqrt":
            return self.lr * (world_size ** 0.5)
        else:
            raise ValueError(f"Unknown lr_scaling: {self.lr_scaling}")

    def get_total_epochs(self) -> int:
        """总 epoch 数"""
        return sum(self.epochs_schedule)

    def print_config(self, world_size: int = 1):
        """打印当前配置"""
        print("=" * 60)
        print("ConfigCS (Curriculum + Sparsity):")
        print("=" * 60)
        print(f"  Model: n_feat={self.n_feat}, n_T={self.n_T}")
        print(f"  Betas: ({self.beta1}, {self.beta2})")
        print(f"  Data: {self.dataset} {self.image_size}x{self.image_size}")
        print(f"  Global batch size: {self.global_batch_size}")
        print(f"  Per-GPU batch size: {self.get_batch_size(world_size)}")
        print(f"  Learning rate: {self.get_lr(world_size)} (scaling: {self.lr_scaling})")
        print("-" * 60)
        print("  Curriculum Learning:")
        print(f"    Stages: {self.num_curriculum_stages}")
        print(f"    Epochs schedule: {self.epochs_schedule}")
        print(f"    Total epochs: {self.get_total_epochs()}")
        print("-" * 60)
        print("  Sparsity Curriculum:")
        print(f"    Enabled: {self.sparsity_enabled}")
        print(f"    Initial sparsity: {self.initial_sparsity:.0%}")
        print(f"    Final sparsity: {self.final_sparsity:.0%}")
        print(f"    Regrowth method: {self.regrowth_method}")
        print(f"    Max accumulated stages: {self.max_accumulated_stages}")
        print(f"    Schedule: {[f'{s:.0%}' for s in self.sparsity_schedule]}")
        print("-" * 60)
        print("  Curriculum Stages:")
        for i, (t_start, t_end, name) in enumerate(self.curriculum_stages):
            sparsity = self.sparsity_schedule[i]
            epochs = self.epochs_schedule[i]
            print(f"    {i+1:2d}. t ∈ [{t_start:.1f}, {t_end:.1f}] | sparsity: {sparsity:.0%} | epochs: {epochs}")
        print("=" * 60)

    def __repr__(self):
        lines = ["ConfigCS("]
        for k in dir(self):
            if not k.startswith("_") and not callable(getattr(self, k)):
                v = getattr(self, k)
                lines.append(f"  {k}={v!r},")
        lines.append(")")
        return "\n".join(lines)


# 默认配置实例
config_cs = ConfigCS()
