"""
Configuration for DDPM Training
统一配置文件，方便调整超参数

原版代码参考: train_cifar10.py
"""


class Config:
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
    n_epoch: int = 100          # 原版: n_epoch=100
    global_batch_size: int = 512  # 原版: batch_size=512 (总batch size)
    lr: float = 1e-5            # 原版: lr=1e-5

    # Learning rate scaling for DDP
    # "none": 不缩放，保持 lr 不变
    # "linear": lr = base_lr * world_size (大batch时使用)
    # "sqrt": lr = base_lr * sqrt(world_size)
    lr_scaling: str = "none"    # 原版: 无缩放

    # ==================== DDP ====================
    num_workers: int = 4        # 原版: num_workers=16 (单卡), DDP建议减少到4
    pin_memory: bool = True     # 加速数据传输

    # ==================== Saving ====================
    save_dir: str = "./contents"          # 原版: "./contents"
    ckpt_dir: str = "./checkpoints"       # checkpoint保存目录
    model_path: str = "./ddpm_cifar_latest.pth"  # 最新模型路径 (每epoch覆盖)
    sample_every: int = 1       # 每N个epoch采样一次
    save_every: int = 20        # 每N个epoch保存一个checkpoint (带epoch编号)
    n_sample: int = 8           # 原版: ddpm.sample(8, ...) 采样8张图

    # ==================== Device (单卡模式) ====================
    device: str = "cuda:0"      # 原版: device="cuda:1"

    def __init__(self, **kwargs):
        """允许通过 kwargs 覆盖默认配置"""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"Unknown config key: {k}")

    def get_batch_size(self, world_size: int = 1) -> int:
        """
        计算每个 GPU 的 batch size

        DDP 模式: global_batch_size 分配到各卡
        单卡模式: world_size=1, 返回 global_batch_size
        """
        return self.global_batch_size // world_size

    def get_lr(self, world_size: int = 1) -> float:
        """
        根据 lr_scaling 策略计算实际学习率

        - none: 保持原 lr (推荐: 当 global_batch_size 不变时)
        - linear: lr * world_size (当每卡 batch 不变，总 batch 变大时)
        - sqrt: lr * sqrt(world_size)
        """
        if self.lr_scaling == "none":
            return self.lr
        elif self.lr_scaling == "linear":
            return self.lr * world_size
        elif self.lr_scaling == "sqrt":
            return self.lr * (world_size ** 0.5)
        else:
            raise ValueError(f"Unknown lr_scaling: {self.lr_scaling}")

    def print_config(self, world_size: int = 1):
        """打印当前配置"""
        print("=" * 50)
        print("Config:")
        print("=" * 50)
        print(f"  Model: n_feat={self.n_feat}, n_T={self.n_T}")
        print(f"  Betas: ({self.beta1}, {self.beta2})")
        print(f"  Data: {self.dataset} {self.image_size}x{self.image_size}")
        print(f"  Training: {self.n_epoch} epochs")
        print(f"  Global batch size: {self.global_batch_size}")
        print(f"  Per-GPU batch size: {self.get_batch_size(world_size)}")
        print(f"  Learning rate: {self.get_lr(world_size)} (scaling: {self.lr_scaling})")
        print(f"  Num workers: {self.num_workers}")
        print(f"  Save dir: {self.save_dir}")
        print("=" * 50)

    def __repr__(self):
        lines = ["Config("]
        for k in dir(self):
            if not k.startswith("_") and not callable(getattr(self, k)):
                v = getattr(self, k)
                lines.append(f"  {k}={v!r},")
        lines.append(")")
        return "\n".join(lines)


# 默认配置实例
config = Config()
