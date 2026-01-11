import ml_collections

# nohup accelerate launch --multi_gpu --num_processes 6 --mixed_precision fp16 train_c.py --config=configs/cifar10_uvit_small_s.py > cifar10_s_training.log 2>&1 &


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'
    config.sparsity_enabled = True

    # === Sparsity-Only Configuration ===
    # t_min=0.0, t_max=1.0 stays constant (full noise range)
    # Only sparsity changes progressively
    config.curriculum = d(
        enabled=True,
        stages=[
            d(t_min=0.0, t_max=1.0, n_steps=10000, sparsity=0.35, name="stage1_sparse"),
            d(t_min=0.0, t_max=1.0, n_steps=10000, sparsity=0.25, name="stage2_sparse"),
            d(t_min=0.0, t_max=1.0, n_steps=20000, sparsity=0.15, name="stage3_sparse"),
            d(t_min=0.0, t_max=1.0, n_steps=10000, sparsity=0.05, name="stage4_medium"),
            d(t_min=0.0, t_max=1.0, n_steps=10000, sparsity=0.0, name="stage5_medium"),
            d(t_min=0.0, t_max=1.0, n_steps=20000, sparsity=0.0, name="stage6_less"),
            d(t_min=0.0, t_max=1.0, n_steps=20000, sparsity=0.0, name="stage7_less"),
            d(t_min=0.0, t_max=1.0, n_steps=20000, sparsity=0.0, name="stage8_full"),
            d(t_min=0.0, t_max=1.0, n_steps=20000, sparsity=0.0, name="stage9_full"),
            d(t_min=0.0, t_max=1.0, n_steps=60000, sparsity=0.0, name="stage10_full"),
        ]
    )

    # Total training steps = sum of all stage n_steps
    total_steps = sum(s['n_steps'] for s in config.curriculum.stages)

    config.train = d(
        n_steps=total_steps,  # 200000
        batch_size=126,  # 126 % 6 = 0 for multi-GPU
        mode='uncond',
        log_interval=10,
        eval_interval=5000,
        save_interval=10000,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.99, 0.999),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=2500
    )

    config.nnet = d(
        name='uvit',
        img_size=32,
        patch_size=2,
        embed_dim=256,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=-1,
    )

    config.dataset = d(
        name='cifar10',
        path='assets/datasets/cifar10',
        random_flip=True,
    )

    config.sample = d(
        sample_steps=50,
        n_samples=1000,
        mini_batch_size=500,
        algorithm='dpm_solver',
        path=''
    )

    return config
