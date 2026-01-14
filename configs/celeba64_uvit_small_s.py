import ml_collections

# nohup accelerate launch --multi_gpu --num_processes 6 --mixed_precision fp16 train_c.py --config=configs/celeba64_uvit_small_s.py > training.log 2>&1 &


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'
    config.sparsity_enabled = True

    # === Sparsity-Only Curriculum Learning Configuration ===
    # t_min=0.0, t_max=1.0 stays constant (full noise range)
    # Only sparsity changes progressively
    config.curriculum = d(
        enabled=True,
        stages=[
            # Stage 1: High sparsity
            d(t_min=0.0, t_max=1.0, n_steps=10000, sparsity=0.15, name="stage1_sparse"),
            # Stage 2
            d(t_min=0.0, t_max=1.0, n_steps=10000, sparsity=0.15, name="stage2_sparse"),
            # Stage 3
            d(t_min=0.0, t_max=1.0, n_steps=20000, sparsity=0.15, name="stage3_sparse"),
            # Stage 4
            d(t_min=0.0, t_max=1.0, n_steps=10000, sparsity=0.13, name="stage4_medium"),
            # Stage 5
            d(t_min=0.0, t_max=1.0, n_steps=10000, sparsity=0.10, name="stage5_medium"),
            # Stage 6
            d(t_min=0.0, t_max=1.0, n_steps=20000, sparsity=0.08, name="stage6_less"),
            # Stage 7
            d(t_min=0.0, t_max=1.0, n_steps=20000, sparsity=0.05, name="stage7_less"),
            # Stage 8: No sparsity
            d(t_min=0.0, t_max=1.0, n_steps=20000, sparsity=0.0, name="stage8_full"),
            # Stage 9
            d(t_min=0.0, t_max=1.0, n_steps=20000, sparsity=0.0, name="stage9_full"),
            # Stage 10
            d(t_min=0.0, t_max=1.0, n_steps=60000, sparsity=0.0, name="stage10_full"),

        ]
    )

    # Total training steps = sum of all stage n_steps
    total_steps = sum(s['n_steps'] for s in config.curriculum.stages)

    config.train = d(
        n_steps=total_steps,  # 200000
        batch_size=126*2,  # 126 = 6 GPU × 21 per GPU
        mode='uncond',
        log_interval=100,
        eval_interval=5000,
        save_interval=10000,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.99, 0.99),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    config.nnet = d(
        name='uvit',
        img_size=64,
        patch_size=4,
        embed_dim=256,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=-1,
    )

    config.dataset = d(
        name='celeba',
        path='assets/datasets/celeba',
        resolution=64,
    )

    config.sample = d(
        sample_steps=100,  # ODE sampling steps
        n_samples=2000,  # Number of samples for FID calculation
        mini_batch_size=500,  # Batch size per GPU
        algorithm='dpm_solver',  # Use ODE sampling
        path=''
    )

    return config
