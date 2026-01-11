import ml_collections

# accelerate launch --multi_gpu --num_processes 6 --mixed_precision fp16 train.py --config=configs/cifar10_uvit_small.py

def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'

    config.train = d(
        n_steps=200000,
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
        patch_size=4,
        embed_dim=128,
        depth=10,
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
