# CIFAR-10 FID Evaluation

评估四种 U-ViT 模型在 CIFAR-10 上的 FID 分数。

## 模型对比

| 模型 | 说明 | Workdir Timestamp |
|------|------|-------------------|
| baseline | 标准训练 | 20260110_065028 |
| _c | Curriculum Learning | 20260110_125959 |
| _cs | Curriculum + Sparsity | 20260110_094434 |
| _s | Sparsity Only | 20260110_193200 |

## FID 结果

| Step | baseline | _c | _cs | _s |
|------|----------|-----|-----|-----|
| 20k | 393.0 | 438.7 | 415.3 | 380.9 |
| 40k | 219.4 | 215.1 | 323.5 | 402.0 |
| 60k | 362.1 | 127.3 | 133.0 | 119.3 |
| 80k | 135.6 | 70.6 | 74.7 | 35.0 |
| 100k | 34.7 | 32.4 | 27.8 | 27.0 |
| 120k | 21.2 | 20.7 | 17.5 | 23.0 |
| 140k | 16.7 | 17.5 | 16.3 | 20.9 |
| 160k | 15.1 | 16.0 | 15.3 | 20.8 |
| 180k | 14.1 | 15.2 | 14.3 | 17.7 |
| **200k** | **13.5** | 14.2 | **13.5** | 15.7 |

## 使用方法

### 运行评估

```bash
# 评估单个模型
bash scripts/eval_cifar10_baseline.sh
bash scripts/eval_cifar10_c.sh
bash scripts/eval_cifar10_cs.sh
bash scripts/eval_cifar10_s.sh

# 或运行全部
bash scripts/eval_cifar10_baseline.sh && \
bash scripts/eval_cifar10_c.sh && \
bash scripts/eval_cifar10_cs.sh && \
bash scripts/eval_cifar10_s.sh
```

### 绘制对比图

```bash
python plot_fid.py --eval-results ../eval_results --output outputs/cifar10_fid_comparison.png
```

## 目录结构

```
cifar10_fid_evaluation/
├── README.md
├── plot_fid.py              # 绘图脚本
├── scripts/
│   ├── eval_cifar10_baseline.sh
│   ├── eval_cifar10_c.sh
│   ├── eval_cifar10_cs.sh
│   └── eval_cifar10_s.sh
└── outputs/
    └── cifar10_fid_comparison.png
```

## 评估配置

- **n_samples**: 5000
- **mini_batch_size**: 1000
- **sample_steps**: 50
- **algorithm**: dpm_solver
- **checkpoints**: 20k, 40k, 60k, 80k, 100k, 120k, 140k, 160k, 180k, 200k

## 结论

- **baseline** 和 **_cs (curriculum+sparsity)** 最终表现最好，FID=13.5
- **_c (curriculum only)** 略差，FID=14.2
- **_s (sparsity only)** 收敛较慢，FID=15.7
