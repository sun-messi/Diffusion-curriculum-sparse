# CIFAR-10 Feature Comparison for U-ViT

Compare bottleneck features between **Baseline**, **CS (Curriculum + Sparsity)**, and **C (Curriculum only)** U-ViT models using Silhouette Score.

## Models

| Model | Description | Directory |
|-------|-------------|-----------|
| Baseline | Standard U-ViT training | `cifar10_uvit_small/` |
| CS Mode | Curriculum + Sparsity | `cifar10_uvit_small_cs/` |
| C Mode | Curriculum only | `cifar10_uvit_small_c/` |

## CIFAR-10 Classes (10 Categories)

| Label | Class |
|-------|-------|
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

## Usage

### 1. Run Silhouette Score Comparison

```bash
cd tsne_cifar10_comparison
python run_comparison.py
```

Output: `outputs/silhouette_cifar10.csv`

### 2. Plot Results

```bash
python plot_silhouette.py
```

Output: `outputs/silhouette_cifar10.png`

## Output Format

`outputs/silhouette_cifar10.csv`:
```
Step,Baseline,CS_Mode,C_Mode
20000,0.xxx,0.xxx,0.xxx
40000,0.xxx,0.xxx,0.xxx
...
```

## Difference from CelebA Experiment

| Item | CelebA | CIFAR-10 |
|------|--------|----------|
| Classification | 12 binary attributes | 10 classes |
| img_size | 64 | 32 |
| patch_size | 4 | 2 |
| Dataset | CelebA faces | CIFAR-10 objects |

## Directory Structure

```
tsne_cifar10_comparison/
├── README.md
├── run_comparison.py      # Main computation script
├── plot_silhouette.py     # Plotting script
├── tsne_utils/
│   ├── __init__.py
│   └── feature_extractor.py
└── outputs/
    ├── silhouette_cifar10.csv
    └── silhouette_cifar10.png
```
