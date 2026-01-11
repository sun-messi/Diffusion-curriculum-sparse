# t-SNE Feature Comparison for U-ViT

Compare bottleneck features between **Baseline**, **CS (Curriculum + Sparsity)**, and **C (Curriculum only)** U-ViT models using Silhouette Score and t-SNE visualization.

## Models

| Model | Description | Directory |
|-------|-------------|-----------|
| Baseline | Standard U-ViT training | `celeba64_uvit_small/` |
| CS Mode | Curriculum + Sparsity | `celeba64_uvit_small_cs/` |
| C Mode | Curriculum only | `celeba64_uvit_small_c/` |

## Binary Classifications (12 Attributes)

| Classification | Categories | CelebA Index |
|---------------|------------|--------------|
| gender | Female vs Male | 20 |
| age | Old vs Young | 39 |
| smiling | Not Smiling vs Smiling | 31 |
| eyeglasses | No Glasses vs Glasses | 15 |
| attractive | Not Attractive vs Attractive | 2 |
| chubby | Not Chubby vs Chubby | 13 |
| heavy_makeup | Light Makeup vs Heavy Makeup | 18 |
| bald | Hair vs Bald | 4 |
| bangs | No Bangs vs Bangs | 5 |
| big_nose | Small Nose vs Big Nose | 7 |
| high_cheekbones | Low Cheekbones vs High Cheekbones | 19 |

## Usage

### 1. Run Full Binary Comparison (GPU Batch Processing)

Compute Silhouette Scores for all 12 binary classifications across all models and steps:

```bash
cd tsne_feature_comparison
python run_binary_comparison.py
```

Output: `outputs/silhouette_binary_all.csv`

Format:
```
Classification,Categories,Step,Baseline,CS_Mode,C_Mode
gender,Female vs Male,20000,0.029551,0.037082,0.050431
gender,Female vs Male,40000,0.034124,0.042056,0.072512
...
```

### 2. Plot Average Silhouette Scores

**All 12 classifications:**
```bash
python plot_average_silhouette.py
```
Output: `outputs/silhouette_average_by_step.png`

**High-level semantic features (middle-peak pattern):**
```bash
python plot_middle_peak_average.py
```
Output: `outputs/silhouette_middle_peak_average.png`

Selected: gender, smiling, attractive, heavy_makeup, bangs, high_cheekbones

These classifications show an inverted-U pattern: peak at 40k-60k, lower at 20k and 200k.

**Selected similar-magnitude features:**
```bash
python plot_selected_average.py
```
Output: `outputs/silhouette_selected_average.png`

### 3. t-SNE Visualization

Compare final checkpoints visually:

```bash
python tsne_comparison.py \
    --baseline_dir ../workdir/celeba64_uvit_small/default_20260101_030900 \
    --cs_dir ../workdir/celeba64_uvit_small_cs/default_20260101_073037 \
    --step 200000 \
    --samples_per_class 100
```

### 4. Single Model Analysis

```bash
python compute_single.py \
    --ckpt_dir ../workdir/celeba64_uvit_small/default_20260101_030900 \
    --step 200000 \
    --classification gender
```

## Key Findings

### Training Dynamics

| Step | Observation |
|------|-------------|
| 20k | C_Mode starts learning semantic features faster |
| 40k-60k | CS/C modes peak for high-level features (gender, smiling, etc.) |
| 100k+ | All models converge to similar performance |
| 200k | Final scores nearly identical across models |

### Feature Type Patterns

**Middle-peak (high-level semantic):** gender, smiling, attractive, heavy_makeup, bangs, high_cheekbones
- Peak at 40k-60k, then decrease
- CS/C modes show faster early learning

**Early-peak (low-level/rare):** bald, chubby, eyeglasses, age, big_nose
- Peak at 20k, then decrease
- These features are learned very early or are rare in dataset

## Silhouette Score Interpretation

| Score Range | Quality |
|-------------|---------|
| 0.7 ~ 1.0 | Excellent clustering |
| 0.5 ~ 0.7 | Good clustering |
| 0.25 ~ 0.5 | Fair clustering |
| < 0.25 | Poor clustering |

Note: Scores in this analysis are typically 0.01-0.07 (cosine metric), indicating weak but detectable clustering in the learned feature space.

## Directory Structure

```
tsne_feature_comparison/
├── README.md                    # This file
├── run_binary_comparison.py     # Main batch processing script (GPU)
├── compute_single.py            # Single checkpoint analysis
├── compute_silhouette.py        # Legacy single model analysis
├── tsne_comparison.py           # t-SNE visualization
├── silhouette_by_epoch.py       # Training progress tracking
├── plot_average_silhouette.py   # Plot all 12 classifications average
├── plot_middle_peak_average.py  # Plot high-level semantic features
├── plot_selected_average.py     # Plot selected features
├── plot_small_diff_average.py   # Plot minimal C-CS difference
├── tsne_utils/
│   ├── __init__.py
│   └── feature_extractor.py     # PyTorch forward hook extractor
└── outputs/
    ├── silhouette_binary_all.csv           # Main results
    ├── silhouette_average_by_step.png      # All avg plot
    ├── silhouette_middle_peak_average.png  # High-level semantic plot
    └── *.csv                               # Various CSV outputs
```

## U-ViT Feature Layers

| Layer | Module Path | Output Shape |
|-------|-------------|--------------|
| in_block_0 | in_blocks.0 | (B, 257, 256) |
| in_block_5 | in_blocks.5 | (B, 257, 256) |
| mid_block | mid_block | (B, 257, 256) |
| out_block_0 | out_blocks.0 | (B, 257, 256) |
| out_block_5 | out_blocks.5 | (B, 257, 256) |

Note: First token is the time token, remaining 256 tokens are patch embeddings.
