#!/usr/bin/env python3
"""
t-SNE Feature Space Visualization for U-ViT

Compare bottleneck features between Baseline and CS (Curriculum+Sparsity) U-ViT models
using t-SNE dimensionality reduction on CelebA dataset.

Classification: Gender x Age (4 classes)
- Old Female (Male=0, Young=0)
- Young Female (Male=0, Young=1)
- Old Male (Male=1, Young=0)
- Young Male (Male=1, Young=1)
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from torchvision import datasets as tvds, transforms
from libs.uvit import UViT
from tsne_utils.feature_extractor import FeatureExtractor


# CelebA attribute indices
MALE_IDX = 20
YOUNG_IDX = 39

# Class names for Gender x Age classification
CLASS_NAMES = [
    'Old Female',    # Male=0, Young=0 -> class 0
    'Young Female',  # Male=0, Young=1 -> class 1
    'Old Male',      # Male=1, Young=0 -> class 2
    'Young Male',    # Male=1, Young=1 -> class 3
]

# Colors for 4 classes
CLASS_COLORS = plt.cm.tab10(np.array([0, 1, 2, 3]))


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='t-SNE visualization comparing Baseline vs CS U-ViT model features'
    )

    parser.add_argument('--baseline_dir', type=str,
                        default='../workdir/celeba64_uvit_small/default_20260101_030900',
                        help='Path to baseline checkpoint directory')
    parser.add_argument('--cs_dir', type=str,
                        default='../workdir/celeba64_uvit_small_cs/default_20260101_073037',
                        help='Path to CS mode checkpoint directory')
    parser.add_argument('--step', type=int, default=200000,
                        help='Training step to load (default: 200000)')
    parser.add_argument('--timestep', type=int, default=0,
                        help='Diffusion timestep (0=clean, default: 0)')
    parser.add_argument('--samples_per_class', type=int, default=100,
                        help='Number of samples per class (default: 100)')
    parser.add_argument('--perplexity', type=float, default=30,
                        help='t-SNE perplexity (default: 30)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (default: cuda:0)')
    parser.add_argument('--output', type=str, default='outputs/tsne_baseline_vs_cs.png',
                        help='Output file path')
    parser.add_argument('--celeba_path', type=str, default='../assets/datasets/celeba',
                        help='Path to CelebA dataset')
    parser.add_argument('--layer', type=str, default='mid_block',
                        help='Layer to extract features from (default: mid_block)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for feature extraction')
    parser.add_argument('--csv_output', type=str, default='outputs/tsne_comparison.csv',
                        help='Output CSV path')

    return parser.parse_args()


def get_uvit_config():
    """Get U-ViT model configuration for CelebA 64x64."""
    return {
        'img_size': 64,
        'patch_size': 4,
        'embed_dim': 256,
        'depth': 12,
        'num_heads': 8,
        'mlp_ratio': 4,
        'qkv_bias': False,
        'mlp_time_embed': False,
        'num_classes': -1,
    }


def load_uvit_model(ckpt_dir: str, step: int, device: torch.device) -> nn.Module:
    """
    Load U-ViT model from checkpoint.

    Args:
        ckpt_dir: Checkpoint directory path
        step: Training step
        device: Compute device

    Returns:
        Loaded model in eval mode
    """
    print(f"Loading model from: {ckpt_dir}")

    config = get_uvit_config()
    model = UViT(**config)

    # Load EMA weights
    ckpt_path = os.path.join(ckpt_dir, 'ckpts', f'{step}.ckpt', 'nnet_ema.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    print(f"  Loaded EMA weights from step {step}")

    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model


class Crop(object):
    """Crop transform for CelebA."""
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return transforms.functional.crop(
            img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1
        )


def get_celeba_with_attrs(root: str = '~/datasets'):
    """
    Load CelebA dataset with attribute labels.

    Returns:
        Dataset that yields (image, attrs) tuples
    """
    root = os.path.expanduser(root)

    # CelebA center crop parameters (same as in datasets.py)
    cx = 89
    cy = 121
    x1 = cy - 64  # 57
    x2 = cy + 64  # 185
    y1 = cx - 64  # 25
    y2 = cx + 64  # 153

    transform = transforms.Compose([
        Crop(x1, x2, y1, y2),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    dataset = tvds.CelebA(
        root=root,
        split='test',
        target_type='attr',
        transform=transform,
        download=False
    )

    return dataset


def get_class_label(attrs: torch.Tensor) -> int:
    """
    Convert CelebA attributes to class label.

    Args:
        attrs: Tensor of 40 binary attributes

    Returns:
        Class label (0-3):
        - 0: Old Female
        - 1: Young Female
        - 2: Old Male
        - 3: Young Male
    """
    male = int(attrs[MALE_IDX].item())
    young = int(attrs[YOUNG_IDX].item())
    return male * 2 + young


def sample_balanced_data(
    dataset,
    samples_per_class: int,
    seed: int = 42
) -> tuple:
    """
    Sample balanced data from CelebA dataset.

    Args:
        dataset: CelebA dataset with attributes
        samples_per_class: Number of samples per class
        seed: Random seed

    Returns:
        (images, labels) tensors
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_classes = 4
    class_samples = {i: [] for i in range(num_classes)}
    class_counts = {i: 0 for i in range(num_classes)}

    print(f"Sampling {samples_per_class} images per class from {num_classes} classes:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  Class {i}: {name}")

    # Shuffle indices
    indices = np.random.permutation(len(dataset))

    for idx in tqdm(indices, desc="Sampling"):
        img, attrs = dataset[idx]
        label = get_class_label(attrs)

        if class_counts[label] < samples_per_class:
            class_samples[label].append(img)
            class_counts[label] += 1

        # Check if we have enough samples
        if all(c >= samples_per_class for c in class_counts.values()):
            break

    # Check if we have enough samples
    for i in range(num_classes):
        if class_counts[i] < samples_per_class:
            print(f"  Warning: Only found {class_counts[i]} samples for class {i} ({CLASS_NAMES[i]})")

    # Stack samples
    all_images = []
    all_labels = []
    for label in range(num_classes):
        all_images.extend(class_samples[label][:samples_per_class])
        all_labels.extend([label] * len(class_samples[label][:samples_per_class]))

    images = torch.stack(all_images)
    labels = torch.tensor(all_labels)

    print(f"  Sampled {len(images)} images total")
    for i in range(num_classes):
        count = (labels == i).sum().item()
        print(f"    Class {i} ({CLASS_NAMES[i]}): {count}")

    return images, labels


def extract_features(
    model: nn.Module,
    images: torch.Tensor,
    timestep: int,
    device: torch.device,
    layer: str = 'mid_block',
    batch_size: int = 64
) -> np.ndarray:
    """
    Extract features from U-ViT model.

    Args:
        model: U-ViT model
        images: Input images [N, C, H, W]
        timestep: Diffusion timestep
        device: Compute device
        layer: Layer to extract features from
        batch_size: Batch size for processing

    Returns:
        Features array [N, D]
    """
    model.eval()
    all_features = []

    num_batches = (len(images) + batch_size - 1) // batch_size

    with FeatureExtractor(model) as extractor:
        extractor.register_layer("target", layer)

        for i in tqdm(range(num_batches), desc=f"Extracting features ({layer})"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(images))
            batch_images = images[start_idx:end_idx].to(device)

            # Create timestep tensor
            t = torch.full((len(batch_images),), timestep, dtype=torch.long, device=device)

            # Extract features
            features = extractor.extract_features(batch_images, t)

            # Get target layer features
            feat = features["target"]  # Shape: [B, num_tokens, embed_dim]

            # Remove time token (first token) and flatten
            if feat.dim() == 3:
                feat = feat[:, 1:, :]  # [B, 256, 256] for mid_block
                feat = feat.flatten(start_dim=1)  # [B, 65536]

            all_features.append(feat.numpy())

            # Clear features for next batch
            extractor.clear()

    return np.concatenate(all_features, axis=0)


def plot_tsne_comparison(
    embeddings_baseline: np.ndarray,
    embeddings_cs: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    timestep: int,
    sil_baseline: float = None,
    sil_cs: float = None
):
    """
    Create side-by-side t-SNE visualization.

    Args:
        embeddings_baseline: 2D embeddings for baseline [N, 2]
        embeddings_cs: 2D embeddings for CS mode [N, 2]
        labels: Class labels [N]
        output_path: Output file path
        timestep: Timestep used (for title)
        sil_baseline: Silhouette score for baseline
        sil_cs: Silhouette score for CS mode
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot baseline
    ax = axes[0]
    for i in range(4):
        mask = labels == i
        ax.scatter(
            embeddings_baseline[mask, 0],
            embeddings_baseline[mask, 1],
            c=[CLASS_COLORS[i]],
            label=CLASS_NAMES[i],
            alpha=0.6,
            s=15
        )
    title_baseline = '(a) Baseline'
    if sil_baseline is not None:
        title_baseline += f'\nSilhouette: {sil_baseline:.3f}'
    ax.set_title(title_baseline, fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot CS mode
    ax = axes[1]
    for i in range(4):
        mask = labels == i
        ax.scatter(
            embeddings_cs[mask, 0],
            embeddings_cs[mask, 1],
            c=[CLASS_COLORS[i]],
            label=CLASS_NAMES[i],
            alpha=0.6,
            s=15
        )
    title_cs = '(b) CS Mode (Curriculum + Sparsity)'
    if sil_cs is not None:
        title_cs += f'\nSilhouette: {sil_cs:.3f}'
    ax.set_title(title_cs, fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    # Add legend
    handles, labels_legend = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='center right', bbox_to_anchor=(0.99, 0.5),
               fontsize=9, title='Classes')

    # Adjust layout
    plt.suptitle(f't-SNE Visualization of U-ViT Features (t={timestep})',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.close()


def main():
    args = parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load models
    print("\n=== Loading Baseline Model ===")
    model_baseline = load_uvit_model(args.baseline_dir, args.step, device)

    print("\n=== Loading CS Model ===")
    model_cs = load_uvit_model(args.cs_dir, args.step, device)

    # Load CelebA test data with attributes
    print("\n=== Loading CelebA Test Data ===")
    dataset = get_celeba_with_attrs(args.celeba_path)

    # Sample balanced data
    images, labels = sample_balanced_data(dataset, args.samples_per_class, args.seed)

    # Extract features
    print(f"\n=== Extracting Baseline Features (t={args.timestep}, layer={args.layer}) ===")
    features_baseline = extract_features(
        model_baseline, images, args.timestep, device, args.layer, args.batch_size
    )
    print(f"  Feature shape: {features_baseline.shape}")

    print(f"\n=== Extracting CS Features (t={args.timestep}, layer={args.layer}) ===")
    features_cs = extract_features(
        model_cs, images, args.timestep, device, args.layer, args.batch_size
    )
    print(f"  Feature shape: {features_cs.shape}")

    # Compute Silhouette Scores
    print("\n=== Computing Silhouette Scores ===")
    labels_np = labels.numpy()
    sil_baseline = silhouette_score(features_baseline, labels_np, metric='cosine')
    sil_cs = silhouette_score(features_cs, labels_np, metric='cosine')
    print(f"  Baseline Silhouette Score: {sil_baseline:.4f}")
    print(f"  CS Mode  Silhouette Score: {sil_cs:.4f}")
    print(f"  Improvement: {sil_cs - sil_baseline:+.4f}")

    # Save CSV
    df = pd.DataFrame({
        'Model': ['Baseline', 'CS_Mode'],
        'Layer': [args.layer, args.layer],
        'Timestep': [args.timestep, args.timestep],
        'Samples_Per_Class': [args.samples_per_class, args.samples_per_class],
        'Total_Samples': [len(labels), len(labels)],
        'Feature_Dim': [features_baseline.shape[1], features_cs.shape[1]],
        'Silhouette_Score': [sil_baseline, sil_cs],
    })
    os.makedirs(os.path.dirname(args.csv_output) if os.path.dirname(args.csv_output) else '.', exist_ok=True)
    df.to_csv(args.csv_output, index=False)
    print(f"  Saved CSV to: {args.csv_output}")

    # Run t-SNE
    print("\n=== Running t-SNE ===")
    all_features = np.concatenate([features_baseline, features_cs], axis=0)
    print(f"  Combined features: {all_features.shape}")

    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        random_state=args.seed,
        n_iter=1000,
        verbose=1
    )
    embeddings = tsne.fit_transform(all_features)

    # Split embeddings
    n_samples = len(labels)
    embeddings_baseline = embeddings[:n_samples]
    embeddings_cs = embeddings[n_samples:]

    # Plot
    print("\n=== Creating Visualization ===")
    plot_tsne_comparison(
        embeddings_baseline,
        embeddings_cs,
        labels_np,
        args.output,
        args.timestep,
        sil_baseline,
        sil_cs
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
