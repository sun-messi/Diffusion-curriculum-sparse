#!/usr/bin/env python3
"""
Silhouette Score by Training Step for U-ViT

Track feature clustering quality across training checkpoints for
Baseline and CS (Curriculum+Sparsity) U-ViT models.
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

CLASS_NAMES = [
    'Old Female',
    'Young Female',
    'Old Male',
    'Young Male',
]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute silhouette scores across training steps'
    )

    parser.add_argument('--baseline_dir', type=str,
                        default='../workdir/celeba64_uvit_small/default_20260101_030900',
                        help='Path to baseline checkpoint directory')
    parser.add_argument('--cs_dir', type=str,
                        default='../workdir/celeba64_uvit_small_cs/default_20260101_073037',
                        help='Path to CS mode checkpoint directory')
    parser.add_argument('--c_dir', type=str,
                        default='../workdir/celeba64_uvit_small_c/default',
                        help='Path to C mode (Curriculum only) checkpoint directory')
    parser.add_argument('--start_step', type=int, default=10000,
                        help='Start step (default: 10000)')
    parser.add_argument('--end_step', type=int, default=200000,
                        help='End step (default: 200000)')
    parser.add_argument('--step_interval', type=int, default=10000,
                        help='Step interval (default: 10000)')
    parser.add_argument('--samples_per_class', type=int, default=100,
                        help='Number of samples per class (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (default: cuda:0)')
    parser.add_argument('--output', type=str, default='outputs/silhouette_by_step.png',
                        help='Output plot path')
    parser.add_argument('--csv_output', type=str, default='outputs/silhouette_by_step.csv',
                        help='Output CSV path')
    parser.add_argument('--celeba_path', type=str, default='../assets/datasets/celeba',
                        help='Path to CelebA dataset')
    parser.add_argument('--layer', type=str, default='mid_block',
                        help='Layer to extract features from (default: mid_block)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for feature extraction')

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
    """Load U-ViT model from checkpoint."""
    config = get_uvit_config()
    model = UViT(**config)

    ckpt_path = os.path.join(ckpt_dir, 'ckpts', f'{step}.ckpt', 'nnet_ema.pth')
    if not os.path.exists(ckpt_path):
        return None

    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
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
    """Load CelebA dataset with attribute labels."""
    root = os.path.expanduser(root)

    cx = 89
    cy = 121
    x1 = cy - 64
    x2 = cy + 64
    y1 = cx - 64
    y2 = cx + 64

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
    """Convert CelebA attributes to class label."""
    male = int(attrs[MALE_IDX].item())
    young = int(attrs[YOUNG_IDX].item())
    return male * 2 + young


def sample_balanced_data(dataset, samples_per_class: int, seed: int = 42) -> tuple:
    """Sample balanced data from CelebA dataset."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_classes = 4
    class_samples = {i: [] for i in range(num_classes)}
    class_counts = {i: 0 for i in range(num_classes)}

    indices = np.random.permutation(len(dataset))

    for idx in indices:
        img, attrs = dataset[idx]
        label = get_class_label(attrs)

        if class_counts[label] < samples_per_class:
            class_samples[label].append(img)
            class_counts[label] += 1

        if all(c >= samples_per_class for c in class_counts.values()):
            break

    all_images = []
    all_labels = []
    for label in range(num_classes):
        all_images.extend(class_samples[label][:samples_per_class])
        all_labels.extend([label] * len(class_samples[label][:samples_per_class]))

    images = torch.stack(all_images)
    labels = torch.tensor(all_labels)

    return images, labels


def extract_features(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    layer: str = 'mid_block',
    batch_size: int = 64
) -> np.ndarray:
    """Extract features from U-ViT model."""
    model.eval()
    all_features = []

    num_batches = (len(images) + batch_size - 1) // batch_size

    with FeatureExtractor(model) as extractor:
        extractor.register_layer("target", layer)

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(images))
            batch_images = images[start_idx:end_idx].to(device)

            t = torch.zeros(len(batch_images), dtype=torch.long, device=device)
            features = extractor.extract_features(batch_images, t)

            feat = features["target"]
            if feat.dim() == 3:
                feat = feat[:, 1:, :]
                feat = feat.flatten(start_dim=1)

            all_features.append(feat.numpy())
            extractor.clear()

    return np.concatenate(all_features, axis=0)


def compute_silhouette_for_step(
    ckpt_dir: str,
    step: int,
    images: torch.Tensor,
    labels: np.ndarray,
    device: torch.device,
    layer: str,
    batch_size: int
) -> float:
    """Compute silhouette score for a single checkpoint."""
    model = load_uvit_model(ckpt_dir, step, device)
    if model is None:
        return None

    features = extract_features(model, images, device, layer, batch_size)
    score = silhouette_score(features, labels, metric='cosine')

    # Clean up
    del model
    torch.cuda.empty_cache()

    return score


def plot_silhouette_curves(
    steps: list,
    baseline_scores: list,
    cs_scores: list,
    c_scores: list,
    output_path: str
):
    """Plot silhouette score curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, baseline_scores, 'b-o', label='Baseline', linewidth=2, markersize=6)
    ax.plot(steps, cs_scores, 'r-s', label='CS Mode', linewidth=2, markersize=6)
    ax.plot(steps, c_scores, 'g-^', label='C Mode', linewidth=2, markersize=6)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Silhouette Score (cosine)', fontsize=12)
    ax.set_title('Feature Clustering Quality During Training\n(CelebA Gender×Age 4-class)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.set_xticks(steps[::2])
    ax.set_xticklabels([f'{s//1000}k' for s in steps[::2]], rotation=45)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    plt.close()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load CelebA data
    print("\n=== Loading CelebA Test Data ===")
    dataset = get_celeba_with_attrs(args.celeba_path)
    images, labels = sample_balanced_data(dataset, args.samples_per_class, args.seed)
    labels_np = labels.numpy()
    print(f"  Sampled {len(images)} images")

    # Generate step list
    steps = list(range(args.start_step, args.end_step + 1, args.step_interval))
    print(f"\n=== Computing Silhouette Scores for {len(steps)} steps ===")

    baseline_scores = []
    cs_scores = []

    for step in tqdm(steps, desc="Processing steps"):
        # Baseline
        score_baseline = compute_silhouette_for_step(
            args.baseline_dir, step, images, labels_np, device, args.layer, args.batch_size
        )
        baseline_scores.append(score_baseline)

        # CS Mode
        score_cs = compute_silhouette_for_step(
            args.cs_dir, step, images, labels_np, device, args.layer, args.batch_size
        )
        cs_scores.append(score_cs)

        print(f"  Step {step}: Baseline={score_baseline:.4f}, CS={score_cs:.4f}")

    # Save CSV
    df = pd.DataFrame({
        'Step': steps,
        'Baseline': baseline_scores,
        'CS_Mode': cs_scores,
        'Difference': [cs - bl for bl, cs in zip(baseline_scores, cs_scores)]
    })

    os.makedirs(os.path.dirname(args.csv_output) if os.path.dirname(args.csv_output) else '.', exist_ok=True)
    df.to_csv(args.csv_output, index=False)
    print(f"\nSaved CSV to: {args.csv_output}")

    # Plot
    plot_silhouette_curves(steps, baseline_scores, cs_scores, args.output)

    # Summary
    print("\n=== Summary ===")
    print(f"Final Step ({steps[-1]}):")
    print(f"  Baseline: {baseline_scores[-1]:.4f}")
    print(f"  CS Mode:  {cs_scores[-1]:.4f}")
    print(f"  Improvement: {cs_scores[-1] - baseline_scores[-1]:+.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()

