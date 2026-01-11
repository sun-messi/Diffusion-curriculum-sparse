#!/usr/bin/env python3
"""
Compute Silhouette Score for a Single U-ViT Model

Supports different layers and distance metrics.
"""

import argparse
import os
import sys
from pathlib import Path

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
from tsne_utils.feature_extractor import FeatureExtractor, get_uvit_target_layers


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
        description='Compute silhouette score for a single U-ViT model'
    )

    parser.add_argument('--ckpt_dir', type=str, required=True,
                        help='Path to checkpoint directory')
    parser.add_argument('--step', type=int, default=200000,
                        help='Training step to load (default: 200000)')
    parser.add_argument('--layer', type=str, default='mid_block',
                        help='Layer to extract features from (default: mid_block)')
    parser.add_argument('--metric', type=str, default='cosine',
                        choices=['cosine', 'euclidean', 'manhattan', 'chebyshev'],
                        help='Distance metric (default: cosine)')
    parser.add_argument('--samples_per_class', type=int, default=100,
                        help='Number of samples per class (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (default: cuda:0)')
    parser.add_argument('--celeba_path', type=str, default='../assets/datasets/celeba',
                        help='Path to CelebA dataset')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for feature extraction')
    parser.add_argument('--all_layers', action='store_true',
                        help='Compute for all predefined layers')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (optional)')

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
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

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

        for i in tqdm(range(num_batches), desc=f"Extracting ({layer})", leave=False):
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

    # Load model
    print(f"\n=== Loading Model (step={args.step}) ===")
    model = load_uvit_model(args.ckpt_dir, args.step, device)

    # Determine layers to process
    if args.all_layers:
        layers = get_uvit_target_layers()
    else:
        layers = [("target", args.layer)]

    results = []

    print(f"\n=== Computing Silhouette Scores (metric={args.metric}) ===")
    for layer_name, layer_path in layers:
        print(f"\nProcessing layer: {layer_path}")

        features = extract_features(model, images, device, layer_path, args.batch_size)
        print(f"  Feature shape: {features.shape}")

        score = silhouette_score(features, labels_np, metric=args.metric)
        print(f"  Silhouette Score: {score:.4f}")

        results.append({
            'Layer': layer_path,
            'Metric': args.metric,
            'Silhouette': score,
            'Feature_Dim': features.shape[1]
        })

    # Save results
    if args.output:
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nSaved results to: {args.output}")

    # Print summary
    print("\n=== Summary ===")
    for r in results:
        print(f"  {r['Layer']}: {r['Silhouette']:.4f} (dim={r['Feature_Dim']})")

    print("\nDone!")


if __name__ == '__main__':
    main()
