#!/usr/bin/env python3
"""
Compare Silhouette Scores for CIFAR-10 10-class classification.

Compares Baseline, CS_Mode, and C_Mode models across training steps.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from torchvision import datasets as tvds, transforms
from libs.uvit import UViT
from tsne_utils.feature_extractor import FeatureExtractor

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Model checkpoint directories
MODELS = {
    'Baseline': '../workdir/cifar10_uvit_small/default_20260110_065028',
    'CS_Mode': '../workdir/cifar10_uvit_small_cs/default_20260110_094434',
    'C_Mode': '../workdir/cifar10_uvit_small_c/default_20260110_125959',
    'S_Mode': '../workdir/cifar10_uvit_small_s/default_20260110_193200',
}

STEPS = [20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000]


def get_uvit_config():
    """CIFAR-10 U-ViT config: img_size=32, patch_size=2"""
    return {
        'img_size': 32, 'patch_size': 2, 'embed_dim': 256,
        'depth': 12, 'num_heads': 8, 'mlp_ratio': 4,
        'qkv_bias': False, 'mlp_time_embed': False, 'num_classes': -1,
    }


def load_model(ckpt_dir, step, device):
    """Load U-ViT model from checkpoint."""
    model = UViT(**get_uvit_config())
    ckpt_path = os.path.join(ckpt_dir, 'ckpts', f'{step}.ckpt', 'nnet_ema.pth')
    if not os.path.exists(ckpt_path):
        return None
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_cifar10(root, num_samples=2000, seed=42):
    """Load CIFAR-10 images with labels (0-9)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = tvds.CIFAR10(root=root, train=False, transform=transform, download=False)

    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))[:num_samples]

    images = []
    labels = []
    for idx in tqdm(indices, desc="Loading CIFAR-10"):
        img, label = dataset[idx]
        images.append(img)
        labels.append(label)

    return torch.stack(images), np.array(labels)


def extract_features(model, images, device, layer='mid_block', batch_size=64):
    """Extract features using GPU batch processing."""
    all_features = []
    num_batches = (len(images) + batch_size - 1) // batch_size

    with FeatureExtractor(model) as ext:
        ext.register_layer("target", layer)
        for i in range(num_batches):
            batch = images[i*batch_size:(i+1)*batch_size].to(device)
            t = torch.zeros(len(batch), dtype=torch.long, device=device)
            feat = ext.extract_features(batch, t)["target"]
            if feat.dim() == 3:
                feat = feat[:, 1:, :].flatten(1)  # Remove time token, flatten
            all_features.append(feat.numpy())
            ext.clear()
    return np.concatenate(all_features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--cifar10_path', type=str, default='../assets/datasets/cifar10')
    parser.add_argument('--output', type=str, default='outputs/silhouette_cifar10.csv')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Load CIFAR-10 ONCE
    print("\n=== Loading CIFAR-10 ===")
    images, labels = load_cifar10(args.cifar10_path, args.num_samples)
    print(f"Loaded {len(images)} images with 10 classes")
    print(f"Label distribution: {np.bincount(labels)}")

    # 2. Process each step
    results = []

    for step in STEPS:
        print(f"\n=== Step {step} ===")
        row = {'Step': step}

        for model_name, ckpt_dir in MODELS.items():
            print(f"  {model_name}...", end=' ', flush=True)
            model = load_model(ckpt_dir, step, device)
            if model is None:
                print("NOT FOUND")
                row[model_name] = float('nan')
                continue

            features = extract_features(model, images, device, batch_size=args.batch_size)
            score = silhouette_score(features, labels, metric='cosine')
            row[model_name] = score
            print(f"score: {score:.6f}")

            del model
            torch.cuda.empty_cache()

        results.append(row)

    # 3. Save CSV
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")

    # Print summary
    print("\n=== Results ===")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
