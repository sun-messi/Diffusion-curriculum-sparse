#!/usr/bin/env python3
"""
Compare Silhouette Scores across multiple binary classifications.

Efficient GPU batch processing:
1. Load CelebA once, keep all attributes
2. For each (model, step): extract features ONCE, compute multiple silhouette scores
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

# Binary classification attributes
BINARY_ATTRS = {
    'gender': 20,
    'age': 39,
    'smiling': 31,
    'eyeglasses': 15,
    'attractive': 2,
    'chubby': 13,
    'heavy_makeup': 18,
    'wearing_hat': 35,
    'bald': 4,
    'bangs': 5,
    'big_nose': 7,
    'high_cheekbones': 19,
}

MODELS = {
    'Baseline': '../workdir/celeba64_uvit_small/default_20260101_030900',
    'CS_Mode': '../workdir/celeba64_uvit_small_cs/default_20260101_073037',
    'C_Mode': '../workdir/celeba64_uvit_small_c/default',
}

STEPS = [20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000]


def get_uvit_config():
    return {
        'img_size': 64, 'patch_size': 4, 'embed_dim': 256,
        'depth': 12, 'num_heads': 8, 'mlp_ratio': 4,
        'qkv_bias': False, 'mlp_time_embed': False, 'num_classes': -1,
    }


def load_model(ckpt_dir, step, device):
    model = UViT(**get_uvit_config())
    ckpt_path = os.path.join(ckpt_dir, 'ckpts', f'{step}.ckpt', 'nnet_ema.pth')
    if not os.path.exists(ckpt_path):
        return None
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class Crop:
    def __init__(self, x1, x2, y1, y2):
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
    def __call__(self, img):
        return transforms.functional.crop(img, self.x1, self.y1, self.x2-self.x1, self.y2-self.y1)


def load_celeba_with_attrs(root, num_samples=2000, seed=42):
    """Load CelebA images with ALL attributes."""
    root = os.path.expanduser(root)
    cx, cy = 89, 121
    transform = transforms.Compose([
        Crop(cy-64, cy+64, cx-64, cx+64),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    dataset = tvds.CelebA(root=root, split='test', target_type='attr', transform=transform, download=False)

    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))[:num_samples]

    images = []
    all_attrs = []
    for idx in tqdm(indices, desc="Loading CelebA"):
        img, attrs = dataset[idx]
        images.append(img)
        all_attrs.append(attrs.numpy())

    return torch.stack(images), np.array(all_attrs)


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
                feat = feat[:, 1:, :].flatten(1)
            all_features.append(feat.numpy())
            ext.clear()
    return np.concatenate(all_features)


def compute_silhouette_for_attr(features, attrs, attr_idx):
    """Compute silhouette score for a binary attribute."""
    labels = attrs[:, attr_idx]
    return silhouette_score(features, labels, metric='cosine')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--celeba_path', type=str, default='../assets/datasets/celeba')
    parser.add_argument('--output', type=str, default='outputs/silhouette_binary_all.csv')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Load CelebA ONCE
    print("\n=== Loading CelebA ===")
    images, attrs = load_celeba_with_attrs(args.celeba_path, args.num_samples)
    print(f"Loaded {len(images)} images with {attrs.shape[1]} attributes")

    # 2. Process each (model, step)
    results = []

    for step in STEPS:
        print(f"\n=== Step {step} ===")
        step_features = {}

        # Extract features for each model
        for model_name, ckpt_dir in MODELS.items():
            print(f"  {model_name}...", end=' ', flush=True)
            model = load_model(ckpt_dir, step, device)
            if model is None:
                print("NOT FOUND")
                step_features[model_name] = None
                continue

            features = extract_features(model, images, device, batch_size=args.batch_size)
            step_features[model_name] = features
            print(f"features: {features.shape}")

            del model
            torch.cuda.empty_cache()

        # Compute silhouette scores for all attributes
        for attr_name, attr_idx in BINARY_ATTRS.items():
            row = {'Classification': attr_name, 'Step': step}
            for model_name in MODELS.keys():
                if step_features[model_name] is None:
                    row[model_name] = float('nan')
                else:
                    score = compute_silhouette_for_attr(step_features[model_name], attrs, attr_idx)
                    row[model_name] = score
            results.append(row)

    # 3. Save CSV
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")

    # Print summary
    print("\n=== Summary (Step 200000) ===")
    df_200k = df[df['Step'] == 200000]
    print(df_200k.to_string(index=False))


if __name__ == '__main__':
    main()
