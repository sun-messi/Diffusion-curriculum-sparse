#!/usr/bin/env python3
"""
Compute Silhouette Score for a SINGLE checkpoint.
Output: prints "step,score" to stdout (easy to capture)

Usage:
    python compute_single.py --ckpt_dir ../workdir/xxx --step 40000
    # Output: 40000,0.0123
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from torchvision import datasets as tvds, transforms
from libs.uvit import UViT
from tsne_utils.feature_extractor import FeatureExtractor

# Attribute indices
MALE_IDX = 20
YOUNG_IDX = 39
BLACK_HAIR_IDX = 8
BLOND_HAIR_IDX = 9
BROWN_HAIR_IDX = 11
GRAY_HAIR_IDX = 17

CLASSIFICATION_SCHEMES = {
    'gender_age': {
        'num_classes': 4,
        'names': ['Old Female', 'Young Female', 'Old Male', 'Young Male'],
    },
    'hair_color': {
        'num_classes': 4,
        'names': ['Black Hair', 'Blond Hair', 'Brown Hair', 'Gray Hair'],
        'indices': [BLACK_HAIR_IDX, BLOND_HAIR_IDX, BROWN_HAIR_IDX, GRAY_HAIR_IDX],
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--layer', type=str, default='mid_block')
    parser.add_argument('--samples_per_class', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--celeba_path', type=str, default='../assets/datasets/celeba')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--classification', type=str, default='gender_age',
                        choices=['gender_age', 'hair_color'],
                        help='Classification scheme (default: gender_age)')
    return parser.parse_args()


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


def load_celeba(root):
    root = os.path.expanduser(root)
    cx, cy = 89, 121
    transform = transforms.Compose([
        Crop(cy-64, cy+64, cx-64, cx+64),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    return tvds.CelebA(root=root, split='test', target_type='attr', transform=transform, download=False)


def get_label_gender_age(attrs):
    """Gender x Age: Male*2 + Young"""
    return int(attrs[MALE_IDX]) * 2 + int(attrs[YOUNG_IDX])


def get_label_hair_color(attrs):
    """Hair color: Black=0, Blond=1, Brown=2, Gray=3, None=-1"""
    indices = CLASSIFICATION_SCHEMES['hair_color']['indices']
    for i, idx in enumerate(indices):
        if attrs[idx] == 1:
            return i
    return -1  # No hair color attribute set


def sample_balanced(dataset, samples_per_class, seed, classification='gender_age'):
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_classes = CLASSIFICATION_SCHEMES[classification]['num_classes']
    class_samples = {i: [] for i in range(num_classes)}
    class_counts = {i: 0 for i in range(num_classes)}

    get_label = get_label_gender_age if classification == 'gender_age' else get_label_hair_color

    for idx in np.random.permutation(len(dataset)):
        img, attrs = dataset[idx]
        label = get_label(attrs)
        if label < 0 or label >= num_classes:
            continue  # Skip invalid labels
        if class_counts[label] < samples_per_class:
            class_samples[label].append(img)
            class_counts[label] += 1
        if all(c >= samples_per_class for c in class_counts.values()):
            break

    images, labels = [], []
    for label in range(num_classes):
        images.extend(class_samples[label][:samples_per_class])
        labels.extend([label] * len(class_samples[label][:samples_per_class]))
    return torch.stack(images), np.array(labels)


def extract_features(model, images, device, layer, batch_size):
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


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model(args.ckpt_dir, args.step, device)
    if model is None:
        print(f"{args.classification},{args.step},NaN", file=sys.stdout)
        return

    # Load data
    dataset = load_celeba(args.celeba_path)
    images, labels = sample_balanced(dataset, args.samples_per_class, args.seed, args.classification)

    # Extract features & compute score
    features = extract_features(model, images, device, args.layer, args.batch_size)
    score = silhouette_score(features, labels, metric='cosine')

    # Output: classification,step,score
    print(f"{args.classification},{args.step},{score:.6f}")


if __name__ == '__main__':
    main()
