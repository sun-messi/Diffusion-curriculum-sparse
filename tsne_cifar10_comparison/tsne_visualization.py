#!/usr/bin/env python3
"""
t-SNE visualization of CIFAR-10 feature space for different models.
Style aligned with plot_macro_comparison.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent.parent))

from torchvision import datasets as tvds, transforms
from libs.uvit import UViT
from tsne_utils.feature_extractor import FeatureExtractor

# CIFAR-10 class names (select 5 classes)
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
SELECTED_CLASSES = [0, 2, 3, 6, 8]  # airplane, bird, cat, frog, ship

# Model checkpoint directories
MODELS = {
    'Baseline': '../workdir/cifar10_uvit_small/default_20260110_065028',
    'CS_Mode': '../workdir/cifar10_uvit_small_cs/default_20260110_094434',
    'C_Mode': '../workdir/cifar10_uvit_small_c/default_20260110_125959',
    'S_Mode': '../workdir/cifar10_uvit_small_s/default_20260110_193200',
}

# Colors for 5 selected classes (high contrast)
COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#42d4f4', '#f58231']


def get_uvit_config():
    return {
        'img_size': 32, 'patch_size': 2, 'embed_dim': 256,
        'depth': 12, 'num_heads': 8, 'mlp_ratio': 4,
        'qkv_bias': False, 'mlp_time_embed': False, 'num_classes': -1,
    }


def load_model(ckpt_dir, step, device):
    model = UViT(**get_uvit_config())
    ckpt_path = os.path.join(ckpt_dir, 'ckpts', '%d.ckpt' % step, 'nnet_ema.pth')
    if not os.path.exists(ckpt_path):
        return None
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_cifar10_balanced(root, samples_per_class=500, seed=42, selected_classes=None):
    """Load balanced samples from selected classes."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = tvds.CIFAR10(root=root, train=True, transform=transform, download=False)

    if selected_classes is None:
        selected_classes = SELECTED_CLASSES

    np.random.seed(seed)

    class_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    selected_indices = []
    new_labels = []
    for new_label, c in enumerate(selected_classes):
        indices = np.random.choice(class_indices[c], samples_per_class, replace=False)
        selected_indices.extend(indices)
        new_labels.extend([new_label] * samples_per_class)

    images = torch.stack([dataset[i][0] for i in selected_indices])
    labels = np.array(new_labels)

    return images, labels


def extract_features(model, images, device, batch_size=64):
    all_features = []
    with FeatureExtractor(model) as ext:
        ext.register_layer("target", "mid_block")
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            t = torch.zeros(len(batch), dtype=torch.long, device=device)
            feat = ext.extract_features(batch, t)["target"]
            if feat.dim() == 3:
                feat = feat[:, 1:, :].flatten(1)
            all_features.append(feat.numpy())
            ext.clear()
    return np.concatenate(all_features)


def plot_tsne(ax, features, labels, title):
    """Plot t-SNE with colored classes."""
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    features_2d = tsne.fit_transform(features)

    for i, c in enumerate(SELECTED_CLASSES):
        mask = labels == i
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=COLORS[i], s=50, alpha=0.7, label=CIFAR10_CLASSES[c],
                   edgecolors='none')

    ax.set_title(title, fontsize=32, fontweight='bold', pad=15)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    # Set global style (aligned with plot_macro_comparison.py)
    plt.rcParams['font.size'] = 28
    plt.rcParams['axes.labelsize'] = 28
    plt.rcParams['axes.titlesize'] = 32
    plt.rcParams['legend.fontsize'] = 22

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    step = 100000
    samples_per_class = 500

    print("Loading CIFAR-10...")
    images, labels = load_cifar10_balanced('../assets/datasets/cifar10', samples_per_class)
    class_names = [CIFAR10_CLASSES[c] for c in SELECTED_CLASSES]
    print("Loaded %d images (%d per class)" % (len(images), samples_per_class))
    print("Classes:", class_names)

    all_features = {}
    for model_name, ckpt_dir in MODELS.items():
        print("Extracting features: %s..." % model_name)
        model = load_model(ckpt_dir, step, device)
        if model is None:
            print("  Checkpoint not found!")
            continue
        features = extract_features(model, images, device)
        all_features[model_name] = features
        del model
        torch.cuda.empty_cache()

    # Plot 2x2 grid (larger figure)
    fig, axes = plt.subplots(2, 2, figsize=(24, 22))
    axes = axes.flatten()

    model_names = ['Baseline', 'CS_Mode', 'C_Mode', 'S_Mode']
    titles = ['(a) Baseline', '(b) CS Mode', '(c) C Mode', '(d) S Mode']

    for ax, model_name, title in zip(axes, model_names, titles):
        if model_name in all_features:
            plot_tsne(ax, all_features[model_name], labels, title)

    # Add legend
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='center right', fontsize=24,
               bbox_to_anchor=(0.98, 0.5), ncol=1, frameon=True,
               markerscale=3, handletextpad=0.5)

    plt.suptitle('t-SNE Feature Visualization (Step 100k)',
                 fontsize=36, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.savefig('outputs/tsne_cifar10_5class_500.png', dpi=300, bbox_inches='tight')
    print("\nSaved: outputs/tsne_cifar10_5class_500.png")


if __name__ == '__main__':
    main()
