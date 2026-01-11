#!/usr/bin/env python3
"""Visualize CIFAR-10 images at different noise levels."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Load CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = datasets.CIFAR10(root='assets/datasets/cifar10', train=True, transform=transform, download=False)

# Select some images
num_images = 4
indices = [0, 100, 500, 1000]  # Different images
images = torch.stack([dataset[i][0] for i in indices])

# Noise levels to visualize
t_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

# VP-SDE noise schedule (linear beta)
def get_alpha_bar(t, beta_min=0.1, beta_max=20.0):
    """Get cumulative alpha for VP-SDE."""
    # Linear beta schedule
    beta_t = beta_min + t * (beta_max - beta_min)
    # Integral of beta from 0 to t
    log_alpha = -0.5 * t * (beta_min + beta_t)
    return torch.exp(log_alpha)

def add_noise(x, t):
    """Add noise to image at timestep t."""
    if t == 0:
        return x
    alpha_bar = get_alpha_bar(torch.tensor(t))
    noise = torch.randn_like(x)
    x_t = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * noise
    return x_t

# Create visualization
fig, axes = plt.subplots(num_images, len(t_values), figsize=(len(t_values) * 2, num_images * 2))

for i, img in enumerate(images):
    for j, t in enumerate(t_values):
        noisy_img = add_noise(img, t)
        # Denormalize for display
        display_img = noisy_img * 0.5 + 0.5
        display_img = display_img.clamp(0, 1)
        display_img = display_img.permute(1, 2, 0).numpy()

        axes[i, j].imshow(display_img)
        axes[i, j].axis('off')
        if i == 0:
            alpha_bar = get_alpha_bar(torch.tensor(t)).item()
            axes[i, j].set_title(f't={t}\nα̅={alpha_bar:.3f}', fontsize=10)

plt.suptitle('CIFAR-10 Images at Different Noise Levels (VP-SDE)', fontsize=14)
plt.tight_layout()
plt.savefig('cifar10_noise_levels.png', dpi=150, bbox_inches='tight')
print('Saved: cifar10_noise_levels.png')
plt.show()
