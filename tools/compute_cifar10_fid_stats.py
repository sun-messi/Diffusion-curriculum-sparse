"""Compute FID statistics for CIFAR-10 training set"""
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.nn.functional import adaptive_avg_pool2d

from inception import InceptionV3


def get_activations_from_dataset(dataset, model, batch_size=50, dims=2048, device='cpu'):
    """Calculate activations for a dataset"""
    model.eval()

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=8, drop_last=False
    )

    pred_arr = np.empty((len(dataset), dims))
    start_idx = 0

    for batch, _ in tqdm(dataloader, desc="Computing activations"):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]

    return pred_arr


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load CIFAR-10 training set
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.CIFAR10(
        root='../assets/datasets/cifar10',
        train=True,
        download=True,
        transform=transform
    )
    print(f"CIFAR-10 training set size: {len(dataset)}")

    # Load Inception model
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    # Compute activations
    act = get_activations_from_dataset(dataset, model, batch_size=100, dims=dims, device=device)

    # Compute statistics
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)

    print(f"mu shape: {mu.shape}")
    print(f"sigma shape: {sigma.shape}")

    # Save
    out_path = '../assets/fid_stats/fid_stats_cifar10_train_pytorch.npz'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, mu=mu, sigma=sigma)
    print(f"Saved to: {out_path}")


if __name__ == '__main__':
    main()
