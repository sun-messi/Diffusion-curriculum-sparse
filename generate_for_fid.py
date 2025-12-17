"""
Generate samples from Baseline and CS models for FID comparison.

Usage:
    python generate_for_fid.py --num_samples 5000

Then use pytorch-fid to calculate:
    pip install pytorch-fid
    python -m pytorch_fid ./generated_samples/real ./generated_samples/baseline
    python -m pytorch_fid ./generated_samples/real ./generated_samples/cs
"""

import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image

# Models
from mindiffusion.unet import NaiveUnet
from mindiffusion.ddpm import DDPM
from mindiffusion.sparse_unet import SparseNaiveUnet
from mindiffusion.curriculum_ddpm import CurriculumDDPM

from dotenv import load_dotenv

load_dotenv("./.env")
CELEBA_PATH = os.getenv("CELEBA_PATH")


def generate_samples(model, num_samples, batch_size, device, output_dir, model_name):
    """Generate and save samples from a model."""
    os.makedirs(output_dir, exist_ok=True)

    num_generated = 0
    batch_idx = 0

    pbar = tqdm(total=num_samples, desc=f"Generating {model_name}")

    while num_generated < num_samples:
        current_batch_size = min(batch_size, num_samples - num_generated)

        with torch.no_grad():
            samples = model.sample(current_batch_size, (3, 32, 32), device)

        # Save each sample as individual image
        for i in range(current_batch_size):
            img = samples[i]
            # Convert from [-1, 1] to [0, 1]
            img = (img + 1) / 2
            img = img.clamp(0, 1)
            save_image(img, os.path.join(output_dir, f'{num_generated + i:05d}.png'))

        num_generated += current_batch_size
        batch_idx += 1
        pbar.update(current_batch_size)

    pbar.close()
    print(f"Saved {num_generated} samples to {output_dir}")


def save_real_samples(num_samples, output_dir):
    """Save real CelebA samples."""
    os.makedirs(output_dir, exist_ok=True)

    tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(root=CELEBA_PATH, transform=tf)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    num_saved = 0
    pbar = tqdm(total=num_samples, desc="Saving real samples")

    for x, _ in dataloader:
        for i in range(x.shape[0]):
            if num_saved >= num_samples:
                break
            save_image(x[i], os.path.join(output_dir, f'{num_saved:05d}.png'))
            num_saved += 1
            pbar.update(1)

        if num_saved >= num_samples:
            break

    pbar.close()
    print(f"Saved {num_saved} real samples to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='./generated_samples')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Save real samples
    print("\n" + "="*50)
    print("Saving real CelebA samples...")
    save_real_samples(args.num_samples, os.path.join(args.output_dir, 'real'))

    # 2. Generate from Baseline model
    print("\n" + "="*50)
    print("Loading Baseline model...")
    baseline_ddpm = DDPM(
        eps_model=NaiveUnet(3, 3, n_feat=128),
        betas=(1e-4, 0.02),
        n_T=1000
    )
    baseline_ddpm.load_state_dict(torch.load('./ddpm_celeba_32.pth', map_location=device, weights_only=True))
    baseline_ddpm.to(device)
    baseline_ddpm.eval()

    generate_samples(baseline_ddpm, args.num_samples, args.batch_size, device,
                    os.path.join(args.output_dir, 'baseline'), 'Baseline')

    del baseline_ddpm
    torch.cuda.empty_cache()

    # 3. Generate from CS model
    print("\n" + "="*50)
    print("Loading CS model...")
    cs_ddpm = CurriculumDDPM(
        eps_model=SparseNaiveUnet(3, 3, n_feat=128, initial_sparsity=0.0),
        betas=(1e-4, 0.02),
        n_T=1000
    )
    cs_ddpm.load_state_dict(torch.load('./ddpm_celeba_cs_32.pth', map_location=device, weights_only=True))
    cs_ddpm.to(device)
    cs_ddpm.eval()
    cs_ddpm.set_time_range(0.0, 1.0)

    generate_samples(cs_ddpm, args.num_samples, args.batch_size, device,
                    os.path.join(args.output_dir, 'cs'), 'CS')

    print("\n" + "="*50)
    print("Done! Now run FID calculation:")
    print(f"  python -m pytorch_fid {args.output_dir}/real {args.output_dir}/baseline")
    print(f"  python -m pytorch_fid {args.output_dir}/real {args.output_dir}/cs")


if __name__ == '__main__':
    main()
