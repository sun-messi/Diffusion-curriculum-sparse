"""
Generate samples and compare FID between C and CS models.
Uses multiple GPUs for parallel generation.

Models:
    - C:  Curriculum ON, Sparsity OFF (ddpm_celeba_c_32.pth)
    - CS: Curriculum ON, Sparsity ON  (ddpm_celeba_cs_32.pth)

Usage:
    python eval_fid.py --num_samples 2000 --batch_size 128 --gpus 0,1,4

"""

import os
import argparse
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image

from mindiffusion.unet import NaiveUnet
from mindiffusion.ddpm import DDPM
from mindiffusion.sparse_unet import SparseNaiveUnet
from mindiffusion.curriculum_ddpm import CurriculumDDPM

from dotenv import load_dotenv
load_dotenv("./.env")
CELEBA_PATH = os.getenv("CELEBA_PATH")


def generate_on_gpu(rank, gpu_ids, num_samples, output_dir, model_type, batch_size):
    """Generate samples on a single GPU."""
    gpu_id = gpu_ids[rank]
    device = torch.device(f'cuda:{gpu_id}')
    num_gpus = len(gpu_ids)

    # Each GPU generates num_samples // num_gpus samples
    samples_per_gpu = num_samples // num_gpus
    start_idx = rank * samples_per_gpu

    # Last GPU handles remainder
    if rank == num_gpus - 1:
        samples_per_gpu += num_samples % num_gpus

    # Load model
    if model_type == 'c':
        # C: Curriculum ON, Sparsity OFF
        model = CurriculumDDPM(
            eps_model=NaiveUnet(3, 3, n_feat=32),
            betas=(1e-4, 0.02),
            n_T=1000
        )
        model.load_state_dict(torch.load('./ddpm_celeba_c_32.pth', map_location=device))
        model.set_time_range(0.0, 1.0)
    else:  # cs
        # CS: Curriculum ON, Sparsity ON
        model = CurriculumDDPM(
            eps_model=SparseNaiveUnet(3, 3, n_feat=32, initial_sparsity=0.0),
            betas=(1e-4, 0.02),
            n_T=1000
        )
        model.load_state_dict(torch.load('./ddpm_celeba_cs_32.pth', map_location=device))
        model.set_time_range(0.0, 1.0)

    model.to(device)
    model.eval()

    # Generate
    count = 0
    pbar = tqdm(total=samples_per_gpu, desc=f"GPU{gpu_id} {model_type}", position=rank)

    while count < samples_per_gpu:
        bs = min(batch_size, samples_per_gpu - count)
        with torch.no_grad():
            xh = model.sample(bs, (3, 32, 32), device)
        for i in range(bs):
            img = (xh[i] + 1) / 2
            save_image(img.clamp(0, 1), f'{output_dir}/{start_idx + count:05d}.png')
            count += 1
        pbar.update(bs)
    pbar.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gpus', type=str, default='0,1,4,5', help='Comma-separated GPU ids, e.g., 0,1,4,5')
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpus.split(',')]
    num_gpus = len(gpu_ids)
    print(f"Using GPUs: {gpu_ids}")

    output_dir = './generated_samples'

    # ==================== 1. Save real samples ====================
    print("="*50)
    print("Saving real CelebA samples...")
    real_dir = os.path.join(output_dir, 'real')
    os.makedirs(real_dir, exist_ok=True)

    tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(root=CELEBA_PATH, transform=tf)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    count = 0
    for x, _ in tqdm(dataloader, desc="Real"):
        for i in range(x.shape[0]):
            if count >= args.num_samples:
                break
            save_image(x[i], f'{real_dir}/{count:05d}.png')
            count += 1
        if count >= args.num_samples:
            break
    print(f"Saved {count} real samples")

    # ==================== 2. C model (Curriculum ON, Sparsity OFF) ====================
    print("\n" + "="*50)
    print(f"Generating C (Curriculum ON, Sparsity OFF) samples on {num_gpus} GPUs: {gpu_ids}")
    c_dir = os.path.join(output_dir, 'c')
    os.makedirs(c_dir, exist_ok=True)

    mp.spawn(
        generate_on_gpu,
        args=(gpu_ids, args.num_samples, c_dir, 'c', args.batch_size),
        nprocs=num_gpus,
        join=True
    )
    print(f"Saved {args.num_samples} C model samples")

    # ==================== 3. CS model (Curriculum ON, Sparsity ON) ====================
    print("\n" + "="*50)
    print(f"Generating CS (Curriculum ON, Sparsity ON) samples on {num_gpus} GPUs: {gpu_ids}")
    cs_dir = os.path.join(output_dir, 'cs')
    os.makedirs(cs_dir, exist_ok=True)

    mp.spawn(
        generate_on_gpu,
        args=(gpu_ids, args.num_samples, cs_dir, 'cs', args.batch_size),
        nprocs=num_gpus,
        join=True
    )
    print(f"Saved {args.num_samples} CS model samples")

    # ==================== 4. Calculate FID ====================
    print("\n" + "="*50)
    print("Calculating FID...")
    print("="*50)

    import subprocess
    from datetime import datetime

    fid_device = f'cuda:{gpu_ids[0]}'
    fid_results = []

    # FID 1: C vs Real
    print(f"\nC vs Real (on {fid_device}):")
    result = subprocess.run(
        ['python', '-m', 'pytorch_fid', real_dir, c_dir, '--device', fid_device],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    # Extract FID value from output
    c_fid = None
    for line in result.stdout.split('\n'):
        if 'FID' in line:
            try:
                c_fid = float(line.split(':')[-1].strip())
            except:
                c_fid = line.strip()
    fid_results.append({
        'comparison': 'C vs Real',
        'path1': c_dir,
        'path2': real_dir,
        'model1': 'CurriculumDDPM + NaiveUnet (Curriculum ON, Sparsity OFF)',
        'model1_weight': 'ddpm_celeba_c_32.pth',
        'model2': 'Real CelebA images',
        'fid': c_fid
    })

    # FID 2: CS vs Real
    print(f"\nCS vs Real (on {fid_device}):")
    result = subprocess.run(
        ['python', '-m', 'pytorch_fid', real_dir, cs_dir, '--device', fid_device],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    # Extract FID value from output
    cs_fid = None
    for line in result.stdout.split('\n'):
        if 'FID' in line:
            try:
                cs_fid = float(line.split(':')[-1].strip())
            except:
                cs_fid = line.strip()
    fid_results.append({
        'comparison': 'CS vs Real',
        'path1': cs_dir,
        'path2': real_dir,
        'model1': 'CurriculumDDPM + SparseUnet (Curriculum ON, Sparsity ON)',
        'model1_weight': 'ddpm_celeba_cs_32.pth',
        'model2': 'Real CelebA images',
        'fid': cs_fid
    })

    # ==================== 5. Save FID results to file ====================
    results_file = os.path.join(output_dir, 'fid_results.txt')
    with open(results_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("FID Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of samples: {args.num_samples}\n")
        f.write(f"Image resolution: 32x32\n")
        f.write("\n")

        for i, res in enumerate(fid_results, 1):
            f.write("-" * 60 + "\n")
            f.write(f"Comparison {i}: {res['comparison']}\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Model:   {res['model1']}\n")
            f.write(f"  Weight:  {res.get('model1_weight', 'N/A')}\n")
            f.write(f"  Path:    {res['path1']}\n")
            f.write(f"  vs:      {res['model2']}\n")
            f.write(f"  Path:    {res['path2']}\n")
            f.write(f"  FID Score: {res['fid']}\n")
            f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"  C  (Curriculum ON, Sparsity OFF) FID: {fid_results[0]['fid']}\n")
        f.write(f"  CS (Curriculum ON, Sparsity ON)  FID: {fid_results[1]['fid']}\n")
        if isinstance(fid_results[0]['fid'], (int, float)) and isinstance(fid_results[1]['fid'], (int, float)):
            diff = fid_results[0]['fid'] - fid_results[1]['fid']
            winner = 'CS (Sparsity helps)' if diff > 0 else 'C (Sparsity hurts)'
            f.write(f"  Difference: {abs(diff):.4f} ({winner})\n")
        f.write("=" * 60 + "\n")

    print(f"\n{'='*50}")
    print(f"FID results saved to: {results_file}")
    print(f"{'='*50}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
