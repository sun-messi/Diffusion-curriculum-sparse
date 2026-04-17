"""
TOPIQ-NR-Face Quality Comparison for CelebA generated images.
Supports multi-GPU parallel scoring and multi-checkpoint curve plotting.
"""

import os
import argparse
import json
import random
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import pyiqa


def get_common_files(dir_a, dir_b, num_samples=None, seed=42):
    """Get common filenames between two directories for paired comparison."""
    files_a = set(
        f for f in os.listdir(dir_a)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    )
    files_b = set(
        f for f in os.listdir(dir_b)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    )
    common = sorted(files_a & files_b)

    if num_samples is not None and num_samples < len(common):
        random.seed(seed)
        common = sorted(random.sample(common, num_samples))

    return common


def score_worker(gpu_id, file_chunk, image_dir, result_dict, worker_id):
    """Worker function: score a chunk of images on one GPU."""
    device = f"cuda:{gpu_id}"
    model = pyiqa.create_metric('topiq_nr-face', device=device)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    scores = {}
    skipped = 0
    for fname in tqdm(file_chunk, desc=f"  GPU{gpu_id}", position=worker_id, leave=False):
        img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        try:
            with torch.no_grad():
                score = model(tensor)
            scores[fname] = score.cpu().item()
        except Exception:
            skipped += 1

    result_dict[worker_id] = {"scores": scores, "skipped": skipped}


def score_images_parallel(image_dir, file_list, gpu_ids):
    """Score images in parallel across multiple GPUs."""
    n_gpus = len(gpu_ids)
    chunks = [[] for _ in range(n_gpus)]
    for i, fname in enumerate(file_list):
        chunks[i % n_gpus].append(fname)

    manager = mp.Manager()
    result_dict = manager.dict()

    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        p = mp.Process(
            target=score_worker,
            args=(gpu_id, chunks[i], image_dir, result_dict, i),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Merge results
    all_scores = {}
    total_skipped = 0
    for i in range(n_gpus):
        all_scores.update(result_dict[i]["scores"])
        total_skipped += result_dict[i]["skipped"]

    if total_skipped > 0:
        print(f"    Skipped {total_skipped}/{len(file_list)} (no face detected)")

    valid_files = sorted(all_scores.keys())
    scores = np.array([all_scores[f] for f in valid_files])
    return scores, valid_files


def extract_model_name(dir_path):
    """Extract model name from path like .../celeba64_uvit_small_c/date/ckpt."""
    parts = dir_path.rstrip("/").split("/")
    for i, p in enumerate(parts):
        if p == "eval_samples" and i + 1 < len(parts):
            return parts[i + 1]
    return parts[-3] if len(parts) >= 3 else os.path.basename(dir_path)


def evaluate_one_checkpoint(dir_a, dir_b, gpu_ids, num_samples, seed):
    """Evaluate a single checkpoint pair, return stats dict."""
    ckpt_name = os.path.basename(dir_a)
    common_files = get_common_files(dir_a, dir_b, num_samples, seed)
    print(f"    Paired files: {len(common_files)}")

    print(f"    Scoring A...")
    scores_a_raw, valid_a = score_images_parallel(dir_a, common_files, gpu_ids)

    print(f"    Scoring B...")
    scores_b_raw, valid_b = score_images_parallel(dir_b, common_files, gpu_ids)

    # Keep only images both scored successfully
    valid_both = sorted(set(valid_a) & set(valid_b))
    map_a = dict(zip(valid_a, scores_a_raw))
    map_b = dict(zip(valid_b, scores_b_raw))
    scores_a = np.array([map_a[f] for f in valid_both])
    scores_b = np.array([map_b[f] for f in valid_both])

    diffs = scores_a - scores_b

    return {
        "checkpoint": ckpt_name,
        "num_paired": len(valid_both),
        "a_mean": float(np.mean(scores_a)),
        "a_std": float(np.std(scores_a)),
        "a_median": float(np.median(scores_a)),
        "b_mean": float(np.mean(scores_b)),
        "b_std": float(np.std(scores_b)),
        "b_median": float(np.median(scores_b)),
        "mean_diff": float(np.mean(diffs)),
        "a_wins": int(np.sum(diffs > 0)),
        "b_wins": int(np.sum(diffs < 0)),
        "ties": int(np.sum(diffs == 0)),
    }


def main():
    parser = argparse.ArgumentParser(description="TOPIQ-NR-Face multi-checkpoint comparison")
    parser.add_argument(
        "--base_a", type=str,
        default="/home/sunj11/Documents/U-ViT-fresh/eval_samples/"
                "celeba64_uvit_small/20260101_154543",
        help="Base dir for model A (contains checkpoint subdirs)",
    )
    parser.add_argument(
        "--base_b", type=str,
        default="/home/sunj11/Documents/U-ViT-fresh/eval_samples/"
                "celeba64_uvit_small_c/20260101_160525",
        help="Base dir for model B (contains checkpoint subdirs)",
    )
    parser.add_argument(
        "--checkpoints", type=str,
        default="100000_ema,120000_ema,140000_ema,160000_ema,180000_ema",
        help="Comma-separated checkpoint names",
    )
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--output_dir", type=str,
        default="/home/sunj11/Documents/U-ViT-fresh/topiq_face_evaluation/outputs",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    gpu_ids = [int(x) for x in args.gpus.split(",")]
    checkpoints = [c.strip() for c in args.checkpoints.split(",")]

    name_a = extract_model_name(args.base_a)
    name_b = extract_model_name(args.base_b)

    print(f"Model A: {name_a}")
    print(f"Model B: {name_b}")
    print(f"Checkpoints: {checkpoints}")
    print(f"GPUs: {gpu_ids}")
    print(f"Samples per checkpoint: {args.num_samples}")

    # Run all checkpoints
    all_results = []
    for ckpt in checkpoints:
        dir_a = os.path.join(args.base_a, ckpt)
        dir_b = os.path.join(args.base_b, ckpt)

        if not os.path.isdir(dir_a):
            print(f"\n  [SKIP] {ckpt}: {dir_a} not found")
            continue
        if not os.path.isdir(dir_b):
            print(f"\n  [SKIP] {ckpt}: {dir_b} not found")
            continue

        print(f"\n{'='*60}")
        print(f"  Checkpoint: {ckpt}")
        print(f"{'='*60}")

        result = evaluate_one_checkpoint(dir_a, dir_b, gpu_ids, args.num_samples, args.seed)
        all_results.append(result)

        # Print this checkpoint's result
        print(f"\n    {name_a}: mean={result['a_mean']:.4f} ± {result['a_std']:.4f}")
        print(f"    {name_b}: mean={result['b_mean']:.4f} ± {result['b_std']:.4f}")
        print(f"    A wins: {result['a_wins']} | B wins: {result['b_wins']} | "
              f"Pairs: {result['num_paired']}")

    # Save JSON
    output = {
        "model_a": name_a,
        "model_b": name_b,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "results": all_results,
    }
    json_path = os.path.join(args.output_dir, "topiq_face_multi_checkpoint.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  Summary: TOPIQ-NR-Face across checkpoints")
    print(f"{'='*70}")
    print(f"  {'Checkpoint':<15} {name_a:<12} {name_b:<12} {'Diff':>8} {'A wins':>8} {'B wins':>8}")
    print(f"  {'-'*63}")
    for r in all_results:
        step = r['checkpoint'].replace('_ema', '')
        print(f"  {step:<15} {r['a_mean']:<12.4f} {r['b_mean']:<12.4f} "
              f"{r['mean_diff']:>+8.4f} {r['a_wins']:>8} {r['b_wins']:>8}")
    print(f"{'='*70}")

    # Plot curve
    try:
        plot_curve(all_results, name_a, name_b, args.output_dir)
    except Exception as e:
        print(f"Plotting skipped: {e}")


def plot_curve(results, name_a, name_b, output_dir):
    """Plot TOPIQ-NR-Face score curve across checkpoints."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [int(r['checkpoint'].replace('_ema', '')) // 1000 for r in results]
    means_a = [r['a_mean'] for r in results]
    means_b = [r['b_mean'] for r in results]
    stds_a = [r['a_std'] for r in results]
    stds_b = [r['b_std'] for r in results]
    wins_b_pct = [r['b_wins'] / r['num_paired'] * 100 for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Score curve with std band
    ax = axes[0]
    ax.plot(steps, means_a, 'o-', color='#2196F3', linewidth=2, markersize=6, label=name_a)
    ax.fill_between(steps,
                    [m - s for m, s in zip(means_a, stds_a)],
                    [m + s for m, s in zip(means_a, stds_a)],
                    alpha=0.15, color='#2196F3')
    ax.plot(steps, means_b, 's-', color='#FF5722', linewidth=2, markersize=6, label=name_b)
    ax.fill_between(steps,
                    [m - s for m, s in zip(means_b, stds_b)],
                    [m + s for m, s in zip(means_b, stds_b)],
                    alpha=0.15, color='#FF5722')
    ax.set_xlabel("Training Steps (k)", fontsize=12)
    ax.set_ylabel("TOPIQ-NR-Face Score", fontsize=12)
    ax.set_title("Face Quality across Checkpoints", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Win rate curve
    ax = axes[1]
    ax.bar(steps, wins_b_pct, width=8, color='#FF5722', alpha=0.7, label=f'{name_b} win %')
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, label='50% baseline')
    ax.set_xlabel("Training Steps (k)", fontsize=12)
    ax.set_ylabel(f"{name_b} Win Rate (%)", fontsize=12)
    ax.set_title("Paired Win Rate across Checkpoints", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "topiq_face_curve.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
