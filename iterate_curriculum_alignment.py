#!/usr/bin/env python3
"""
Iterate curriculum infographic generation and save visual comparisons.

This script runs the generator, saves the output into an iteration folder,
and creates side-by-side and diff images for manual visual review.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from PIL import Image, ImageChops, ImageOps
import subprocess

ROOT = Path('/home/sunj11/Documents/U-ViT-fresh')
TARGET_PATH = ROOT / 'joint_curriculum_final.png'
OUTPUT_PATH = ROOT / 'curriculum_infographic.png'
ITER_DIR = ROOT / 'analysis_outputs' / 'iterations'


def next_iteration_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted([p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith('iter_')])
    if not existing:
        return base_dir / 'iter_0001'
    last = existing[-1].name.split('_')[-1]
    try:
        idx = int(last)
    except ValueError:
        idx = len(existing)
    return base_dir / f'iter_{idx + 1:04d}'


def create_comparisons(gen_path: Path, target_path: Path, out_dir: Path) -> None:
    gen = Image.open(gen_path).convert('RGB')
    target = Image.open(target_path).convert('RGB')

    # Normalize sizes for direct compare
    if gen.size != target.size:
        target = target.resize(gen.size, Image.LANCZOS)

    side_by_side = Image.new('RGB', (gen.width * 2, gen.height))
    side_by_side.paste(gen, (0, 0))
    side_by_side.paste(target, (gen.width, 0))
    side_by_side.save(out_dir / 'side_by_side.png')

    diff = ImageChops.difference(gen, target)
    diff_enhanced = ImageOps.autocontrast(diff, cutoff=1)
    diff_enhanced.save(out_dir / 'diff_autocontrast.png')

    # Overlay for quick scanning
    overlay = Image.blend(gen, target, alpha=0.5)
    overlay.save(out_dir / 'overlay_50.png')


def run_iteration(config_path: Path, notes: str) -> Path:
    out_dir = next_iteration_dir(ITER_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run generator
    cmd = ['python', str(ROOT / 'create_curriculum_infographic.py'), '--config', str(config_path)]
    subprocess.run(cmd, check=True)

    # Copy outputs
    gen_copy = out_dir / 'curriculum_infographic.png'
    gen_copy.write_bytes(OUTPUT_PATH.read_bytes())

    create_comparisons(gen_copy, TARGET_PATH, out_dir)

    meta = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'config_path': str(config_path),
        'generator_output': str(gen_copy),
        'target_path': str(TARGET_PATH),
        'notes': notes
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    if notes:
        (out_dir / 'notes.txt').write_text(notes + '\n', encoding='utf-8')

    return out_dir


def main():
    parser = argparse.ArgumentParser(description='Iterate curriculum infographic alignment.')
    parser.add_argument('--config', default=str(ROOT / 'curriculum_infographic_config.json'))
    parser.add_argument('--notes', default='', help='Optional notes for this iteration.')
    args = parser.parse_args()

    out_dir = run_iteration(Path(args.config), args.notes)
    print(f'Iteration saved to {out_dir}')


if __name__ == '__main__':
    main()
