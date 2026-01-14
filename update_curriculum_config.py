#!/usr/bin/env python3
"""
Update curriculum_infographic_config.json using dotted paths.

Example:
  python update_curriculum_config.py section1.grid_axes_y 0.58
  python update_curriculum_config.py title.font_size 24
"""

import argparse
import json
from pathlib import Path

ROOT = Path('/home/sunj11/Documents/U-ViT-fresh')
DEFAULT_CONFIG = ROOT / 'curriculum_infographic_config.json'


def set_path(data, path, value):
    parts = path.split('.')
    cur = data
    for key in parts[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[parts[-1]] = value


def parse_value(raw):
    if raw and raw[0] in '[{':
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    if raw.lower() in {'true', 'false'}:
        return raw.lower() == 'true'
    try:
        if '.' in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def main():
    parser = argparse.ArgumentParser(description='Update infographic config values.')
    parser.add_argument('path', help='Dotted config path, e.g. section1.grid_axes_y')
    parser.add_argument('value', help='New value (number/bool/string)')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config_path = Path(args.config)
    data = json.loads(config_path.read_text(encoding='utf-8'))
    set_path(data, args.path, parse_value(args.value))
    config_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(f'Updated {args.path} -> {args.value} in {config_path}')


if __name__ == '__main__':
    main()
