#!/usr/bin/env python3
"""
Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
Conference: SIGGRAPH Asia 2026
Project Page: https://opueyociutad.github.io/InstantRPH_Web/

Merge three single-wavelength train_3d.py runs (R, G, B) into one RGB 3D LAE checkpoint.

Inputs are run directories (best.pt or last.pt; older accelerate runs: best/ or last/) or
checkpoint files. The output directory gets best.pt and a config.json usable by eval_3d.py,
benchmark_3d.py and render_video_3d.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from instantrph.common import find_checkpoint, load_json, load_state_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for color in 'rgb':
        parser.add_argument(f'--{color}', required=True, help=f'{color.upper()} run directory or checkpoint.')
    parser.add_argument('--output-run-dir', default='', help='Default: <R run>_merged_rgb.')
    parser.add_argument('--checkpoint-name', default='best.pt')
    args = parser.parse_args()

    inputs = [Path(args.r), Path(args.g), Path(args.b)]
    checkpoints = [find_checkpoint(path) for path in inputs]
    run_dirs = [path if path.is_dir() else path.parent for path in inputs]
    states = [load_state_dict(path) for path in checkpoints]

    if any(set(state) != {'scale', 'bias'} for state in states):
        raise ValueError('Expected 3D LAE checkpoints (scale and bias only)')
    merged = {}
    for key in ('scale', 'bias'):
        tensors = [state[key] for state in states]
        if any(t.shape != tensors[0].shape for t in tensors) or tensors[0].shape[2] != 1:
            raise ValueError(f'Expected three single-wavelength {key} maps of equal shape, got {[tuple(t.shape) for t in tensors]}')
        merged[key] = torch.cat(tensors, dim=2)  # [1, tmx, 1, D, H, W] x 3 -> [1, tmx, 3, D, H, W]

    out_dir = Path(args.output_run_dir) if args.output_run_dir else Path(f'{run_dirs[0]}_merged_rgb')
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dirs[0] / 'config.json'
    config = load_json(config_path) if config_path.exists() else {}
    config.update(
        train_wavelength='rgb',
        train_wavelength_channels=['R', 'G', 'B'],
        output_dir=str(out_dir),
        merged_from_runs={color: str(path) for color, path in zip('rgb', checkpoints)},
    )
    torch.save({'model_state_dict': merged, 'config': config}, out_dir / args.checkpoint_name)
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'Merged {", ".join(map(str, checkpoints))} -> {out_dir / args.checkpoint_name}')


if __name__ == '__main__':
    main()
