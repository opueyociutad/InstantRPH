#!/usr/bin/env python3
"""
Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
Conference: SIGGRAPH Asia 2026
Project Page: https://opueyociutad.github.io/InstantRPH_Web/

Generate the random time-multiplexed phase tensor used by InstantRPH.

Writes i.i.d. uniform phases in [0, 2pi) with shape [LAYERS, TMX, RGB, H, W] to
<out-dir>/l<layers>_tmx<tmx>.pt. The 3D renderer uses one layer per MPI layer, the 2D model a
single layer. Smaller tmx or image sizes use a slice of it.

    python generate_phase_pattern.py                                            # 3D: data/models_3d/phase/l21_tmx24.pt
    python generate_phase_pattern.py --layers 1 --out-dir data/models_2d/phase  # 2D: data/models_2d/phase/l1_tmx24.pt

These reproduce the released tensors bit for bit (seed 0); the 2D tensor equals the first layer of the 3D one.
"""

import argparse
from pathlib import Path

import torch

from instantrph.common import DATA_DIR
from instantrph.constants import C, R, WAVELENGTHS_M


def generate_phase_pattern(
    layers: int,
    tmx: int,
    channels: int,
    rows: int,
    cols: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((layers, tmx, channels, rows, cols), generator=generator) * (2.0 * torch.pi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--layers', type=int, default=21, help='Number of MPI layers.')
    parser.add_argument('--tmx', type=int, default=24, help='Number of time-multiplexed repeats.')
    parser.add_argument('--channels', type=int, default=len(WAVELENGTHS_M), help='Number of color channels.')
    parser.add_argument('--rows', type=int, default=R, help='SLM rows.')
    parser.add_argument('--cols', type=int, default=C, help='SLM cols.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (0 reproduces the released phase tensor).')
    parser.add_argument('--out-dir', type=str, default=str(DATA_DIR / 'models_3d' / 'phase'), help='Output directory.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_pattern = generate_phase_pattern(
        layers=args.layers,
        tmx=args.tmx,
        channels=args.channels,
        rows=args.rows,
        cols=args.cols,
        seed=args.seed,
    )
    out_path = out_dir / f'l{args.layers}_tmx{args.tmx}.pt'
    torch.save(phase_pattern, out_path)
    print(f'[Phase] Saved {tuple(phase_pattern.shape)} phase tensor to {out_path}')


if __name__ == '__main__':
    main()
