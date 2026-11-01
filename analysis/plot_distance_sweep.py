#!/usr/bin/env python3
"""
Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
Conference: SIGGRAPH Asia 2026
Project Page: https://opueyociutad.github.io/InstantRPH_Web/

Plot a propagation-distance sweep of 2D LAE runs (analysis/train_distance_sweep.sh): baseline
PSNR, LAE PSNR and PSNR gain vs. distance, from each run's sweep_metrics.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sweep-dir', required=True, help='Folder with one train_2d.py run per distance.')
    parser.add_argument('--output', default=None, help='Default: <sweep-dir>/sweep_plot.png.')
    parser.add_argument('--title', default=None)
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    results = sorted((json.loads(p.read_text()) for p in sweep_dir.glob('*/sweep_metrics.json')), key=lambda r: r['target_distance_mm'])
    if not results:
        raise SystemExit(f'[Error] No sweep_metrics.json files found under {sweep_dir}')
    missing = [r['target_distance_mm'] for r in results if r.get('best_val_psnr') is None]
    if missing:
        print(f'[Warning] Missing best_val_psnr for distances: {missing}. Those runs may not have completed.')

    distances = [r['target_distance_mm'] for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, key, color, title, ylabel in (
        (axes[0], 'best_val_dummy_psnr', 'tab:blue', 'PSNR without LAE\n(amplitude-discard baseline)', 'PSNR (dB)'),
        (axes[1], 'best_val_psnr', 'tab:orange', 'PSNR with LAE', 'PSNR (dB)'),
        (axes[2], 'psnr_gain', 'tab:green', 'PSNR gain\n(LAE improvement)', 'ΔPSNR (dB)'),
    ):
        points = [(d, r.get(key)) for d, r in zip(distances, results) if r.get(key) is not None]
        if points:
            ax.plot(*zip(*points), 'o-', color=color, linewidth=1.8, markersize=6)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Propagation distance (mm)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[2].axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    fig.suptitle(args.title or f'Distance sweep — {len(results)} distances', fontsize=13, fontweight='bold')
    plt.tight_layout()
    output = Path(args.output) if args.output else sweep_dir / 'sweep_plot.png'
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Plot] Saved to {output}')

    print(f'\n{"Distance":>12}  {"Dummy PSNR":>12}  {"Model PSNR":>12}  {"Gain":>8}')
    print('-' * 52)
    for r in results:
        cells = [f'{r[k]:.2f} dB' if r.get(k) is not None else '    N/A' for k in ('best_val_dummy_psnr', 'best_val_psnr')]
        gain = f'{r["psnr_gain"]:+.2f} dB' if r.get('psnr_gain') is not None else '    N/A'
        print(f'{r["target_distance_mm"]:>12.2f}  {cells[0]:>12}  {cells[1]:>12}  {gain:>8}')


if __name__ == '__main__':
    main()
