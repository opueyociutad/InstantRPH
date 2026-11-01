#!/usr/bin/env python3
"""
Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
Conference: SIGGRAPH Asia 2026
Project Page: https://opueyociutad.github.io/InstantRPH_Web/

Per-image PSNR of a trained 2D LAE and of the amplitude-discard baseline on any folder of images.

    python eval_2d_images.py --run-dir data/models_2d/lae/<run> --images-dir path/to/images [--bypass-lae]

--bypass-lae replaces the learned maps with the identity (ablation).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from instantrph.common import PHASE_FILE_2D, pick_device, run_phase_file
from instantrph.data_2d import ImageFolder
from instantrph.hologram_2d import Hologram2D
from instantrph.metrics import psnr
from instantrph.models import IdentityEncoder, load_lae_2d
from instantrph.viz import save_png


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', required=True, help='A train_2d.py output directory.')
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--checkpoint', default='', help='Default: <run-dir>/best.pt.')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--bypass-lae', action='store_true', help='Use the replicated input instead of the learned LAE maps.')
    parser.add_argument('--save-csv', default='', help='Optional CSV path for the per-image results.')
    parser.add_argument('--output-dir', default='', help='Optional folder for <image>_{input,model,dummy}.png.')
    args = parser.parse_args()

    device = pick_device(args.device)
    model, config = load_lae_2d(Path(args.checkpoint or Path(args.run_dir) / 'best.pt'), device)
    model.eval()
    model.bypass = args.bypass_lae
    hologram = Hologram2D.from_phase_file(
        run_phase_file(config, PHASE_FILE_2D), tmx=model.tmx, layer_id=int(config.get('layer_id', 0)),
        distance_m=float(config.get('target_distance_mm', 100.0)) / 1000.0, image_hw=model.image_hw, device=device,
    )
    baseline = IdentityEncoder(model.tmx)
    dataset = ImageFolder(args.images_dir, model.image_hw)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    out_dir = Path(args.output_dir) if args.output_dir else None

    width = max(len(p.name) for p in dataset.files) + 2
    header = f"{'image':<{width}}  {'model_psnr':>12}  {'dummy_psnr':>12}  {'delta':>8}"
    print(header)
    print('-' * len(header))
    rows = []
    for images in loader:
        images = images.to(device)
        preds_model, preds_baseline = hologram(model, images), hologram(baseline, images)
        for i, (model_psnr, baseline_psnr) in enumerate(zip(psnr(preds_model, images).tolist(), psnr(preds_baseline, images).tolist())):
            path = dataset.files[len(rows)]
            print(f'{path.name:<{width}}  {model_psnr:>12.3f}  {baseline_psnr:>12.3f}  {model_psnr - baseline_psnr:>+8.3f}')
            rows.append({'image': path.name, 'model_psnr': model_psnr, 'dummy_psnr': baseline_psnr, 'delta': model_psnr - baseline_psnr})
            if out_dir is not None:
                save_png(images[i], out_dir / f'{path.stem}_input.png')
                save_png(preds_model[i], out_dir / f'{path.stem}_model.png')
                save_png(preds_baseline[i], out_dir / f'{path.stem}_dummy.png')

    mean_model = sum(r['model_psnr'] for r in rows) / len(rows)
    mean_baseline = sum(r['dummy_psnr'] for r in rows) / len(rows)
    print('-' * len(header))
    print(f"{'AVERAGE':<{width}}  {mean_model:>12.3f}  {mean_baseline:>12.3f}  {mean_model - mean_baseline:>+8.3f}")
    if args.save_csv:
        with open(args.save_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'\n[info] Results saved to {args.save_csv}')


if __name__ == '__main__':
    main()
