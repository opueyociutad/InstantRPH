#!/usr/bin/env python3
"""
Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
Conference: SIGGRAPH Asia 2026
Project Page: https://opueyociutad.github.io/InstantRPH_Web/

Evaluate a trained 2D LAE on the validation set: reconstructions and per-image PSNR / SSIM
for the LAE and the amplitude-discard baseline.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from PIL import Image

from instantrph.common import DATASET_2D_DIR, OUTPUT_DIR, PHASE_FILE_2D, pick_device, run_phase_file, seed_all
from instantrph.data_2d import build_loaders
from instantrph.hologram_2d import Hologram2D
from instantrph.metrics import colorbalance, psnr, ssim
from instantrph.models import IdentityEncoder, load_lae_2d
from instantrph.viz import to_pil

TWO_PI = float(2.0 * np.pi)


def _channel_frame_mosaic(tensor: torch.Tensor) -> np.ndarray:
    """[C, tmx, H, W] -> 2D array with one row per channel and one column per frame."""
    tensor = tensor.detach().float().cpu()
    return np.concatenate([np.concatenate([tensor[c, t].numpy() for t in range(tensor.shape[1])], axis=1) for c in range(tensor.shape[0])], axis=0)


def _save_rgb_frames(tensor: torch.Tensor, out_dir: Path, name: str, to_unit: Callable[[np.ndarray], np.ndarray]) -> None:
    """Save each frame of a [C, tmx, H, W] tensor as an RGB PNG, mapped to [0, 1] by `to_unit`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in range(tensor.shape[1]):
        rgb = np.clip(to_unit(tensor[:3, t].permute(1, 2, 0).numpy()), 0.0, 1.0)
        Image.fromarray((rgb * 255.0).astype(np.uint8), mode='RGB').save(out_dir / f'{name}{t:02d}.png')


def _show(ax, array: np.ndarray, cmap: str, vmin: float, vmax: float, title: str, fig) -> None:
    im = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


@torch.no_grad()
def save_single_sample_plots(hologram: Hologram2D, model: torch.nn.Module, image: torch.Tensor, out_dir: Path, index: int) -> None:
    """Optical intermediates of one image: random phases, SLM amplitude/phase before amplitude
    discard, and the per-frame reconstructions before time averaging."""
    out_dir.mkdir(parents=True, exist_ok=True)
    slm, recon = hologram.fields(model, image.unsqueeze(0))
    phase_pattern = hologram.phase_pattern.detach().cpu()
    frames = colorbalance((recon.abs() ** 2)[0].cpu(), image.cpu())
    slm_amplitude, slm_phase = slm.abs()[0].cpu(), slm.angle()[0].cpu()
    prefix = f'{index:06d}'

    _save_rgb_frames(phase_pattern, out_dir / f'{prefix}_phase_pattern_raw', 'phase_pattern_rgb_tmx', lambda x: x / TWO_PI)
    _save_rgb_frames(frames, out_dir / f'{prefix}_tm_predictions_raw', 'tm_pred_rgb_frame', lambda x: x)
    _save_rgb_frames(slm_amplitude, out_dir / f'{prefix}_amplitude_slm_raw', 'amplitude_slm_rgb_frame', lambda x: x)
    _save_rgb_frames(slm_phase, out_dir / f'{prefix}_phase_slm_raw', 'phase_slm_rgb_frame', lambda x: (x + np.pi) / TWO_PI)

    panels = []
    for name, tensor, cmap, value_range, title in (
        ('phase_pattern_at_slm', phase_pattern, 'hsv', (0.0, TWO_PI), 'Phase pattern at SLM'),
        ('tm_predictions_before_averaging', frames, 'magma', None, 'Time-multiplexed predictions before averaging'),
        ('amplitude_at_slm_before_discarding', slm_amplitude, 'magma', None, 'Amplitude at SLM before discarding'),
        ('phase_at_slm', slm_phase, 'hsv', (float(-np.pi), float(np.pi)), 'Phase at SLM'),
    ):
        mosaic = _channel_frame_mosaic(tensor)
        vmin, vmax = value_range or (float(np.min(mosaic)), float(np.max(mosaic)))
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        _show(ax, mosaic, cmap, vmin, vmax, title, fig)
        fig.savefig(out_dir / f'{prefix}_{name}.png', dpi=200)
        plt.close(fig)
        panels.append((mosaic, cmap, vmin, vmax, title))

    fig, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
    for ax, (mosaic, cmap, vmin, vmax, title) in zip(axes.flat, panels):
        _show(ax, mosaic, cmap, vmin, vmax, title, fig)
    fig.suptitle(f'Single-sample optical visualization for sample {prefix}')
    fig.savefig(out_dir / f'{prefix}_summary.png', dpi=200)
    plt.close(fig)
    print(f'[Viz] Saved single-sample plots to {out_dir}')


def save_phase_pattern_pngs(phase_pattern: torch.Tensor, out_dir: Path) -> None:
    """One HSV-colored PNG (0..2pi) per channel and frame of a [C, tmx, H, W] phase pattern."""
    out_dir.mkdir(parents=True, exist_ok=True)
    phase_pattern = phase_pattern.detach().cpu()
    for c in range(phase_pattern.shape[0]):
        for t in range(phase_pattern.shape[1]):
            normalized = np.clip(phase_pattern[c, t].numpy() / (TWO_PI + 1e-8), 0.0, 1.0)
            colored = (plt.get_cmap('hsv')(normalized)[:, :, :3] * 255.0).astype(np.uint8)
            Image.fromarray(colored, mode='RGB').save(out_dir / f'phase_ch{c:02d}_tmx{t:02d}_0_to_2pi.png')
    print(f'[Init] Saved {phase_pattern.shape[0] * phase_pattern.shape[1]} phase pattern PNGs to {out_dir}')


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    device = pick_device(args.device)
    model, config = load_lae_2d(args.checkpoint, device)
    model.eval()

    # Optics default to the trained ones. tmx is fixed by the LAE's shape; the distance and
    # phase layer may be changed to evaluate a trained LAE under different optics.
    tmx = model.tmx
    if args.tmx is not None and args.tmx != tmx:
        raise ValueError(f'--tmx {args.tmx} does not match the checkpoint (trained with tmx={tmx}).')
    layer_id = args.layer_id if args.layer_id is not None else int(config.get('layer_id', 0))
    distance_mm = args.target_distance_mm if args.target_distance_mm is not None else float(config.get('target_distance_mm', 100.0))
    image_hw = model.image_hw
    print(f'[Init] Optics: tmx={tmx} layer_id={layer_id} target_distance_mm={distance_mm}')
    print(f'[Init] Device: {device}')
    print(f'[Init] Image size (R, C): {image_hw}')

    _, val_loader = build_loaders(
        args.datasets_root, args.train_subdir, args.val_subdir, image_hw=image_hw, batch_size=args.batch_size,
        num_workers=args.num_workers, debug_sample_index=args.debug_sample_index if args.debug_single_sample else None,
    )
    print(f'[Data] Val samples: {len(val_loader.dataset)} | Batch size: {args.batch_size}')
    hologram = Hologram2D.from_phase_file(
        args.phase_file or run_phase_file(config, PHASE_FILE_2D), tmx=tmx, layer_id=layer_id, distance_m=distance_mm / 1000.0,
        image_hw=image_hw, device=device,
    )

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR / '2d' / 'eval' / Path(args.checkpoint).resolve().parent.name
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    if args.save_input_phase_pattern:
        save_phase_pattern_pngs(hologram.phase_pattern, Path(args.input_phase_pattern_output_dir or output_dir) / 'input_phase_patterns')

    baseline = IdentityEncoder(tmx)
    rows = []
    for images in tqdm.tqdm(val_loader):
        images = images.to(device)
        preds_ours, preds_baseline = hologram(model, images), hologram(baseline, images)
        metrics = zip(psnr(preds_baseline, images), ssim(preds_baseline, images), psnr(preds_ours, images), ssim(preds_ours, images))
        for i, (psnr_baseline, ssim_baseline, psnr_ours, ssim_ours) in enumerate(metrics):
            index = len(rows)
            for name, image in (('input', images[i]), ('pred_ours', preds_ours[i]), ('pred_ad_baseline', preds_baseline[i]), ('target', images[i])):
                to_pil(image).save(images_dir / f'{index:06d}_{name}.png')
            if args.save_single_sample_plots and index == args.single_sample_index:
                save_single_sample_plots(hologram, model, images[i], Path(args.single_sample_output_dir or output_dir / 'single_sample_plots'), index)
            rows.append({'sample_idx': index, 'psnr_ad_baseline': psnr_baseline.item(), 'ssim_ad_baseline': ssim_baseline.item(),
                         'psnr_ours': psnr_ours.item(), 'ssim_ours': ssim_ours.item()})
    if args.save_single_sample_plots and args.single_sample_index >= len(rows):
        print(f'[Viz] Requested single-sample index {args.single_sample_index} was not found in the evaluated split.')

    csv_path = output_dir / 'metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\n[Results] Saved {len(rows)} samples to {images_dir}')
    print(f'[Results] Metrics saved to {csv_path}')

    def column(key: str) -> list[float]:
        return [row[key] for row in rows]

    for name, key in (('AD Baseline', 'ad_baseline'), ('OURS', 'ours')):
        p, s = column(f'psnr_{key}'), column(f'ssim_{key}')
        print(f'\n[{name}] Mean PSNR: {sum(p) / len(p):.2f} dB')
        print(f'[{name}] Mean SSIM: {sum(s) / len(s):.4f}')
        print(f'[{name}] PSNR range: [{min(p):.2f}, {max(p):.2f}] dB')
        print(f'[{name}] SSIM range: [{min(s):.4f}, {max(s):.4f}]')
    psnr_gain = [o - b for o, b in zip(column('psnr_ours'), column('psnr_ad_baseline'))]
    ssim_gain = [o - b for o, b in zip(column('ssim_ours'), column('ssim_ad_baseline'))]
    print(f'\n[Improvement] Mean PSNR gain: {sum(psnr_gain) / len(psnr_gain):+.2f} dB')
    print(f'[Improvement] Mean SSIM gain: {sum(ssim_gain) / len(ssim_gain):+.4f}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True, help='Checkpoint (.pt) of a train_2d.py run.')
    parser.add_argument('--output-dir', default='', help='Default: output/2d/eval/<run folder of --checkpoint>.')
    parser.add_argument('--datasets-root', default=str(DATASET_2D_DIR))
    parser.add_argument('--train-subdir', default='DIV2K_train_HR')
    parser.add_argument('--val-subdir', default='DIV2K_valid_HR')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug-single-sample', action='store_true', help='Evaluate only one (training) image.')
    parser.add_argument('--debug-sample-index', type=int, default=0)
    parser.add_argument('--tmx', type=int, default=None, help='Defaults to the trained value (and must match it).')
    parser.add_argument('--layer-id', type=int, default=None, help='Phase tensor layer. Defaults to the trained value.')
    parser.add_argument('--target-distance-mm', type=float, default=None, help='Defaults to the trained value.')
    parser.add_argument('--phase-file', type=Path, default=None, help='Defaults to the trained one.')
    parser.add_argument('--save-single-sample-plots', action='store_true', help='Save the optical intermediates of one image.')
    parser.add_argument('--single-sample-index', type=int, default=0)
    parser.add_argument('--single-sample-output-dir', default='', help='Default: <output-dir>/single_sample_plots.')
    parser.add_argument('--save-input-phase-pattern', action='store_true', help='Save the random phases as HSV PNGs.')
    parser.add_argument('--input-phase-pattern-output-dir', default='', help='Default: <output-dir>.')
    return parser.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())
