#!/usr/bin/env python3
"""
Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
Conference: SIGGRAPH Asia 2026
Project Page: https://opueyociutad.github.io/InstantRPH_Web/

Render the learned scale/bias maps of a trained 2D LAE: per-channel grids, RGB maps per
time-multiplexed frame, frame-averaged RGB maps (raw and Gaussian-smoothed) and their statistics.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `instantrph`

from instantrph.common import OUTPUT_DIR  # noqa: E402
from instantrph.models import load_lae_2d  # noqa: E402

BACKGROUND = (20, 20, 20)


def normalize(x: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Min-max normalize to [0, 1] with one range for the whole tensor. Returns (x, min, max)."""
    vmin, vmax = float(x.min()), float(x.max())
    return (torch.zeros_like(x) if abs(vmax - vmin) < 1e-12 else (x - vmin) / (vmax - vmin)), vmin, vmax


def to_uint8(x: torch.Tensor):
    return (x.clamp(0.0, 1.0).numpy() * 255.0).astype('uint8')


def rgb_image(chw: torch.Tensor) -> Image.Image:
    return Image.fromarray(to_uint8(chw.permute(1, 2, 0)), mode='RGB')


def channel_grid(x: torch.Tensor, tile_height: int, columns: int) -> Image.Image:
    """Grayscale tiles of the [C, H, W] channels, `columns` per row."""
    channels, height, width = x.shape
    tile_width = max(1, int(round(width * (tile_height / float(max(height, 1))))))
    pad, rows = 8, int(math.ceil(channels / float(columns)))
    canvas = Image.new('RGB', (columns * tile_width + (columns + 1) * pad, rows * tile_height + (rows + 1) * pad), color=BACKGROUND)
    tiles = to_uint8(F.interpolate(x.unsqueeze(0), size=(tile_height, tile_width), mode='bilinear', align_corners=False).squeeze(0))
    for i in range(channels):
        canvas.paste(Image.fromarray(tiles[i], mode='L').convert('RGB'), (pad + (i % columns) * (tile_width + pad), pad + (i // columns) * (tile_height + pad)))
    return canvas


def tile(images: list[Image.Image], horizontal: bool, gap: int = 10) -> Image.Image:
    """Concatenate images with a gap, centered on the other axis."""
    long = sum(im.width if horizontal else im.height for im in images) + gap * (len(images) - 1)
    short = max(im.height if horizontal else im.width for im in images)
    canvas = Image.new('RGB', (long, short) if horizontal else (short, long), color=BACKGROUND)
    offset = 0
    for im in images:
        canvas.paste(im, (offset, (short - im.height) // 2) if horizontal else ((short - im.width) // 2, offset))
        offset += (im.width if horizontal else im.height) + gap
    return canvas


def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    g = g / g.sum()
    kernel = torch.outer(g, g)
    return kernel / kernel.sum()


def gaussian_blur(chw: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    pad = kernel.shape[-1] // 2
    x = F.pad(chw.unsqueeze(0), (pad, pad, pad, pad), mode='reflect') if pad > 0 else chw.unsqueeze(0)
    weight = kernel.view(1, 1, *kernel.shape).repeat(chw.shape[0], 1, 1, 1)
    return F.conv2d(x, weight, groups=chw.shape[0]).squeeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', required=True, help='A train_2d.py output directory.')
    parser.add_argument('--checkpoint', default='', help='Default: <run-dir>/best.pt.')
    parser.add_argument('--output-dir', default='', help='Default: output/2d/analysis/lae_params/<run folder>.')
    parser.add_argument('--tile-height', type=int, default=180)
    parser.add_argument('--columns', type=int, default=8)
    parser.add_argument('--save-individual', action='store_true', help='Also save each channel as a grayscale PNG.')
    parser.add_argument('--gaussian-sigma', type=float, default=2.0, help='Smoothing of the frame-averaged maps.')
    parser.add_argument('--gaussian-kernel-size', type=int, default=0, help='Odd size; 0: 2*ceil(3*sigma)+1.')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.output_dir) if args.output_dir else OUTPUT_DIR / '2d' / 'analysis' / 'lae_params' / run_dir.resolve().name
    out.mkdir(parents=True, exist_ok=True)
    model, _ = load_lae_2d(Path(args.checkpoint) if args.checkpoint else run_dir / 'best.pt', torch.device('cpu'))
    kernel_size = args.gaussian_kernel_size or int(2 * math.ceil(3.0 * args.gaussian_sigma) + 1)
    kernel_size += 1 - kernel_size % 2
    kernel = gaussian_kernel(kernel_size, args.gaussian_sigma)

    stats = {}
    for name, param in (('scale', model.scale), ('bias', model.bias)):
        raw = param.detach().float().squeeze(0)  # [3 * tmx, H, W], channel-major
        torch.save(raw, out / f'{name}_map.pt')
        shared, vmin, vmax = normalize(raw)
        channel_grid(shared, args.tile_height, args.columns).save(out / f'{name}_grid_shared_range.png')
        if args.save_individual:
            (out / f'{name}_channels').mkdir(exist_ok=True)
            for i, channel in enumerate(shared):
                Image.fromarray(to_uint8(channel), mode='L').save(out / f'{name}_channels' / f'ch_{i:02d}.png')

        frames = shared.view(3, model.tmx, *shared.shape[-2:]).transpose(0, 1)  # [tmx, 3, H, W]
        (out / f'{name}_rgb_repeats').mkdir(exist_ok=True)
        for t, frame in enumerate(frames):
            rgb_image(frame).save(out / f'{name}_rgb_repeats' / f'{name}_rgb_repeat_{t:02d}.png')
        preview_height = max(1, int(round(args.tile_height / 2)))
        preview_width = max(1, int(round(frames.shape[-1] * (preview_height / max(frames.shape[-2], 1)))))
        previews = F.interpolate(frames, size=(preview_height, preview_width), mode='bilinear', align_corners=False)
        stats[name] = {'panel': tile([rgb_image(p) for p in previews], horizontal=True), 'range': [vmin, vmax],
                       'mean': float(raw.mean()), 'std': float(raw.std(unbiased=False))}
        stats[name]['panel'].save(out / f'{name}_rgb_repeats_shared_range.png')

        averaged, avg_min, avg_max = normalize(frames.mean(dim=0))
        rgb_image(averaged).save(out / f'{name}_rgb_averaged_shared_range.png')
        smoothed, smooth_min, smooth_max = normalize(gaussian_blur(frames.mean(dim=0), kernel))
        rgb_image(smoothed).save(out / f'{name}_rgb_averaged_gaussian_shared_range.png')
        stats[name].update(averaged_range=[avg_min, avg_max], averaged_gaussian_range=[smooth_min, smooth_max])

    tile([stats['scale']['panel'], stats['bias']['panel']], horizontal=False).save(out / 'scale_bias_rgb_repeats_shared_range.png')
    height, width = model.image_hw
    native = torch.zeros((height, width), dtype=kernel.dtype)
    y0, x0 = (height - kernel_size) // 2, (width - kernel_size) // 2
    native[y0:y0 + kernel_size, x0:x0 + kernel_size] = kernel
    torch.save(native, out / 'gaussian_kernel_native_resolution.pt')
    Image.fromarray(to_uint8(native / native.max().clamp_min(1e-12)), mode='L').save(out / 'gaussian_kernel_native_resolution.png')

    metadata = {'tmx': model.tmx, 'image_hw': list(model.image_hw), 'gaussian_sigma': args.gaussian_sigma, 'gaussian_kernel_size': kernel_size}
    metadata.update({name: {k: v for k, v in s.items() if k != 'panel'} for name, s in stats.items()})
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'LAE maps of {run_dir} written to {out}')


if __name__ == '__main__':
    main()
