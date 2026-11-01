#!/usr/bin/env python3
"""
Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
Conference: SIGGRAPH Asia 2026
Project Page: https://opueyociutad.github.io/InstantRPH_Web/

Evaluate an Instant RPH run (typically merge_rgb_3d.py output) against the amplitude-discard (AD)
baseline (without LAEs) on the run's scenes: focal-stack L1 loss and PSNR, plus optional outputs.

Writes to --output-dir (default: output/3d/comparisons/<run folder>):
  validation_target_scene_rgb_means.json   mean RGB of each scene's first target plane
  validation_rgb_output.{pt,png}           first batch's focal stack, and a GT / AD / AD+Q / OURS / OURS+Q
                                           preview (+Q: SLM phase quantized with the measured LUTs, DC blocked)
  validation_phase_patterns/               with --save-phase-patterns: SLM phase (0..2pi) per scene
--export-dir DIR saves, per scene and method, the rendered focal stacks (color-balanced, as in the PSNR), the
target focal stacks and the SLM phases.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import islice
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from instantrph.common import (
    OUTPUT_DIR, WAVELENGTH_CHANNELS, find_checkpoint, load_json, pick_device, prepare_volume, run_focus_distances, select_stack_channels,
)
from instantrph.data_3d import run_eval_loader
from instantrph.metrics import colorbalance
from instantrph.models import IdentityEncoder3D, load_lae_3d
from instantrph.quantization import quantize_slm_field
from instantrph.renderer_3d import LayeredFocalStackRenderer, evaluate
from instantrph.viz import export_scenes, scene_file_name, to_pil

DTYPES = {'float16': torch.float16, 'float32': torch.float32, 'bfloat16': torch.bfloat16}
PREVIEW_ROWS = (('GT', (200, 240, 200)), ('AD', (255, 210, 170)), ('AD+Q', (255, 220, 120)), ('OURS', (255, 210, 170)), ('OURS+Q', (255, 220, 120)))


@torch.no_grad()
def render_preview(model, renderer: LayeredFocalStackRenderer, batch, channels, device, *, quantize: bool) -> torch.Tensor:
    """Color-balanced focal stack [B, S, C, H, W] in [0, 1], optionally through the quantized SLM."""
    rgb_vol, alpha_vol = prepare_volume(batch['rgb_vol'], batch['alpha_vol'], device, channels)
    target_stack = select_stack_channels(batch['focal_stack'].to(device=device, dtype=torch.float32), channels)
    field = renderer.render_field(model(rgb_vol), alpha_vol)
    if quantize:
        field = quantize_slm_field(field, channels)
    pred_stack = torch.stack(list(renderer.focal_planes(field)), dim=1)
    return colorbalance(pred_stack, target_stack).clamp(0, 1).cpu()


def _draw_label(canvas: Image.Image, label: str, xy: tuple[int, int], fill: tuple[int, int, int] = (255, 255, 255)) -> None:
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    left, top, right, bottom = draw.textbbox(xy, label, font=font)
    draw.rectangle((left - 2, top - 2, right + 2, bottom + 2), fill=(0, 0, 0))
    draw.text(xy, label, fill=fill, font=font)


def save_preview_png(samples: list[list[torch.Tensor]], focus_distances_m: list[float], path: Path) -> None:
    """One section per sample, one row per PREVIEW_ROWS entry, up to 5 evenly spaced focal planes."""
    stride = max(1, samples[0][0].shape[1] // 5)
    planes = list(range(0, samples[0][0].shape[1], stride))[:5]
    tile = to_pil(samples[0][0][0, 0]).convert('RGB')
    tile_w, tile_h = tile.width, tile.height
    label_w, row_gap, section_gap, title_h = 88, 8, 18, 28
    section_w = len(planes) * tile_w
    canvas = Image.new('RGB', (label_w + len(samples) * (section_w + section_gap) - section_gap,
                               title_h + len(PREVIEW_ROWS) * (tile_h + row_gap) - row_gap), color=(18, 18, 18))
    for row, (label, fill) in enumerate(PREVIEW_ROWS):
        _draw_label(canvas, label, (8, title_h + row * (tile_h + row_gap) + 4), fill)
    for index, stacks in enumerate(samples):
        x0 = label_w + index * (section_w + section_gap)
        _draw_label(canvas, f'Sample {index}', (x0 + 8, 6))
        for row, stack in enumerate(stacks):
            for column, plane in enumerate(planes):
                tile = Image.new('RGB', (tile_w, tile_h), color=(32, 32, 32))
                tile.paste(to_pil(stack[0, plane]).convert('RGB'), (0, 0))
                _draw_label(tile, f'S{plane + 1} {focus_distances_m[plane] * 1000.0:.2f} mm', (6, tile_h - 18))
                canvas.paste(tile, (x0 + column * tile_w, title_h + row * (tile_h + row_gap)))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    print(f'Saved RGB preview PNG to: {path}')


@torch.no_grad()
def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_json(run_dir / 'config.json')
    checkpoint = Path(args.checkpoint) if args.checkpoint else find_checkpoint(run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR / '3d' / 'comparisons' / run_dir.resolve().name
    device = pick_device(args.device)
    channels = WAVELENGTH_CHANNELS[args.wavelength]
    tmx = args.tmx or int(config['tmx'])

    loader, split = run_eval_loader(
        config, data_root=args.data_root or None, split=args.eval_split, batch_size=args.batch_size or int(config.get('batch_size', 1)),
        num_workers=args.num_workers if args.num_workers >= 0 else int(config.get('num_workers', 2)),
    )
    batches = list(islice(loader, args.max_batches)) if args.max_batches > 0 else list(loader)
    preview_batches = list(islice(loader, args.max_batches if args.max_batches > 0 else 5))
    first = batches[0]
    image_hw = tuple(first['rgb_vol'].shape[-2:])
    focus_m = run_focus_distances(config, first['focal_stack'].shape[1])

    model = load_lae_3d(checkpoint, tmx=tmx, channels=channels).to(device).eval()
    baseline = IdentityEncoder3D(tmx=tmx, num_rgb_channels=len(channels))
    renderer_args = dict(image_hw=image_hw, focus_distances_m=focus_m, tmx=tmx, channels=channels, device=device)
    renderer = LayeredFocalStackRenderer.from_run_config(config, **renderer_args, compile=False)
    optimized = LayeredFocalStackRenderer.from_run_config(
        config, **renderer_args, compile=True, compute_dtype=DTYPES[args.compute_dtype], use_autocast=args.use_autocast,
    ) if args.optimize else None
    ours_renderer = optimized or renderer
    scenes = [name for batch in batches for name in batch['scene_name']]
    print(f'Run: {run_dir} | checkpoint: {checkpoint} | device: {device}')
    print(f'Eval split: {split} | scenes ({len(scenes)}): {", ".join(dict.fromkeys(scenes))}')
    print(f'Wavelengths: {["RGB"[c] for c in channels]} | tmx: {tmx} | focus planes (mm): {[round(d * 1000, 3) for d in renderer.focus_distances_m]}')

    records = []
    for batch in batches:
        target_stack = select_stack_channels(batch['focal_stack'].to(dtype=torch.float32), channels)
        for i, scene in enumerate(batch['scene_name']):
            records.append({'scene_name': scene, 'channel_indices': list(channels),
                            'first_target_plane_rgb_mean': target_stack[i, 0].mean(dim=(-1, -2)).tolist()})
    means_path = output_dir / 'validation_target_scene_rgb_means.json'
    means_path.parent.mkdir(parents=True, exist_ok=True)
    means_path.write_text(json.dumps({'records': records, 'num_records': len(records)}, indent=2) + '\n')

    if not args.no_rgb_output:
        rgb_vol, alpha_vol = prepare_volume(first['rgb_vol'], first['alpha_vol'], device, channels)
        target_stack = select_stack_channels(first['focal_stack'].to(device=device, dtype=torch.float32), channels)
        pred_stack = colorbalance(ours_renderer.render_stack(model(rgb_vol), alpha_vol), target_stack)
        torch.save({'pred_stack': pred_stack.cpu(), 'target_stack': target_stack.cpu(), 'channel_indices': list(channels),
                    'rgb_output_shape': list(pred_stack.shape)}, output_dir / 'validation_rgb_output.pt')
        samples = []
        for batch in preview_batches:
            target = select_stack_channels(batch['focal_stack'].to(dtype=torch.float32), channels)
            samples.append([target] + [
                render_preview(encoder, method_renderer, batch, channels, device, quantize=quantize)
                for encoder, method_renderer in ((baseline, renderer), (model, ours_renderer)) for quantize in (False, True)
            ])
        save_preview_png(samples, renderer.focus_distances_m, output_dir / 'validation_rgb_output.png')

    if args.save_phase_patterns:
        for folder, encoder, method_renderer in (('checkpoint_model', model, ours_renderer), ('raw_phase_identity_uncorrected', baseline, renderer)):
            out = output_dir / 'validation_phase_patterns' / folder
            out.mkdir(parents=True, exist_ok=True)
            for batch in batches:
                rgb_vol, alpha_vol = prepare_volume(batch['rgb_vol'], batch['alpha_vol'], device, channels)
                phase = torch.angle(method_renderer.render_field(encoder(rgb_vol), alpha_vol)).cpu() + torch.pi
                for i, scene in enumerate(batch['scene_name']):
                    torch.save(phase[i], out / f'{scene_file_name(scene)}.pt')
        print(f'Saved phase patterns to: {output_dir / "validation_phase_patterns"}')

    if args.export_dir:
        methods = {'ours': lambda rgb, alpha: renderer.render_field(model(rgb), alpha),
                   'baseline': lambda rgb, alpha: renderer.render_field(baseline(rgb), alpha)}
        export_scenes(methods, renderer, batches, channels, device, Path(args.export_dir))

    evaluations = [('AD baseline', baseline, renderer), ('Ours', model, renderer)]
    if optimized is not None:
        evaluations.append((f'Ours (optimized renderer, {args.compute_dtype})', model, optimized))
    for name, encoder, method_renderer in evaluations:
        start = time.perf_counter()
        loss, psnr_sum, count = evaluate(encoder, method_renderer, batches, channels, device)
        print(f'{name}: loss={loss / count:.6f} PSNR={psnr_sum / count:.3f} dB ({time.perf_counter() - start:.2f}s)')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', required=True, help='Run directory with config.json and best.pt / last.pt.')
    parser.add_argument('--checkpoint', default='', help='Default: best.pt (or last.pt) in --run-dir.')
    parser.add_argument('--data-root', default='', help="Default: the run's data root.")
    parser.add_argument('--eval-split', default='auto', choices=['auto', 'val', 'train', 'all'],
                        help="The run's validation or training scenes, or every scene ('auto': val if the run has any).")
    parser.add_argument('--wavelength', default='rgb', choices=list(WAVELENGTH_CHANNELS), help='Wavelengths to evaluate.')
    parser.add_argument('--tmx', type=int, default=0, help='Evaluate only the first N frames (default: the trained tmx).')
    parser.add_argument('--batch-size', type=int, default=0, help="Default: the run's batch size.")
    parser.add_argument('--num-workers', type=int, default=-1, help="Default: the run's value.")
    parser.add_argument('--max-batches', type=int, default=5, help='Evaluate the first N batches (0: all).')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--output-dir', default='', help='Default: output/3d/comparisons/<run folder>.')
    parser.add_argument('--no-rgb-output', action='store_true', help='Skip validation_rgb_output.{pt,png}.')
    parser.add_argument('--save-phase-patterns', action='store_true')
    parser.add_argument('--export-dir', default='', help='Export per-scene focal stacks and phase patterns here.')
    parser.add_argument('--optimize', action='store_true', help='Render ours with torch.compile and --compute-dtype convolutions.')
    parser.add_argument('--compute-dtype', default='float16', choices=list(DTYPES))
    parser.add_argument('--use-autocast', action='store_true', help='With --optimize: CUDA autocast in the layer loop.')
    return parser.parse_args()


if __name__ == '__main__':
    main()
