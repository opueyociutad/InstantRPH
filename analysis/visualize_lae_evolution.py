#!/usr/bin/env python3
"""
Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
Conference: SIGGRAPH Asia 2026
Project Page: https://opueyociutad.github.io/InstantRPH_Web/

Build evolution videos from LAE/hologram/PSNR snapshots saved by train_2d.py.

Reads a run's viz_snapshots/ directory (snap_*/lae_scale_repeatNN.png,
lae_bias_repeatNN.png, hologram_*.png, meta.json, psnr_history.csv) and
produces standalone H.264 MP4 animations for each per-repeat LAE channel plus
PSNR. Purely post-hoc: does not touch the model or re-run anything, so it can
be regenerated any time from the saved snapshots. LAE PNGs are already
normalized by train_2d.py against a fixed range (see LAE_SCALE_RANGE/
LAE_BIAS_RANGE in train_2d.py), so frames are consistent across the run without any
further rescaling here.
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLOR_MODEL = '#2a78d6'   # slot 1, blue
COLOR_GRID = '#d8d7d0'
COLOR_TEXT = '#0b0b0b'
COLOR_SURFACE = '#fcfcfb'

REPEAT_RE = re.compile(r'^lae_(scale|bias)_repeat(\d+)\.png$')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Animate LAE/hologram/PSNR evolution from train_2d.py snapshots.')
    parser.add_argument('--run-dir', type=str, required=True, help='Run directory containing viz_snapshots/ (i.e. --output-dir passed to train_2d.py).')
    parser.add_argument('--viz-dir', type=str, default='', help='Snapshot directory. Defaults to <run-dir>/viz_snapshots.')
    parser.add_argument('--output-dir', type=str, default='', help='Where to write frames/animations. Defaults to the viz-dir itself.')
    parser.add_argument('--fps', type=float, default=6.0, help='Frames per second for GIF/MP4 output.')
    parser.add_argument('--frame-height', type=int, default=280, help='Height (px) each per-snapshot tile is resized to before assembly.')
    parser.add_argument('--debug-first-two-repeats', action='store_true', help='Only render repeat 00 and repeat 01 for scale/bias MP4s.')
    parser.add_argument('--scale-tone-min', type=float, default=0.05, help='Fixed lower bound for scale-video tone mapping.')
    parser.add_argument('--scale-tone-max', type=float, default=0.95, help='Fixed upper bound for scale-video tone mapping.')
    parser.add_argument('--bias-tone-min', type=float, default=0.05, help='Fixed lower bound for bias-video tone mapping.')
    parser.add_argument('--bias-tone-max', type=float, default=0.95, help='Fixed upper bound for bias-video tone mapping.')
    return parser.parse_args()


def _load_snapshots(viz_dir: Path) -> list[dict[str, Any]]:
    snapshots = []
    for snap_dir in sorted(viz_dir.glob('snap_*')):
        try:
            snap_idx = int(snap_dir.name.split('_')[1])
        except (IndexError, ValueError):
            continue
        meta = {'epoch': 0, 'global_step': 0, 'hologram_psnr': []}
        meta_path = snap_dir / 'meta.json'
        if meta_path.exists():
            meta.update(json.loads(meta_path.read_text()))

        repeat_scale: dict[int, Path] = {}
        repeat_bias: dict[int, Path] = {}
        for p in snap_dir.glob('lae_*_repeat*.png'):
            m = REPEAT_RE.match(p.name)
            if not m:
                continue
            kind, idx_str = m.group(1), m.group(2)
            (repeat_scale if kind == 'scale' else repeat_bias)[int(idx_str)] = p

        snapshots.append({
            'snap_idx': snap_idx,
            'dir': snap_dir,
            'epoch': meta['epoch'],
            'global_step': meta['global_step'],
            'scale_path': snap_dir / 'lae_scale.png',
            'bias_path': snap_dir / 'lae_bias.png',
            'repeat_scale': repeat_scale,
            'repeat_bias': repeat_bias,
            'holograms': sorted(snap_dir.glob('hologram_*.png')),
            'hologram_psnr': meta['hologram_psnr'],
        })
    return snapshots


def _load_psnr_history(csv_path: Path) -> list[dict[str, float]]:
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            rows.append({
                'snap_idx': int(row['snap_idx']),
                'epoch': int(row['epoch']),
                'global_step': int(row['global_step']),
                'val_psnr': float(row['val_psnr']),
                'val_dummy_psnr': float(row['val_dummy_psnr']) if row.get('val_dummy_psnr') else float('nan'),
            })
    return sorted(rows, key=lambda r: r['global_step'])


def _tone_map_frame(image: Image.Image, tone_min: float, tone_max: float) -> Image.Image:
    array = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
    if tone_max <= tone_min:
        return image.convert('RGB')
    array = np.clip((array - tone_min) / max(tone_max - tone_min, 1e-6), 0.0, 1.0)
    return Image.fromarray((array * 255.0).astype(np.uint8), mode='RGB')


def _encode_mp4(frames: list[Image.Image], out_path: Path, fps: float) -> None:
    if not frames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg not found on PATH. Install ffmpeg to write MP4 outputs.')

    width, height = frames[0].size
    for frame in frames[1:]:
        if frame.size != (width, height):
            raise ValueError(f'All frames must have the same size. Expected {(width, height)}, got {frame.size}.')

    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}',
        '-r', str(fps),
        '-i', 'pipe:0',
        '-c:v', 'libx264',
        '-profile:v', 'high',
        '-preset', 'slow',
        '-crf', '14',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        str(out_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in frames:
            proc.stdin.write(frame.convert('RGB').tobytes())
        proc.stdin.close()
        return_code = proc.wait()
    finally:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()

    if return_code != 0:
        raise RuntimeError(f'ffmpeg failed while writing {out_path} (exit code {return_code}).')

    print(f'[MP4] Saved {out_path} ({len(frames)} frames)')


def _build_lae_repeat_channel_frames(
    snapshots: list[dict[str, Any]],
    channel: str,
    repeat_idx: int,
    tone_min: float,
    tone_max: float,
) -> list[Image.Image]:
    key = f'repeat_{channel}'
    have = [s for s in snapshots if repeat_idx in s[key]]
    if not have:
        return []

    frames: list[Image.Image] = []
    for s in have:
        frame = Image.open(s[key][repeat_idx]).convert('RGB')
        frames.append(_tone_map_frame(frame, tone_min, tone_max))
    return frames


_FONT_DIR = Path(__file__).resolve().parent / 'fonts' / 'linux-libertine'

_FONT_CANDIDATES = (
    _FONT_DIR / 'LinLibertine_R.otf',
    Path('/usr/share/fonts/opentype/linux-libertine/LinLibertine_R.otf'),
    Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),
    Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
    Path('/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'),
)


def _find_truetype_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in _FONT_CANDIDATES:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def _overlay_psnr_text(image: Image.Image, psnr_value: float, margin: int = 8) -> Image.Image:
    """Draw 'PSNR: xx.xx dB' in the top-right corner: white fill, thin black
    stroke, no background box, for a clean/professional look on any image."""
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font_size = max(12, round(image.height * 0.065))
    font = _find_truetype_font(font_size)
    text = f'PSNR: {psnr_value:.2f} dB'
    stroke_width = max(1, round(font_size * 0.04))

    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    text_w = bbox[2] - bbox[0]
    x = image.width - margin - text_w - bbox[0]
    y = margin
    draw.text(
        (x, y), text, font=font,
        fill=(255, 255, 255), stroke_width=stroke_width, stroke_fill=(0, 0, 0),
    )
    return image


def _build_hologram_frames(snapshots: list[dict[str, Any]], hologram_idx: int) -> list[Image.Image]:
    have = [s for s in snapshots if s['holograms']]
    if not have:
        return []
    frames: list[Image.Image] = []
    for s in have:
        if hologram_idx >= len(s['holograms']):
            continue
        frame = Image.open(s['holograms'][hologram_idx]).convert('RGB')
        psnr_list = s.get('hologram_psnr') or []
        if hologram_idx < len(psnr_list):
            frame = _overlay_psnr_text(frame, psnr_list[hologram_idx])
        frames.append(frame)
    return frames


def _build_psnr_frames(history: list[dict[str, float]], frame_height: int) -> list[Image.Image]:
    if not history:
        print('[PSNR] No psnr_history.csv rows found — skipping PSNR frames.')
        return []

    steps = [r['global_step'] for r in history]
    model_vals = [r['val_psnr'] for r in history]
    y_min = min(model_vals)
    y_max = max(model_vals)
    y_pad = max(0.5, (y_max - y_min) * 0.1)

    frames: list[Image.Image] = []
    dpi = 100
    fig_w, fig_h = 5.0, frame_height / dpi
    for idx in range(1, len(history) + 1):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        fig.patch.set_facecolor(COLOR_SURFACE)
        ax.set_facecolor(COLOR_SURFACE)

        ax.plot(steps[:idx], model_vals[:idx], color=COLOR_MODEL, linewidth=2.0, marker='o', markersize=3, label='Model (LAE)')

        ax.set_xlim(min(steps) - 0.5, max(steps) + 0.5)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xlabel('Optimizer step', color=COLOR_TEXT, fontsize=9)
        ax.set_ylabel('Val PSNR (dB)', color=COLOR_TEXT, fontsize=9)
        ax.tick_params(colors=COLOR_TEXT, labelsize=8)
        ax.grid(True, color=COLOR_GRID, linewidth=0.8, alpha=0.8)
        for spine in ax.spines.values():
            spine.set_color(COLOR_GRID)

        fig.tight_layout()
        fig.canvas.draw()
        frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy(), mode='RGB')
        plt.close(fig)
        frames.append(frame)
    return frames


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    viz_dir = Path(args.viz_dir) if args.viz_dir else (run_dir / 'viz_snapshots')
    output_dir = Path(args.output_dir) if args.output_dir else viz_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not viz_dir.exists():
        raise FileNotFoundError(f'Snapshot directory not found: {viz_dir}')

    snapshots = _load_snapshots(viz_dir)
    if not snapshots:
        raise ValueError(f'No snap_* snapshot directories found under {viz_dir}')
    print(f'[Load] Found {len(snapshots)} snapshots under {viz_dir}')

    history = _load_psnr_history(viz_dir / 'psnr_history.csv')

    _encode_mp4(_build_hologram_frames(snapshots, 0), output_dir / 'hologram_00_evolution.mp4', args.fps)
    _encode_mp4(_build_hologram_frames(snapshots, 1), output_dir / 'hologram_01_evolution.mp4', args.fps)
    _encode_mp4(_build_psnr_frames(history, args.frame_height), output_dir / 'psnr_evolution.mp4', args.fps)

    repeat_indices = sorted(
        set().union(
            *(set(s['repeat_scale'].keys()) | set(s['repeat_bias'].keys()) for s in snapshots)
        )
    ) if snapshots else []
    if args.debug_first_two_repeats:
        repeat_indices = repeat_indices[:2]
        print(f'[Debug] Limiting repeat export to first two indices: {repeat_indices}')
    for repeat_idx in repeat_indices:
        scale_frames = _build_lae_repeat_channel_frames(
            snapshots,
            'scale',
            repeat_idx,
            args.scale_tone_min,
            args.scale_tone_max,
        )
        bias_frames = _build_lae_repeat_channel_frames(
            snapshots,
            'bias',
            repeat_idx,
            args.bias_tone_min,
            args.bias_tone_max,
        )
        _encode_mp4(scale_frames, output_dir / f'lae_scale_repeat{repeat_idx:02d}_evolution.mp4', args.fps)
        _encode_mp4(bias_frames, output_dir / f'lae_bias_repeat{repeat_idx:02d}_evolution.mp4', args.fps)

    print(f'[Done] Outputs written to {output_dir}')


if __name__ == '__main__':
    main()
