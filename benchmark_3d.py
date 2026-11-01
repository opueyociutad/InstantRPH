#!/usr/bin/env python3
"""
Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
Conference: SIGGRAPH Asia 2026
Project Page: https://opueyociutad.github.io/InstantRPH_Web/

Hologram generation time of InstantRPH and of the ARH baseline, with one shared timing loop.

  instantrph: LAE + layered occlusion-aware propagation to the SLM + amplitude discard (render_field)
  arh:        MPI -> hard RGBD + HoloNet initial phases + per-layer propagation to the SLM + HoloNet encoding

Inputs are moved to the device first. Optical kernels and networks are prepared once for both
methods; per-frame work (encoding, compositing, propagation) is timed. Warm-up runs (torch.compile,
cuDNN autotuning) are excluded.

    python benchmark_3d.py --method instantrph --run-dir data/models_3d/lae/<merged run> --tmx 24 --optimize --use-autocast
    python benchmark_3d.py --method arh --mpi-dir path/to/rgba_layers --tmx 8
"""
from __future__ import annotations

import argparse
import statistics
import time
from itertools import islice
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image

from instantrph.arh import arh_render_field, layer_kernels, load_arh_models
from instantrph.common import ARH_MODEL_DIR, WAVELENGTH_CHANNELS, find_checkpoint, load_json, pick_device, prepare_volume, run_focus_distances
from instantrph.constants import C, R
from instantrph.data_3d import run_eval_loader
from instantrph.models import load_lae_3d
from instantrph.renderer_3d import LayeredFocalStackRenderer

DTYPES = {'float16': torch.float16, 'float32': torch.float32, 'bfloat16': torch.bfloat16}


def time_runs(fn: Callable[[], object], device: torch.device, *, warmup: int, runs: int) -> list[float]:
    def once() -> float:
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        return time.perf_counter() - start

    for _ in range(warmup):
        once()
    return [once() for _ in range(runs)]


def report(label: str, durations: list[float], holograms: int, tmx: int) -> None:
    """A hologram is one scene: tmx time-multiplexed SLM frames."""
    mean = statistics.fmean(durations)
    print(f'{label}: mean={mean:.6f}s median={statistics.median(durations):.6f}s std={statistics.pstdev(durations):.6f}s '
          f'min={min(durations):.6f}s max={max(durations):.6f}s -> {holograms / mean:.2f} holograms/s '
          f'({holograms * tmx / mean:.2f} SLM frames/s)')


def load_mpi_layers(mpi_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """RGBA layer PNGs (sorted by name, nearest first) -> amplitude [1,3,D,H,W] and alpha [1,1,D,H,W]."""
    files = sorted(p for p in mpi_dir.iterdir() if p.suffix.lower() == '.png')
    if not files:
        raise ValueError(f'No PNG files found in {mpi_dir}')
    layers = torch.stack([torch.from_numpy(np.asarray(Image.open(f).convert('RGBA'), dtype=np.float32) / 255.0) for f in files])
    layers = layers.permute(3, 0, 1, 2).unsqueeze(0)  # [1, 4, D, H, W]
    return layers[:, :3].clamp(min=0.0).sqrt(), layers[:, 3:4].clamp(0.0, 1.0)


def benchmark_instantrph(args: argparse.Namespace, device: torch.device) -> None:
    run_dir = Path(args.run_dir)
    config = load_json(run_dir / 'config.json')
    channels = WAVELENGTH_CHANNELS['rgb']
    tmx = args.tmx or int(config['tmx'])
    loader, _ = run_eval_loader(config, data_root=args.data_root or None, batch_size=args.batch_size)
    batches = list(islice(loader, args.max_batches))
    model = load_lae_3d(find_checkpoint(run_dir), tmx=tmx, channels=channels).to(device).eval()
    renderer = LayeredFocalStackRenderer.from_run_config(
        config, image_hw=tuple(batches[0]['rgb_vol'].shape[-2:]), focus_distances_m=run_focus_distances(config, batches[0]['focal_stack'].shape[1]),
        tmx=tmx, channels=channels, device=device, compile=args.optimize, compute_dtype=DTYPES[args.compute_dtype] if args.optimize else torch.float32,
        use_autocast=args.use_autocast,
    )
    if args.optimize:
        model = torch.compile(model)
    print(f'InstantRPH: tmx={tmx} optimize={args.optimize} compute_dtype={renderer.compute_dtype} autocast={args.use_autocast} device={device}')
    all_durations, frames = [], 0
    with torch.inference_mode():
        for index, batch in enumerate(batches):
            rgb_vol, alpha_vol = prepare_volume(batch['rgb_vol'], batch['alpha_vol'], device, channels)
            durations = time_runs(lambda: renderer.render_field(model(rgb_vol), alpha_vol), device, warmup=args.warmup, runs=args.runs)
            report(f'Sample {index} ({", ".join(batch["scene_name"])})', durations, rgb_vol.shape[0], tmx)
            all_durations += durations
            frames += rgb_vol.shape[0] * len(durations)
    print(f'Aggregated over {len(all_durations)} timed runs: {frames / sum(all_durations):.2f} holograms/s')


def benchmark_arh(args: argparse.Namespace, device: torch.device) -> None:
    if args.mpi_dir:
        rgb_vol, alpha_vol = load_mpi_layers(Path(args.mpi_dir))
        source = f'MPI {args.mpi_dir}'
    else:
        rgb_vol, alpha_vol = torch.rand(1, 3, args.depth, R, C), torch.rand(1, 1, args.depth, R, C)
        source = 'synthetic MPI'
    rgb_vol, alpha_vol = rgb_vol.to(device), alpha_vol.to(device)
    _, _, depth, height, width = rgb_vol.shape
    models = load_arh_models(args.arh_model_path, args.tmx, device)
    kernels = layer_kernels(height, width, depth, args.z_dist_mm * 1e-3, args.layer_spacing_mm * 1e-3, device)
    print(f'ARH: {source}, {depth} layers, {height}x{width}, tmx={args.tmx}, device={device}')
    durations = time_runs(lambda: arh_render_field(rgb_vol, alpha_vol, models=models, kernels=kernels), device,
                          warmup=args.warmup, runs=args.runs)
    report('ARH', durations, 1, args.tmx)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--method', required=True, choices=['instantrph', 'arh'])
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--warmup', type=int, default=3, help='Untimed runs per input (compilation, autotuning).')
    parser.add_argument('--runs', type=int, default=5, help='Timed runs per input.')
    parser.add_argument('--tmx', type=int, default=0, help='instantrph: default the trained tmx; arh: default 8.')
    ours = parser.add_argument_group('instantrph')
    ours.add_argument('--run-dir', help='Run directory (merge_rgb_3d.py output).')
    ours.add_argument('--data-root', default='', help="Default: the run's data root (its validation scenes are used).")
    ours.add_argument('--max-batches', type=int, default=1)
    ours.add_argument('--batch-size', type=int, default=1)
    ours.add_argument('--optimize', action='store_true', help='torch.compile the LAE and renderer; --compute-dtype convolutions.')
    ours.add_argument('--compute-dtype', default='float16', choices=list(DTYPES))
    ours.add_argument('--use-autocast', action='store_true', help='CUDA autocast in the layer loop (with --optimize).')
    arh = parser.add_argument_group('arh')
    arh.add_argument('--arh-model-path', default=str(ARH_MODEL_DIR), help='Folder with {red,green,blue}_{frame}.pth.')
    arh.add_argument('--mpi-dir', default='', help='RGBA layer PNGs (default: a random MPI of --depth layers at the SLM size).')
    arh.add_argument('--depth', type=int, default=8)
    arh.add_argument('--z-dist-mm', type=float, default=100.0)
    arh.add_argument('--layer-spacing-mm', type=float, default=0.25)
    args = parser.parse_args()

    device = pick_device(args.device)
    if args.method == 'instantrph':
        if not args.run_dir:
            parser.error('--method instantrph needs --run-dir')
        benchmark_instantrph(args, device)
    else:
        args.tmx = args.tmx or 8
        benchmark_arh(args, device)


if __name__ == '__main__':
    main()
