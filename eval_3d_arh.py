#!/usr/bin/env python3
"""
Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
Conference: SIGGRAPH Asia 2026
Project Page: https://opueyociutad.github.io/InstantRPH_Web/

Evaluate the ARH baseline (instantrph/arh.py: Artifact-Resilient Real-Time Holography) with the
same scenes, focus planes and color-balanced metric as eval_3d.py, for apples-to-apples comparisons with a 3D LAE.

The scenes (--eval-split, --batch-size, --max-batches), the focus planes, the SLM distance (the farthest focus
plane) and the MPI layer spacing all come from --run-dir, exactly as in eval_3d.py, so the same flags evaluate the
same scenes. The released ARH weights ({red,green,blue}_{frame}.pth in --arh-model-path, see
data/model_ARH/README.md) hold 8 frames for a 100 mm SLM distance; compare them with the LAE at 8 frames:

    python eval_3d_arh.py --run-dir data/models_3d/lae/lae_3d_tmx8_rgb
    python eval_3d.py --run-dir data/models_3d/lae/lae_3d_tmx8_rgb            # AD baseline and LAE, same scenes
    python eval_3d.py --run-dir data/models_3d/lae/lae_3d_tmx24_rgb --tmx 8   # a tmx-24 LAE on its first 8 frames

Writes to --output-dir (default: output/arh/comparisons/<run folder>):
  validation_phase_patterns/arh/           with --save-phase-patterns: SLM phase (0..2pi) per scene
--export-dir DIR saves, per scene, the ARH focal stacks (color-balanced, as in the PSNR), the target focal stacks
and the SLM phases, in the layout of eval_3d.py (DIR/arh/<scene>/).
"""
from __future__ import annotations

import argparse
import time
from itertools import islice
from pathlib import Path

import torch

from instantrph.arh import COLOR_NAMES, arh_render_field, layer_kernels, load_arh_models
from instantrph.common import (
    ARH_MODEL_DIR, OUTPUT_DIR, WAVELENGTH_CHANNELS, load_json, pick_device, prepare_volume, run_focus_distances, select_stack_channels,
)
from instantrph.data_3d import run_eval_loader
from instantrph.renderer_3d import LayeredFocalStackRenderer
from instantrph.viz import export_scenes, scene_file_name

ARH_SLM_DISTANCE_MM = 100.0  # SLM distance of the released ARH weights


@torch.no_grad()
def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config = load_json(run_dir / 'config.json')
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR / 'arh' / 'comparisons' / run_dir.resolve().name
    device = pick_device(args.device)
    channels = WAVELENGTH_CHANNELS['rgb']
    missing = [f'{color}_{frame}.pth' for frame in range(1, args.tmx + 1) for color in COLOR_NAMES
               if not (Path(args.arh_model_path) / f'{color}_{frame}.pth').exists()]
    if missing:
        raise FileNotFoundError(f'ARH weights missing in {args.arh_model_path}: {", ".join(missing)} (see data/model_ARH/README.md)')

    loader, split = run_eval_loader(
        config, data_root=args.data_root or None, split=args.eval_split, batch_size=args.batch_size or int(config.get('batch_size', 1)),
        num_workers=args.num_workers if args.num_workers >= 0 else int(config.get('num_workers', 2)),
    )
    batches = list(islice(loader, args.max_batches)) if args.max_batches > 0 else list(loader)
    first = batches[0]
    depth, height, width = first['rgb_vol'].shape[2:]
    renderer = LayeredFocalStackRenderer.from_run_config(
        config, image_hw=(height, width), focus_distances_m=run_focus_distances(config, first['focal_stack'].shape[1]),
        tmx=args.tmx, channels=channels, device=device, compile=False,
    )
    slm_distance_m = renderer.focus_distances_m[0]  # as in the LAE renderer: MPI layer d sits d * spacing closer to the SLM
    models = load_arh_models(args.arh_model_path, args.tmx, device)
    kernels = layer_kernels(height, width, depth, slm_distance_m, renderer.layer_spacing_m, device)

    def arh_field(rgb_vol: torch.Tensor, alpha_vol: torch.Tensor) -> torch.Tensor:
        """Phase-only SLM field [B, tmx, 3, H, W]; ARH encodes one MPI at a time."""
        return torch.cat([arh_render_field(rgb_vol[i:i + 1], alpha_vol[i:i + 1], models=models, kernels=kernels)
                          for i in range(rgb_vol.shape[0])])

    scenes = [name for batch in batches for name in batch['scene_name']]
    print(f'Run: {run_dir} | ARH weights: {args.arh_model_path} | device: {device}')
    print(f'Eval split: {split} | scenes ({len(scenes)}): {", ".join(dict.fromkeys(scenes))}')
    print(f'tmx: {args.tmx} | MPI layers: {depth} ({renderer.layer_spacing_m * 1000.0:.3f} mm apart) | SLM distance: '
          f'{slm_distance_m * 1000.0:.3f} mm | focus planes (mm): {[round(d * 1000, 3) for d in renderer.focus_distances_m]}')
    if abs(slm_distance_m * 1000.0 - ARH_SLM_DISTANCE_MM) > 1e-3:
        print(f'[Warning] The ARH weights are for a {ARH_SLM_DISTANCE_MM:.0f} mm SLM distance, this run uses {slm_distance_m * 1000.0:.3f} mm')

    phase_dir = output_dir / 'validation_phase_patterns' / 'arh'
    if args.save_phase_patterns:
        phase_dir.mkdir(parents=True, exist_ok=True)
    total_loss = total_psnr = 0.0
    count = 0
    start = time.perf_counter()
    for batch in batches:
        rgb_vol, alpha_vol = prepare_volume(batch['rgb_vol'], batch['alpha_vol'], device, channels)
        target_stack = select_stack_channels(batch['focal_stack'].to(device=device, dtype=torch.float32), channels)
        field = arh_field(rgb_vol, alpha_vol)
        loss, psnr_value = renderer.field_stack_loss(field, target_stack)
        total_loss += loss.item() * rgb_vol.shape[0]
        total_psnr += psnr_value.item() * rgb_vol.shape[0]
        count += rgb_vol.shape[0]
        if args.save_phase_patterns:
            phase = torch.angle(field).cpu() + torch.pi
            for i, scene in enumerate(batch['scene_name']):
                torch.save(phase[i], phase_dir / f'{scene_file_name(scene)}.pt')
    if args.save_phase_patterns:
        print(f'Saved phase patterns to: {phase_dir}')

    if args.export_dir:
        export_scenes({'arh': arh_field}, renderer, batches, channels, device, Path(args.export_dir))
    print(f'ARH: loss={total_loss / count:.6f} PSNR={total_psnr / count:.3f} dB ({time.perf_counter() - start:.2f}s)')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', required=True, help='3D LAE run (config.json) whose scenes, focus planes and optics to use.')
    parser.add_argument('--data-root', default='', help="Default: the run's data root.")
    parser.add_argument('--eval-split', default='auto', choices=['auto', 'val', 'train', 'all'],
                        help="The run's validation or training scenes, or every scene ('auto': val if the run has any).")
    parser.add_argument('--batch-size', type=int, default=0, help="Default: the run's batch size.")
    parser.add_argument('--num-workers', type=int, default=-1, help="Default: the run's value.")
    parser.add_argument('--max-batches', type=int, default=5, help='Evaluate the first N batches (0: all).')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--tmx', type=int, default=8, help='ARH frames: uses {color}_1.pth ... {color}_<tmx>.pth.')
    parser.add_argument('--arh-model-path', default=str(ARH_MODEL_DIR), help='Folder with {red,green,blue}_{frame}.pth.')
    parser.add_argument('--output-dir', default='', help='Default: output/arh/comparisons/<run folder>.')
    parser.add_argument('--save-phase-patterns', action='store_true')
    parser.add_argument('--export-dir', default='', help='Export per-scene focal stacks and phase patterns here.')
    return parser.parse_args()


if __name__ == '__main__':
    main()
