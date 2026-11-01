#!/usr/bin/env python3
"""
Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
Conference: SIGGRAPH Asia 2026
Project Page: https://opueyociutad.github.io/InstantRPH_Web/

Render an MPI video with a trained 3D LAE at the focus distances of a trajectory (data/video_config/*.yaml).

Input: --data-video-dir (default: data/dataset_video) with one clip_XXXX/ folder per MPI layer, each holding the RGBA frames
(frame_0000.png, ...). Layers are merged (alpha compositing) or repeated to the model's layer count.
Output: <output-dir>/frame_XXXXX/dist_<mm>mm.png for every frame and every trajectory distance,
color-balanced to the composited input frame.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from instantrph.common import DATASET_VIDEO_DIR, OUTPUT_DIR, find_checkpoint, load_json, pick_device
from instantrph.metrics import colorbalance
from instantrph.models import load_lae_3d
from instantrph.renderer_3d import LayeredFocalStackRenderer
from instantrph.viz import composite_mpi, save_png

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg'}


@dataclass
class DistanceTrajectory:
    """Constant focus distance with optional back-and-forth focal sweeps (see data/video_config/*.yaml)."""
    base_distance_mm: float = 100.0
    sweeps: list[dict] = field(default_factory=list)
    description: str = ''

    @classmethod
    def from_yaml(cls, path: Path) -> DistanceTrajectory:
        config = yaml.safe_load(path.read_text())
        return cls(config.get('base_distance_mm', 100.0), config.get('sweeps', []), config.get('description', ''))

    def distance_mm(self, frame_idx: int) -> float:
        for sweep in self.sweeps:
            if frame_idx != sweep.get('frame_idx'):
                continue
            duration = sweep.get('duration_frames', 30)
            start = sweep.get('distance_start_mm', self.base_distance_mm - 5)
            end = sweep.get('distance_end_mm', self.base_distance_mm + 5)
            offset = 0
            for s in self.sweeps:
                if s.get('frame_idx', 0) <= frame_idx:
                    offset = frame_idx - s.get('frame_idx', 0)
            position = (offset % (duration * 2)) / (duration * 2)
            t = position * 2.0 if position < 0.5 else 2.0 - position * 2.0
            return start + t * (end - start)
        return self.base_distance_mm


def _clip_dirs(data_video_dir: Path) -> list[Path]:
    clips = sorted(d for d in data_video_dir.iterdir() if d.is_dir() and d.name.startswith('clip_'))
    if not clips:
        raise ValueError(f'No clip_* subdirectories found in {data_video_dir}')
    return clips


def _frames(clip: Path) -> list[Path]:
    return sorted(f for f in clip.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)


def _resample_layers(layers: list[np.ndarray], target_layers: int) -> tuple[torch.Tensor, torch.Tensor]:
    """[H,W,4] layers (index 0 farthest) -> rgb [L,3,H,W] and alpha [L,1,H,W] with L = target_layers.
    More layers are merged by alpha compositing contiguous groups; fewer are repeated (nearest)."""
    if len(layers) > target_layers:
        bounds = np.linspace(0, len(layers), target_layers + 1, dtype=int)
        merged = []
        for start, end in zip(bounds[:-1], bounds[1:]):
            end = end if end > start else min(start + 1, len(layers))
            rgba = np.zeros_like(layers[0])
            for layer in layers[start:end]:
                alpha = layer[:, :, 3:4]
                rgba[:, :, :3] = rgba[:, :, :3] * (1 - alpha) + layer[:, :, :3] * alpha
                rgba[:, :, 3:4] += alpha * (1 - rgba[:, :, 3:4])
            merged.append(rgba)
        layers = merged
    elif len(layers) < target_layers:
        layers = [layers[min(int(i * len(layers) / target_layers), len(layers) - 1)] for i in range(target_layers)]
    stacked = torch.from_numpy(np.stack(layers)).permute(0, 3, 1, 2)
    return stacked[:, :3], stacked[:, 3:4]


def load_frames(clips: list[Path], frame_indices: list[int], target_layers: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MPI frames as rgb [B,3,L,H,W] and alpha [B,1,L,H,W], plus the composited input frames [B,3,H,W]."""
    rgbs, alphas, references = [], [], []
    for frame_idx in frame_indices:
        layers = [np.array(Image.open(frames[frame_idx]).convert('RGBA'), dtype=np.float32) / 255.0
                  for frames in map(_frames, clips) if frame_idx < len(frames)]
        stacked = torch.from_numpy(np.stack(layers)).permute(3, 0, 1, 2).unsqueeze(0)  # [1, 4, L, H, W]
        references.append(composite_mpi(stacked[:, :3], stacked[:, 3:4], front_to_back=False)[0])
        rgb, alpha = _resample_layers(layers, target_layers)
        rgbs.append(rgb)
        alphas.append(alpha)
    return torch.stack(rgbs).permute(0, 2, 1, 3, 4), torch.stack(alphas).permute(0, 2, 1, 3, 4), torch.stack(references)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-video-dir', type=Path, default=DATASET_VIDEO_DIR, help='Folder with clip_XXXX/ layer folders.')
    parser.add_argument('--model-run', type=Path, required=True, help='3D run directory (merge_rgb_3d.py output).')
    parser.add_argument('--trajectory-yaml', type=Path, default=None, help='Focus trajectory (default: constant 100 mm).')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR / 'video')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-frames', type=int, default=None, help='Only the first N frames.')
    parser.add_argument('--device', default='cuda', choices=['auto', 'cpu', 'cuda'], help='cuda falls back to cpu if unavailable.')
    args = parser.parse_args()

    device = pick_device('auto' if args.device == 'cuda' else args.device)
    clips = _clip_dirs(args.data_video_dir)
    num_frames = len(_frames(clips[0])) if args.num_frames is None else min(args.num_frames, len(_frames(clips[0])))
    height, width = np.asarray(Image.open(_frames(clips[0])[0])).shape[:2]
    config = load_json(args.model_run / 'config.json')
    model = load_lae_3d(find_checkpoint(args.model_run)).to(device).eval()
    num_layers = model.volume_shape[0]
    trajectory = DistanceTrajectory.from_yaml(args.trajectory_yaml) if args.trajectory_yaml else DistanceTrajectory()
    distances_mm = sorted({trajectory.distance_mm(i) for i in range(num_frames)})
    renderer = LayeredFocalStackRenderer.from_run_config(
        config, image_hw=(height, width), focus_distances_m=[d / 1000.0 for d in distances_mm], tmx=model.tmx,
        channels=(0, 1, 2), device=device,
    )
    print(f'{len(clips)} input layers -> {num_layers} model layers, {num_frames} frames of {height}x{width}, device {device}')
    print(f"Trajectory '{trajectory.description}': {len(distances_mm)} distance(s), {min(distances_mm):.3f}-{max(distances_mm):.3f} mm")

    for start in tqdm(range(0, num_frames, args.batch_size), unit='batch'):
        frame_indices = list(range(start, min(start + args.batch_size, num_frames)))
        rgb, alpha, references = (x.to(device).float() for x in load_frames(clips, frame_indices, num_layers))
        encoded = model(rgb)
        for distance_mm in distances_mm:
            plane = renderer.focus_distances_m.index(distance_mm / 1000.0)
            for i, frame_idx in enumerate(frame_indices):
                stack = renderer.render_stack(encoded[i:i + 1], alpha[i:i + 1])
                image = colorbalance(stack[0, plane], references[i]).clamp(0, 1)
                save_png(image, args.output_dir / f'frame_{frame_idx:05d}' / f'dist_{distance_mm:.3f}mm.png')
    print(f'Saved {num_frames} x {len(distances_mm)} images to {args.output_dir}')


if __name__ == '__main__':
    main()
