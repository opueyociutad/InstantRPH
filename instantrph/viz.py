"""
Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
Conference: SIGGRAPH Asia 2026
Project Page: https://opueyociutad.github.io/InstantRPH_Web/

Image saving, labeled preview mosaics, MPI compositing and the per-scene 3D export.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from instantrph.common import prepare_volume, select_stack_channels, to_rgb_stack
from instantrph.metrics import colorbalance


def to_uint8_hwc(chw: torch.Tensor) -> np.ndarray:
    """[C,H,W] float in ~[0,1] -> uint8 [H,W] (1 channel), [H,W,3|4] (3 or 4 channels, first 3 if more), else grayscale mean."""
    chw = chw.detach().cpu().float().clamp(0.0, 1.0)
    channels = chw.shape[0]
    if channels == 1:
        hwc = chw[0]
    elif channels in (3, 4):
        hwc = chw.permute(1, 2, 0)
    elif channels > 4:
        hwc = chw[:3].permute(1, 2, 0)
    else:
        hwc = chw.mean(dim=0)
    return (hwc.numpy() * 255.0).round().astype(np.uint8)


def to_pil(chw: torch.Tensor) -> Image.Image:
    arr = to_uint8_hwc(chw)
    mode = 'L' if arr.ndim == 2 else ('RGBA' if arr.shape[-1] == 4 else 'RGB')
    return Image.fromarray(arr, mode=mode)


def save_png(chw: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_pil(chw).save(path)


def overlay_text(image_chw: torch.Tensor, text: str, x: int = 6, y: int = 6) -> torch.Tensor:
    """Draw white text on a black box. Images with other than 1 or 3 channels are averaged to grayscale."""
    if image_chw.shape[0] not in (1, 3):
        image_chw = image_chw.mean(dim=0, keepdim=True)
    gray = image_chw.shape[0] == 1
    image_np = (image_chw.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    pil = Image.fromarray(image_np[0] if gray else np.transpose(image_np, (1, 2, 0)), mode='L' if gray else 'RGB')
    draw = ImageDraw.Draw(pil)
    draw.rectangle([x - 2, y - 2, x + max(6 * len(text), 24) + 2, y + 12 + 2], fill=0 if gray else (0, 0, 0))
    draw.text((x, y), text, fill=255 if gray else (255, 255, 255))
    out = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0)
    return out.unsqueeze(0) if gray else out.permute(2, 0, 1)


def _resize(images: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(images, size=size, mode='bilinear', align_corners=False)


def hologram_mosaic(
    inputs: torch.Tensor,
    preds_baseline: torch.Tensor,
    preds: torch.Tensor,
    psnr_baseline: torch.Tensor,
    psnr_model: torch.Tensor,
    *,
    distance_mm: float,
    max_images: int,
    size: tuple[int, int],
) -> torch.Tensor:
    """2D TensorBoard grid per image: [input | baseline] over [target | ours], with PSNRs."""
    n = min(max_images, inputs.shape[0])
    inputs_v, baseline_v, preds_v = (_resize(x[:n], size) for x in (inputs, preds_baseline, preds))
    tiles = []
    for i in range(n):
        top = torch.cat([
            overlay_text(inputs_v[i], 'INPUT'),
            overlay_text(baseline_v[i], f'DUMMY {psnr_baseline[i].item():.2f} dB {distance_mm:.1f}mm'),
        ], dim=-1)
        bottom = torch.cat([
            overlay_text(inputs_v[i], 'TARGET'),
            overlay_text(preds_v[i], f'OURS {psnr_model[i].item():.2f} dB {distance_mm:.1f}mm'),
        ], dim=-1)
        tiles.append(torch.cat([top, bottom], dim=-2))
    return torch.cat(tiles, dim=-2).clamp(0.0, 1.0)


def lae_output_mosaic(amp_inputs: torch.Tensor, raw_outputs: torch.Tensor, *, max_images: int, size: tuple[int, int]) -> torch.Tensor:
    """2D TensorBoard grid per image: [LAE output averaged over tmx | input amplitude]."""
    n = min(max_images, amp_inputs.shape[0])
    channels = amp_inputs.shape[1]
    raw = raw_outputs[:n]
    raw = raw.view(n, channels, raw.shape[1] // channels, *raw.shape[-2:]).mean(dim=2)
    amp_v, raw_v = _resize(amp_inputs[:n], size), _resize(raw, size)
    rows = [
        torch.cat([overlay_text(raw_v[i], 'RAW model(amp) avg_tmx'), overlay_text(amp_v[i], 'AMP sqrt(input RGB)')], dim=-1)
        for i in range(n)
    ]
    return torch.cat(rows, dim=-2).clamp(0.0, 1.0)


def focal_stack_mosaic(
    pred_stack: torch.Tensor,
    target_stack: torch.Tensor,
    focus_distances_m: list[float],
    *,
    max_images: int,
    size: tuple[int, int],
    max_planes: int = 3,
) -> torch.Tensor:
    """3D TensorBoard grid per scene: a row of target planes over a row of predicted planes."""
    n = min(max_images, pred_stack.shape[0])
    planes = min(max_planes, pred_stack.shape[1])
    rows = []
    for b in range(n):
        targets, preds = [], []
        for s in range(planes):
            label = f' {focus_distances_m[s] * 1000.0:.1f}mm'
            targets.append(overlay_text(_resize(target_stack[b, s].unsqueeze(0), size).squeeze(0), f'T{s}{label}'))
            preds.append(overlay_text(_resize(pred_stack[b, s].unsqueeze(0), size).squeeze(0), f'P{s}{label}'))
        rows.append(torch.cat([torch.cat(targets, dim=-1), torch.cat(preds, dim=-1)], dim=-2))
    return torch.cat(rows, dim=-2).clamp(0.0, 1.0)


def mpi_data_preview(rgb_vol: torch.Tensor, focal_stack: torch.Tensor, count: int = 4) -> torch.Tensor:
    """One training sample: `count` evenly spaced MPI layers over `count` focal-stack planes ([C,D,H,W], [S,C,H,W])."""
    def row(images: list[torch.Tensor]) -> torch.Tensor:
        picks = images if len(images) <= count else [images[int(i * len(images) / count)] for i in range(count)]
        return torch.cat([x.repeat(3, 1, 1) if x.shape[0] == 1 else x[:3] for x in picks], dim=-1).clamp(0.0, 1.0)

    layers, planes = row(list(rgb_vol.unbind(1))), row(list(focal_stack.unbind(0)))
    if layers.shape[-1] != planes.shape[-1]:
        planes = _resize(planes.unsqueeze(0), (planes.shape[-2], layers.shape[-1])).squeeze(0)
    return torch.cat([layers, planes], dim=-2)


def composite_mpi(rgb_vol: torch.Tensor, alpha_vol: torch.Tensor, *, front_to_back: bool) -> torch.Tensor:
    """Alpha-composite an MPI ([B,C,L,H,W] color, [B,1,L,H,W] alpha) to [B,C,H,W], back to front.

    front_to_back=True: layer 0 is nearest the camera; False: layer 0 is farthest.
    """
    composited = torch.zeros_like(rgb_vol[:, :, 0])
    num_layers = rgb_vol.shape[2]
    for layer in (range(num_layers - 1, -1, -1) if front_to_back else range(num_layers)):
        alpha = alpha_vol[:, :, layer]
        composited = rgb_vol[:, :, layer] * alpha + composited * (1.0 - alpha)
    return composited


def scene_file_name(name: str) -> str:
    """A scene name made safe to use as a file name."""
    return re.sub(r'[^A-Za-z0-9._-]+', '_', name.strip()) or 'unnamed_scene'


@torch.no_grad()
def export_scenes(
    methods: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    renderer,
    loader,
    channels: Sequence[int],
    device: torch.device,
    export_dir: Path,
) -> None:
    """Per method and scene, under export_dir/<method>/<scene>/: color-balanced predicted focal stacks (each plane
    and channel scaled to the target's mean, as in the PSNR), target focal stacks, and SLM phases.

    methods maps each method name to a function (rgb_vol, alpha_vol) -> phase-only SLM field [B, tmx, C, H, W];
    renderer (a LayeredFocalStackRenderer) propagates the fields to its focus planes.
    """
    for batch in loader:
        rgb_vol, alpha_vol = prepare_volume(batch['rgb_vol'], batch['alpha_vol'], device, channels)
        target_stack = to_rgb_stack(batch['focal_stack'].to(dtype=torch.float32))
        balance_target = select_stack_channels(batch['focal_stack'].to(device=device, dtype=torch.float32), channels)
        for method, render_field in methods.items():
            field = render_field(rgb_vol, alpha_vol)
            pred_stack = to_rgb_stack(colorbalance(torch.stack(list(renderer.focal_planes(field)), dim=1), balance_target)).cpu()
            phase = torch.angle(field).cpu()
            for i, scene in enumerate(batch['scene_name']):
                out = export_dir / method / scene
                out.mkdir(parents=True, exist_ok=True)
                torch.save(phase[i:i + 1], out / f'phase_pattern_{scene}.pt')
                for t in range(phase.shape[1]):
                    save_png(((phase[i, t] + np.pi) / (2.0 * np.pi)).clamp(0.0, 1.0), out / f'phase_pattern_{scene}_b00_slot{t:02d}.png')
                for name, stack in (('pred', pred_stack[i:i + 1]), ('target', target_stack[i:i + 1])):
                    torch.save(stack, out / f'{name}_focal_stack.pt')
                    np.save(out / f'{name}_focal_stack.npy', stack.squeeze(0).numpy())
                    for f in range(stack.shape[1]):
                        save_png(stack[0, f], out / f'{name}_b00_f{f:02d}.png')
                (out / 'scene_info.json').write_text(json.dumps({
                    'scene_name': scene,
                    'focus_distances_mm': [round(d * 1000.0, 4) for d in renderer.focus_distances_m],
                    'pred_focal_stack_shape': list(pred_stack[i].shape),
                }, indent=2))
    print(f'Exported focal stacks and phase patterns to: {export_dir}')
