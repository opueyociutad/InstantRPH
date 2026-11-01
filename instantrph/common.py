"""
Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
Conference: SIGGRAPH Asia 2026
Project Page: https://opueyociutad.github.io/InstantRPH_Web/

Shared helpers: seeding, devices, run files, phase tensors and RGB channel selection.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

# Inputs live under data/ (see data/README.md), everything the scripts write under output/ (see output/README.md).
DATA_DIR = REPO_ROOT / 'data'
OUTPUT_DIR = REPO_ROOT / 'output'
DATASET_2D_DIR = DATA_DIR / 'dataset_2d'
DATASET_3D_DIR = DATA_DIR / 'dataset_3d'
DATASET_VIDEO_DIR = DATA_DIR / 'dataset_video'
PHASE_FILE_2D = DATA_DIR / 'models_2d' / 'phase' / 'l1_tmx24.pt'
PHASE_FILE_3D = DATA_DIR / 'models_3d' / 'phase' / 'l21_tmx24.pt'
KERNELS_DIR = DATA_DIR / 'models_3d' / 'kernels'
ARH_MODEL_DIR = DATA_DIR / 'model_ARH'

WAVELENGTH_CHANNELS = {'rgb': [0, 1, 2], 'r': [0], 'g': [1], 'b': [2]}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str = 'auto') -> torch.device:
    """Resolve a --device value ('auto' | 'cpu' | 'cuda'). Select the GPU with CUDA_VISIBLE_DEVICES."""
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    return torch.device(name)


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'File not found: {path}')
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Phase tensors
# ---------------------------------------------------------------------------


def load_phase(
    path: str | Path,
    *,
    tmx: int,
    channels: Sequence[int] = (0, 1, 2),
    image_hw: Sequence[int] | None = None,
    layers: slice = slice(None),
) -> torch.Tensor:
    """Read the [layers, :tmx, channels, :H, :W] block of a [LAYERS, TMX, RGB, H, W] phase file.

    The file is memory-mapped, so only the requested block is read. Phases are i.i.d., so a
    smaller image simply uses the top-left crop.
    """
    phase = torch.load(path, map_location='cpu', mmap=True)
    if phase.ndim != 5:
        raise ValueError(f'Expected a [LAYERS, TMX, RGB, H, W] phase tensor in {path}, got {tuple(phase.shape)}')
    if (layers.start or 0) >= phase.shape[0]:
        raise ValueError(f'Requested phase layer {layers.start}, but {path} only holds {phase.shape[0]} layer(s)')
    if tmx > phase.shape[1]:
        raise ValueError(f'Requested tmx={tmx}, but {path} only holds {phase.shape[1]} frames')
    rows, cols = image_hw if image_hw is not None else phase.shape[-2:]
    if rows > phase.shape[-2] or cols > phase.shape[-1]:
        raise ValueError(
            f'Requested {rows}x{cols} phases, but {path} is {phase.shape[-2]}x{phase.shape[-1]}. '
            'Generate a larger one with generate_phase_pattern.py --rows/--cols.'
        )
    return phase[layers, :tmx, list(channels), :rows, :cols].to(torch.float32).contiguous()


def run_phase_file(config: dict[str, Any], default: Path) -> Path:
    """Phase file used by a run, or `default` (PHASE_FILE_2D / PHASE_FILE_3D) when the run's file is
    not on this machine, e.g. for a pretrained LAE. Older runs stored a directory holding '*_tmx24.pt'."""
    if config.get('phase_file'):
        path = Path(config['phase_file'])
        if path.exists():
            return path
        print(f'[Phase] {path} not found, using {default}')
    elif config.get('phase_dir'):
        matches = sorted(Path(config['phase_dir']).glob('*_tmx24.pt'))
        if matches:
            return matches[-1]
    return default


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def find_checkpoint(path: str | Path) -> Path:
    """A checkpoint file, or the best one in a run directory.

    Run directories hold best.pt / last.pt. Runs trained with an older version of
    train_3d.py --use-accelerate hold accelerate state directories best/ / last/ instead.
    """
    path = Path(path)
    if path.is_file() or any(path.glob('*.safetensors')):
        return path
    for name in ('best.pt', 'last.pt', 'best', 'last'):
        if (path / name).exists():
            return path / name
    raise FileNotFoundError(f'No best.pt, last.pt, best/ or last/ in {path}')


def load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Model weights from a checkpoint file, run directory or accelerate state directory."""
    path = find_checkpoint(path)
    if path.is_dir() or path.suffix == '.safetensors':
        from safetensors.torch import load_file

        files = sorted(path.glob('*.safetensors')) if path.is_dir() else [path]
        if not files:
            raise FileNotFoundError(f'No .safetensors file in {path}')
        state: dict[str, torch.Tensor] = {}
        for file in files:
            state.update(load_file(str(file), device='cpu'))
        return state
    obj = torch.load(path, map_location='cpu')
    if not isinstance(obj, dict):
        return obj
    if obj.get('phase_tensor_state_dict') is not None:
        raise ValueError(f'{path} was trained with an optimized phase tensor, which is no longer supported')
    return obj.get('model_state_dict', obj)


# ---------------------------------------------------------------------------
# MPI volumes and focal stacks
# ---------------------------------------------------------------------------


def focus_distances_m(
    stack_count: int,
    *,
    start_mm: float,
    end_mm: float,
    center_mm: float = 100.0,
    distances_mm: str = '',
) -> list[float]:
    """Focus distances of a focal stack: an explicit comma-separated list, or a linear sweep."""
    if distances_mm:
        values = [float(token) for token in distances_mm.split(',') if token.strip()]
        if len(values) != stack_count:
            raise ValueError(f'{len(values)} focus distances given for a {stack_count}-plane focal stack')
        return [value / 1000.0 for value in values]
    if stack_count == 1:
        return [center_mm / 1000.0]
    return torch.linspace(start_mm / 1000.0, end_mm / 1000.0, steps=stack_count).tolist()


def run_focus_distances(config: dict[str, Any], stack_count: int) -> list[float]:
    """Focus distances (m) of a train_3d.py run; older runs only stored the sweep range."""
    if config.get('focus_distances_m'):
        return [float(d) for d in config['focus_distances_m']]
    return focus_distances_m(
        stack_count,
        start_mm=float(config.get('focus_start_mm', 97.5)),
        end_mm=float(config.get('focus_end_mm', 101.0)),
        center_mm=float(config.get('focus_center_mm', 100.0)),
    )


def prepare_volume(
    rgb_vol: torch.Tensor, alpha_vol: torch.Tensor, device: torch.device, channels: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select RGB channels of a [B,3,D,H,W] MPI volume and move it and its [B,1,D,H,W] alpha to device."""
    rgb_vol = rgb_vol[:, list(channels)]
    return rgb_vol.to(device=device, dtype=torch.float32), alpha_vol.to(device=device, dtype=torch.float32)


def select_stack_channels(stack: torch.Tensor, channels: Sequence[int]) -> torch.Tensor:
    """Select channels of a [B,S,C,H,W] focal stack (a 1-channel stack is broadcast to RGB first)."""
    if stack.shape[2] == 1 and max(channels) >= 1:
        stack = stack.repeat(1, 1, 3, 1, 1)
    return stack[:, :, list(channels)]


def to_rgb_stack(stack: torch.Tensor) -> torch.Tensor:
    """Broadcast a 1-channel [B,S,C,H,W] stack to 3 channels for display."""
    return stack.repeat(1, 1, 3, 1, 1) if stack.shape[2] == 1 else stack[:, :, :3]
