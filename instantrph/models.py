"""
Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
Conference: SIGGRAPH Asia 2026
Project Page: https://opueyociutad.github.io/InstantRPH_Web/

Linear Amplitude Encoders (LAE) and the identity encoders of the amplitude-discard baseline.

The LAE is a learned per-pixel affine map, `scale * x + bias`, with one map per color channel
and per time-multiplexed (tmx) frame. It is applied to the target amplitude before the
random phase is added.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

from instantrph.common import load_state_dict
from instantrph.constants import C, R, WAVELENGTHS_M


class LinearAmplitudeEncoder(nn.Module):
    """2D LAE: [B, C, H, W] amplitude -> [B, C * tmx, H, W] (channel-major: c * tmx + t)."""

    def __init__(self, in_channels: int = len(WAVELENGTHS_M), tmx: int = 1, image_hw: Sequence[int] = (R, C)) -> None:
        super().__init__()
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.tmx = int(tmx)
        shape = (1, in_channels * self.tmx, *self.image_hw)
        self.scale = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.bypass = False  # evaluate the replicated input without the learned map (ablation)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints saved before the LAE rename nest the maps under 'spatial_filter.'.
        for name in ('scale', 'bias'):
            legacy_key = f'{prefix}spatial_filter.{name}'
            if legacy_key in state_dict and prefix + name not in state_dict:
                state_dict[prefix + name] = state_dict.pop(legacy_key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-2:] != self.image_hw:
            raise ValueError(f'Expected input shape (*, C, {self.image_hw[0]}, {self.image_hw[1]}), got {tuple(image.shape)}')
        x = torch.repeat_interleave(image, repeats=self.tmx, dim=1)
        return x if self.bypass else x * self.scale + self.bias


class IdentityEncoder(nn.Module):
    """2D amplitude-discard baseline: the input amplitude replicated over the tmx frames."""

    def __init__(self, tmx: int) -> None:
        super().__init__()
        self.tmx = int(tmx)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.repeat_interleave(image, repeats=self.tmx, dim=1)


class LinearAmplitudeEncoder3D(nn.Module):
    """3D LAE: [B, >=C, D, H, W] MPI amplitude -> [B, tmx, C, D, H, W]. Extra input channels (alpha) are ignored."""

    def __init__(self, *, tmx: int, depth: int, height: int, width: int, num_rgb_channels: int = 3) -> None:
        super().__init__()
        self.tmx = int(tmx)
        self.num_rgb_channels = int(num_rgb_channels)
        self.volume_shape = (int(depth), int(height), int(width))
        shape = (1, self.tmx, self.num_rgb_channels, *self.volume_shape)
        self.scale = nn.Parameter(torch.ones(shape))  # identity at initialization
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        if volume.ndim != 5 or tuple(volume.shape[2:]) != self.volume_shape or volume.shape[1] < self.num_rgb_channels:
            raise ValueError(
                f'Expected a [B, >={self.num_rgb_channels}, {", ".join(map(str, self.volume_shape))}] volume, '
                f'got {tuple(volume.shape)}'
            )
        rgb = volume[:, :self.num_rgb_channels].unsqueeze(1)
        batch_size = volume.shape[0]
        return rgb * self.scale.expand(batch_size, -1, -1, -1, -1, -1) + self.bias.expand(batch_size, -1, -1, -1, -1, -1)


class IdentityEncoder3D(nn.Module):
    """3D amplitude-discard baseline: the MPI amplitude replicated over the tmx frames."""

    def __init__(self, *, tmx: int, num_rgb_channels: int) -> None:
        super().__init__()
        self.tmx = int(tmx)
        self.num_rgb_channels = int(num_rgb_channels)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return volume[:, :self.num_rgb_channels].unsqueeze(1).expand(-1, self.tmx, -1, -1, -1, -1)


def load_lae_2d(checkpoint_path: str | Path, device: torch.device) -> tuple[LinearAmplitudeEncoder, dict[str, Any]]:
    """Rebuild a trained 2D LAE from the run config embedded in its checkpoint. Returns (model, config)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
    model = LinearAmplitudeEncoder(
        tmx=int(config.get('tmx', 1)),
        image_hw=(int(config.get('image_rows', R)), int(config.get('image_cols', C))),
    )
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=True)
    return model.to(device), config


def load_lae_3d(path: str | Path, *, tmx: int | None = None, channels: Sequence[int] | None = None) -> LinearAmplitudeEncoder3D:
    """Load a 3D LAE from a checkpoint or run directory, keeping only its first `tmx` frames
    and the RGB `channels` (a checkpoint that already has len(channels) channels is used as is)."""
    state = load_state_dict(path)
    scale, bias = state['scale'], state['bias']
    if tmx is not None:
        if tmx > scale.shape[1]:
            raise ValueError(f'Requested tmx={tmx}, but {path} was trained with tmx={scale.shape[1]}')
        scale, bias = scale[:, :tmx], bias[:, :tmx]
    if channels is not None and scale.shape[2] != len(channels):
        scale, bias = scale[:, :, list(channels)], bias[:, :, list(channels)]
    _, frames, num_channels, depth, height, width = scale.shape
    model = LinearAmplitudeEncoder3D(tmx=frames, depth=depth, height=height, width=width, num_rgb_channels=num_channels)
    model.load_state_dict({'scale': scale.contiguous(), 'bias': bias.contiguous()})
    return model
