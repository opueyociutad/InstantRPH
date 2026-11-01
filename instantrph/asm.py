"""
Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
Conference: SIGGRAPH Asia 2026
Project Page: https://opueyociutad.github.io/InstantRPH_Web/

Adapted from Neural Holography (propagation_ASM.py):
    https://github.com/computational-imaging/neural-holography
    Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop Training.
    ACM Transactions on Graphics (SIGGRAPH Asia), 2020.
    The original code is released under CC BY-NC 4.0 (non-commercial use only).

Band-limited angular spectrum method (ASM) propagation.

Band limiting follows Matsushima and Shimobaba (2009). Fields are zero-padded to twice their
size for a linear convolution.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from instantrph.constants import PITCH_M

_SPATIAL_DIMS = (-2, -1)


def _pad_widths(n: int) -> tuple[int, int]:
    """Front/back zero padding that doubles a dimension of size n."""
    odd = n % 2
    return (n + odd) // 2, (n + 1 - odd) // 2


def asm_kernel(
    image_hw: Sequence[int],
    wavelengths: torch.Tensor | Sequence[float],
    distance_m: float,
    *,
    device: torch.device | str = 'cpu',
    pitch_m: float = PITCH_M,
) -> torch.Tensor:
    """Transfer function for propagating an [..., C, H, W] field by `distance_m`.

    Returns a complex64 tensor [1, C, 2H, 2W] (one kernel per wavelength), already
    ifftshifted so it can multiply the field spectrum directly.
    """
    wavelengths = torch.as_tensor(wavelengths, dtype=torch.float32).cpu().numpy()
    rows, cols = 2 * int(image_hw[0]), 2 * int(image_hw[1])
    height, width = pitch_m * float(rows), pitch_m * float(cols)

    fy = np.linspace(-1 / (2 * pitch_m) + 0.5 / (2 * height), 1 / (2 * pitch_m) - 0.5 / (2 * height), rows)
    fx = np.linspace(-1 / (2 * pitch_m) + 0.5 / (2 * width), 1 / (2 * pitch_m) - 0.5 / (2 * width), cols)
    FX, FY = np.meshgrid(fx, fy)

    phase = torch.cat([
        torch.tensor(2 * math.pi * np.sqrt(1 / wvl**2 - (FX**2 + FY**2)), dtype=torch.float32).to(device).reshape(1, 1, rows, cols)
        for wvl in wavelengths
    ], dim=1)
    phase = torch.mul(phase, distance_m)

    fy_max = 1 / np.sqrt((2 * distance_m * (1 / height)) ** 2 + 1) / wavelengths
    fx_max = 1 / np.sqrt((2 * distance_m * (1 / width)) ** 2 + 1) / wavelengths
    band = (np.abs(FX[None]) < fx_max[:, None, None]) & (np.abs(FY[None]) < fy_max[:, None, None])
    band = torch.tensor(band.astype(np.uint8), dtype=torch.float32).unsqueeze(0).to(device)

    kernel = torch.complex(band * torch.cos(phase), band * torch.sin(phase))
    return torch.fft.ifftshift(kernel, dim=_SPATIAL_DIMS)


def asm_propagate(field: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Propagate a complex field [..., H, W] with a kernel from `asm_kernel` (broadcast over leading dims)."""
    height, width = field.shape[-2:]
    padded = F.pad(field, (*_pad_widths(width), *_pad_widths(height)))
    spectrum = torch.fft.fftn(torch.fft.ifftshift(padded, dim=_SPATIAL_DIMS), dim=_SPATIAL_DIMS, norm='ortho')
    out = torch.fft.fftshift(torch.fft.ifftn(kernel * spectrum, dim=_SPATIAL_DIMS, norm='ortho'), dim=_SPATIAL_DIMS)
    top, left = (height + 1) // 2, (width + 1) // 2
    return out[..., top:top + height, left:left + width]
