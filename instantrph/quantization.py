"""
Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
Conference: SIGGRAPH Asia 2026
Project Page: https://opueyociutad.github.io/InstantRPH_Web/

SLM phase quantization with the measured phase lookup tables, for hardware-realistic evaluation.
"""
from __future__ import annotations

from typing import Sequence

import torch

# Measured phase (rad) of each of the SLM's 16 levels, per color channel (660, 520, 450 nm).
PHASE_LUTS = (
    torch.tensor([0.0000, 0.0468, 0.0957, 0.1967, 0.2609, 0.3387, 0.5271, 0.8085, 1.5969,
                  1.7280, 1.9710, 2.2918, 2.7944, 3.1100, 3.7153, 4.3704]),
    torch.tensor([0.0000, 0.0666, 0.1362, 0.2799, 0.3714, 0.4821, 0.7502, 1.1508, 2.2730,
                  2.4597, 2.8055, 3.2621, 3.9775, 4.4266, 5.2882, 6.2207]),
    torch.tensor([0.0000, 0.0806, 0.1650, 0.3391, 0.4499, 0.5840, 0.9088, 1.3942,
                  2.7536, 2.9797, 3.3987, 3.9519, 4.8185, 5.3626, 6.4063, 7.5360]),
)


def quantize_phase(phase: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Snap each phase to the nearest LUT level (both wrapped to [0, 2pi))."""
    lut = lut % (2 * torch.pi)
    distance = torch.abs((phase % (2 * torch.pi)).unsqueeze(0) - lut.reshape(-1, *([1] * phase.ndim)))
    level = torch.argmin(distance, dim=0).float()
    return lut.flatten()[level.int()] % (2 * torch.pi)


def quantize_rgb_phase(phase: torch.Tensor, channels: Sequence[int] = (0, 1, 2)) -> torch.Tensor:
    """Quantize an [N, C, H, W] phase pattern of the given wavelength channels with their LUTs."""
    return torch.stack([quantize_phase(phase[:, i], PHASE_LUTS[c].to(phase.device)) for i, c in enumerate(channels)], dim=1)


def block_dc(field: torch.Tensor, half_width: int = 11) -> torch.Tensor:
    """Zero a (2 * half_width)^2 block around the DC term of the field's spectrum."""
    spectrum = torch.fft.fftshift(torch.fft.fft2(field), dim=(-2, -1))
    rows, cols = spectrum.shape[-2] // 2, spectrum.shape[-1] // 2
    spectrum[..., rows - half_width:rows + half_width, cols - half_width:cols + half_width] = 0
    return torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=(-2, -1)))


def quantize_slm_field(field: torch.Tensor, channels: Sequence[int] = (0, 1, 2)) -> torch.Tensor:
    """Phase-only SLM field [B, TMX, C, H, W] as displayed: LUT-quantized phase, then DC block."""
    batch_size, tmx, num_channels, height, width = field.shape
    phase = quantize_rgb_phase(torch.angle(field.reshape(batch_size * tmx, num_channels, height, width)) + torch.pi, channels)
    return block_dc(torch.exp(1j * phase)).reshape(batch_size, tmx, num_channels, height, width)
