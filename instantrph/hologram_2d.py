"""
Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
Conference: SIGGRAPH Asia 2026
Project Page: https://opueyociutad.github.io/InstantRPH_Web/

2D random-phase holography with amplitude discard (the 2D forward model the LAE is trained through).

For an intensity image I and an encoder f (the LAE, or the identity for the baseline):
  1. f(sqrt(I)) gives C * tmx amplitude maps, each paired with its own random phase;
  2. each field is propagated to the SLM (ASM, -distance), where only its phase is kept;
  3. it is propagated back to the image plane (+distance) and the tmx intensities are averaged;
  4. the result is color-balanced to the target's mean intensity.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from instantrph.asm import asm_kernel, asm_propagate
from instantrph.common import load_phase
from instantrph.constants import WAVELENGTHS_M


class Hologram2D:
    def __init__(self, phase_pattern: torch.Tensor, distance_m: float, device: torch.device) -> None:
        """phase_pattern: [C, tmx, H, W] random phases, one per color channel and frame."""
        self.channels, self.tmx, height, width = phase_pattern.shape
        self.phase_pattern = phase_pattern.to(device)
        self.distance_m = float(distance_m)
        wavelengths = WAVELENGTHS_M[:self.channels]
        # [1, C, 1, 2H, 2W]: one kernel per wavelength, shared by the tmx frames.
        self.kernel_to_slm = asm_kernel((height, width), wavelengths, -self.distance_m, device=device).unsqueeze(2)
        self.kernel_from_slm = asm_kernel((height, width), wavelengths, self.distance_m, device=device).unsqueeze(2)

    @classmethod
    def from_phase_file(
        cls,
        phase_file: str | Path,
        *,
        tmx: int,
        layer_id: int,
        distance_m: float,
        image_hw: Sequence[int],
        device: torch.device,
    ) -> Hologram2D:
        """Use layer `layer_id` of a [LAYERS, TMX, RGB, H, W] phase file, cropped to image_hw."""
        phase = load_phase(phase_file, tmx=tmx, image_hw=image_hw, layers=slice(layer_id, layer_id + 1))[0]
        return cls(phase.permute(1, 0, 2, 3).contiguous(), distance_m, device)

    def fields(self, encoder: nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(SLM field before amplitude discard, reconstructed image-plane field), both [B, C, tmx, H, W]."""
        amplitude = encoder(images.sqrt())
        batch_size, _, height, width = amplitude.shape
        field = amplitude.view(batch_size, self.channels, self.tmx, height, width) * torch.exp(1j * self.phase_pattern)
        slm = asm_propagate(field, self.kernel_to_slm)
        phase_only = slm / (slm.abs() + 1e-8)
        return slm, asm_propagate(phase_only, self.kernel_from_slm)

    def __call__(self, encoder: nn.Module, images: torch.Tensor) -> torch.Tensor:
        """Time-averaged, color-balanced reconstruction [B, C, H, W] of intensity images [B, C, H, W]."""
        _, recon = self.fields(encoder, images)
        preds = (recon.abs() ** 2).mean(dim=2)
        return preds * (images.mean(dim=[2, 3], keepdim=True) / (preds.mean(dim=[2, 3], keepdim=True) + 1e-8))
