"""
Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
Conference: SIGGRAPH Asia 2026
Project Page: https://opueyociutad.github.io/InstantRPH_Web/

Image metrics. Reconstructions are color-balanced to the target before every metric.
"""
from __future__ import annotations

import torch


def colorbalance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Scale each channel of `pred` to the mean intensity of `target`."""
    pred_mean = pred.mean(dim=[-1, -2], keepdim=True)
    target_mean = target.mean(dim=[-1, -2], keepdim=True)
    if pred.ndim == 4 and target_mean.ndim == 3:
        target_mean = target_mean.unsqueeze(1)
    return pred * (target_mean / (pred_mean + 1e-8))


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Per-image PSNR in dB of [B, C, H, W] images."""
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3)).clamp(min=1e-12)
    return 10.0 * torch.log10((max_val ** 2) / mse)


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Per-image SSIM of [B, C, H, W] images, computed on the channel-averaged (grayscale) image."""
    from skimage.metrics import structural_similarity

    pred_np = pred.detach().cpu().numpy().mean(axis=1)
    target_np = target.detach().cpu().numpy().mean(axis=1)
    values = [structural_similarity(t, p, data_range=data_range) for p, t in zip(pred_np, target_np)]
    return torch.tensor(values, dtype=torch.float32, device=pred.device)
