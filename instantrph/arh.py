"""
Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
Conference: SIGGRAPH Asia 2026
Project Page: https://opueyociutad.github.io/InstantRPH_Web/

Adapted from Artifact-Resilient Real-Time Holography (ARH): holonet.py,
utils/pytorch_prototyping/pytorch_prototyping.py and the ARH 3D rendering.
    https://github.com/princeton-computational-imaging/ArtifactResilientHolography
    V. Chu, O. Pueyo-Ciutad, E. Tseng, F. Schiffers, G. Kuo, N. Matsuda, A. Redo-Sanchez, D. Lanman,
    O. Cossairt, F. Heide. Artifact-Resilient Real-Time Holography. ACM Transactions on Graphics
    (SIGGRAPH Asia), 2025.
    HoloNet itself is from Neural Holography [Peng et al. 2020]. The original code is released under BSL-1.0.

ARH baseline: HoloNet-based phase encoding of a hard-RGBD version of the MPI (inference only).

The U-Net modules follow the ARH code exactly, so the released per-color, per-frame
checkpoints ({red,green,blue}_{frame}.pth) load unchanged.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from instantrph.asm import asm_kernel, asm_propagate
from instantrph.constants import WAVELENGTHS_M
from instantrph.viz import composite_mpi

COLOR_NAMES = ('red', 'green', 'blue')


# ---------------------------------------------------------------------------
# U-Net (module layout must match the released checkpoints)
# ---------------------------------------------------------------------------


class Conv2dSame(nn.Module):
    """Conv2d with reflection padding that preserves the spatial size."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool = True) -> None:
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(nn.ReflectionPad2d((ka, kb, ka, kb)), nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias))
        self.weight = self.net[1].weight
        self.bias = self.net[1].bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _norm_act(channels: int, norm: type[nn.Module] | None, act: nn.Module) -> list[nn.Module]:
    return ([norm(channels, affine=True)] if norm is not None else []) + [act]


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: type[nn.Module] | None = nn.BatchNorm2d) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(in_channels, in_channels, kernel_size=3, bias=norm is None),
            *_norm_act(in_channels, norm, nn.LeakyReLU(0.2, True)),
            nn.ReflectionPad2d(1), nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, bias=norm is None),
            *_norm_act(out_channels, norm, nn.LeakyReLU(0.2, True)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: type[nn.Module] | None = nn.BatchNorm2d) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=norm is None),
            *_norm_act(out_channels, norm, nn.ReLU(True)),
            Conv2dSame(out_channels, out_channels, kernel_size=3, bias=norm is None),
            *_norm_act(out_channels, norm, nn.ReLU(True)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc: int, inner_nc: int, norm: type[nn.Module] | None = nn.BatchNorm2d, submodule: nn.Module | None = None) -> None:
        super().__init__()
        inner = [] if submodule is None else [submodule]
        up_in = inner_nc if submodule is None else 2 * inner_nc
        self.model = nn.Sequential(DownBlock(outer_nc, inner_nc, norm=norm), *inner, UpBlock(up_in, outer_nc, norm=norm))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.model(x)], dim=1)


class Unet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, nf0: int, num_down: int, max_channels: int, norm: type[nn.Module] = nn.BatchNorm2d) -> None:
        super().__init__()
        self.in_layer = nn.Sequential(Conv2dSame(in_channels, nf0, kernel_size=3, bias=False), norm(nf0, affine=True), nn.LeakyReLU(0.2, True))
        channels = min(2 ** (num_down - 1) * nf0, max_channels)
        block = UnetSkipConnectionBlock(channels, channels, norm=None)
        for i in reversed(range(num_down - 1)):
            block = UnetSkipConnectionBlock(min(2 ** i * nf0, max_channels), min(2 ** (i + 1) * nf0, max_channels), norm=norm, submodule=block)
        self.unet_block = block
        self.out_layer = nn.Sequential(Conv2dSame(2 * nf0, out_channels, kernel_size=3, bias=True))
        self.out_layer_weight = self.out_layer[0].weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.unet_block(self.in_layer(x)))


class PhaseTanh(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.pi * torch.tanh(x)


class InitialPhaseUnet(nn.Module):
    """Target amplitude -> initial phase."""

    def __init__(self, num_down: int = 8, num_features_init: int = 32, max_features: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(Unet(1, 1, num_features_init, num_down, max_features), PhaseTanh())

    def forward(self, amp: torch.Tensor) -> torch.Tensor:
        return self.net(amp)


class FinalPhaseOnlyUnet(nn.Module):
    """SLM amplitude and phase -> phase-only SLM pattern."""

    def __init__(self, num_down: int = 8, num_features_init: int = 32, max_features: int = 256, num_in: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(Unet(num_in, 1, num_features_init, num_down, max_features), PhaseTanh())

    def forward(self, amp_phase: torch.Tensor) -> torch.Tensor:
        return self.net(amp_phase)


class HoloNet(nn.Module):
    """Holds the two U-Nets under the attribute names of the released checkpoints."""

    def __init__(self, initial_phase: nn.Module, final_phase_only: nn.Module) -> None:
        super().__init__()
        self.initial_phase = initial_phase
        self.final_phase_only = final_phase_only


# ---------------------------------------------------------------------------
# ARH rendering
# ---------------------------------------------------------------------------


def load_arh_models(model_dir: str, tmx: int, device: torch.device) -> list[list[HoloNet]]:
    """[tmx][color] grid of HoloNets loaded from {color}_{frame}.pth (frames are 1-indexed)."""
    models = []
    for frame in range(tmx):
        row = []
        for color in COLOR_NAMES:
            model = HoloNet(InitialPhaseUnet(4, 16), FinalPhaseOnlyUnet(4, 16, num_in=2)).to(device)
            model.load_state_dict(torch.load(os.path.join(model_dir, f'{color}_{frame + 1}.pth'), map_location=device))
            row.append(model.eval())
        models.append(row)
    return models


def layer_kernels(height: int, width: int, depth: int, z_dist_slm_m: float, layer_spacing_m: float, device: torch.device) -> list[torch.Tensor]:
    """ASM kernels from each MPI layer (layer 0 at z_dist_slm_m, then layer_spacing_m closer) to the SLM."""
    return [asm_kernel((height, width), WAVELENGTHS_M, -(z_dist_slm_m - i * layer_spacing_m), device=device) for i in range(depth)]


def mpi_to_rgbd(rgb_vol: torch.Tensor, alpha_vol: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Hard-RGBD conversion of an MPI ([1,3,D,H,W] amplitude, [1,1,D,H,W] alpha).

    Returns the composited 2D image [1,3,H,W] (input of the initial-phase network) and the
    volume [1,3,D,H,W] that places that image, per pixel, on the layer with the largest alpha
    contribution alpha_d * prod_{j<d} (1 - alpha_j) (layer 0 nearest), as ARH's
    simulate_rgbd_from_layered_volume(depth_mode='argmax').
    """
    rgb_image = composite_mpi(rgb_vol, alpha_vol, front_to_back=True)
    alpha = alpha_vol[0, 0]
    transmittance = torch.cumprod(torch.cat([torch.ones_like(alpha[:1]), 1.0 - alpha[:-1]], dim=0), dim=0)
    depth = (alpha * transmittance).argmax(dim=0, keepdim=True)  # first maximum, as in ARH
    mask = torch.zeros_like(alpha).scatter_(0, depth, 1.0)
    return rgb_image, rgb_image[:, :, None] * mask[None, None]


@torch.no_grad()
def arh_render_field(
    rgb_vol: torch.Tensor,
    alpha_vol: torch.Tensor,
    *,
    models: list[list[HoloNet]],
    kernels: list[torch.Tensor],
) -> torch.Tensor:
    """Phase-only SLM field [1, tmx, 3, H, W] for one MPI sample.

    Per frame: initial phases from the composited image, coherent sum of every layer's masked
    field propagated to the SLM, then the final phase-only encoding, per color.
    """
    rgb_image, rgbd_vol = mpi_to_rgbd(rgb_vol, alpha_vol)
    tmx, depth = len(models), rgb_vol.shape[2]
    height, width = rgb_vol.shape[-2:]
    device = rgb_vol.device

    refined_phases = torch.zeros(tmx, 3, height, width, dtype=torch.float32, device=device)
    for frame in range(tmx):
        init_phases = torch.zeros(3, height, width, device=device)
        for color in range(3):
            init_phases[color] = models[frame][color].initial_phase(rgb_image[:, [color]]).squeeze()
        init_phases = init_phases % (2.0 * torch.pi) - torch.pi  # as in ARH's HoloNet.forward (phase_shift = 0)

        slm_field = torch.zeros(3, height, width, dtype=torch.cfloat, device=device)
        for layer in range(depth):
            layer_field = rgbd_vol[0, :, layer] * torch.exp(1j * init_phases)
            for color in range(3):
                slm_field[color:color + 1] += asm_propagate(layer_field.unsqueeze(0)[:, color:color + 1], kernels[layer][:, color:color + 1]).squeeze(0)

        for color in range(3):
            amp_phase = torch.stack([slm_field[color].abs(), slm_field[color].angle()])[None]
            refined_phases[frame, color] = models[frame][color].final_phase_only(amp_phase).squeeze()

    return torch.exp(1j * refined_phases).unsqueeze(0)
