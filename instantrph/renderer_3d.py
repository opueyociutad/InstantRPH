"""
Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
Conference: SIGGRAPH Asia 2026
Project Page: https://opueyociutad.github.io/InstantRPH_Web/

Instant RPH: MPI amplitude volume -> phase-only SLM field
It also performs focal stack generation.

render_field builds the hologram. Walking the MPI from the farthest layer to the nearest, it
propagates the accumulated field across one layer spacing (compact RSD kernel, see data/models_3d/kernels/),
then blends in the current layer's amplitude, carrying its own random phase, with
sqrt(alpha) occlusion weights. It then propagates the result to the SLM (ASM) and discards
the amplitude. render_stack propagates that field to every focus plane and averages the
tmx intensities.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from instantrph.asm import asm_kernel, asm_propagate
from instantrph.common import KERNELS_DIR, PHASE_FILE_3D, load_phase, prepare_volume, run_phase_file, select_stack_channels
from instantrph.constants import WAVELENGTHS_M
from instantrph.metrics import colorbalance, psnr


def _interlayer_kernel_file(kernels_dir: Path, layer_spacing_m: float) -> Path:
    """kernels_dir/fullkernel.pt, or the <spacing>mm_* subfolder closest to layer_spacing_m."""
    if (kernels_dir / 'fullkernel.pt').exists():
        return kernels_dir / 'fullkernel.pt'
    candidates = []
    for folder in sorted(kernels_dir.glob('*')):
        try:
            spacing_mm = float(folder.name.split('mm')[0])
        except ValueError:
            continue
        if (folder / 'fullkernel.pt').exists():
            candidates.append((abs(spacing_mm - abs(layer_spacing_m) * 1000.0), folder / 'fullkernel.pt'))
    if not candidates:
        raise FileNotFoundError(f'Could not find fullkernel.pt under {kernels_dir}')
    return min(candidates, key=lambda item: item[0])[1]


class LayeredFocalStackRenderer:
    def __init__(
        self,
        *,
        image_rows: int,
        image_cols: int,
        layer_spacing_m: float,
        focus_distances_m: Sequence[float],
        phase_tensor: torch.Tensor,
        tmx: int,
        layer_id: int = 0,
        wavelength_indices: Sequence[int] = (0, 1, 2),
        compute_dtype: torch.dtype = torch.float32,
        use_autocast: bool = False,
        compile: bool = True,
        kernels_dir: str | Path = KERNELS_DIR,
    ) -> None:
        """
        phase_tensor: [LAYERS, >=tmx, C, H, W] random phases for the selected wavelengths; MPI
            layer d uses phase layer layer_id + d (clamped to the last one).
        focus_distances_m: focal-stack planes, as distances from the SLM. They are stored far to
            near: focal-stack index s is rendered at self.focus_distances_m[s], and the farthest
            plane is also the SLM distance.
        compute_dtype / use_autocast: precision of the inter-layer convolutions (and autocast on
            CUDA); FFT propagation always runs in complex64.
        compile: torch.compile the layer loop.
        """
        self.image_rows, self.image_cols = int(image_rows), int(image_cols)
        self.layer_spacing_m = float(layer_spacing_m)
        self.focus_distances_m = list(focus_distances_m)[::-1]
        self.z_SLM = -self.focus_distances_m[0]
        self.tmx = int(tmx)
        self.layer_id = int(layer_id)
        self.wavelength_indices = tuple(int(i) for i in wavelength_indices)
        self.color_channels = len(self.wavelength_indices)
        self.wavelengths_m = tuple(float(WAVELENGTHS_M[i]) for i in self.wavelength_indices)
        if phase_tensor.ndim != 5 or phase_tensor.shape[2] != self.color_channels or phase_tensor.shape[1] < self.tmx:
            raise ValueError(
                f'Expected a phase tensor [LAYERS, >={self.tmx}, {self.color_channels}, H, W], got {tuple(phase_tensor.shape)}'
            )
        self.phase_tensor = phase_tensor
        self.compute_dtype = compute_dtype
        self.use_autocast = use_autocast

        kernel = torch.load(_interlayer_kernel_file(Path(kernels_dir), self.layer_spacing_m), map_location='cpu')
        if kernel.shape[0] == 1:  # stored as [1, RGB, K, K]; depthwise conv2d wants [RGB, 1, K, K]
            kernel = kernel.permute(1, 0, 2, 3)
        kernel = kernel[list(self.wavelength_indices)]
        self._conv_kernel_real_cpu = kernel.real.to(torch.float32).contiguous()
        self._conv_kernel_imag_cpu = kernel.imag.to(torch.float32).contiguous()
        self._conv_padding = int(kernel.shape[-1] // 2)
        self._conv_kernel_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._transfer_cache: dict[float, torch.Tensor] = {}
        self._slm_transfer_cache: dict[str, torch.Tensor] = {}
        self._build_field = torch.compile(self._build_field_impl) if compile else self._build_field_impl

    @classmethod
    def from_run_config(
        cls,
        config: dict,
        *,
        image_hw: Sequence[int],
        focus_distances_m: Sequence[float],
        tmx: int,
        channels: Sequence[int],
        device: torch.device,
        **kwargs,
    ) -> LayeredFocalStackRenderer:
        """Renderer with the optics of a train_3d.py run (phase file, phase layer, layer spacing)."""
        phase = load_phase(run_phase_file(config, PHASE_FILE_3D), tmx=tmx, channels=channels, image_hw=image_hw)
        return cls(
            image_rows=image_hw[0],
            image_cols=image_hw[1],
            layer_spacing_m=float(config.get('layer_spacing_mm', 0.25)) / 1000.0,
            focus_distances_m=focus_distances_m,
            phase_tensor=phase.to(device),
            tmx=tmx,
            layer_id=int(config.get('phase_layer_id', 0)),
            wavelength_indices=channels,
            **kwargs,
        )

    # -- kernels -------------------------------------------------------------

    def _transfer(self, distance_m: float, device: torch.device) -> torch.Tensor:
        """ASM kernel for a focus plane (cached on the CPU, copied to `device` per call)."""
        if distance_m not in self._transfer_cache:
            self._transfer_cache[distance_m] = asm_kernel(
                (self.image_rows, self.image_cols), self.wavelengths_m, distance_m, device=device
            ).cpu()
        return self._transfer_cache[distance_m].to(device)

    def _slm_transfer(self, device: torch.device) -> torch.Tensor:
        """ASM kernel to the SLM; used by every frame, so it stays resident on `device`."""
        key = str(device)
        if key not in self._slm_transfer_cache:
            self._slm_transfer_cache[key] = self._transfer(self.z_SLM, device)
        return self._slm_transfer_cache[key]

    def _conv_kernels(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = f'{device}:{dtype}'
        if key not in self._conv_kernel_cache:
            self._conv_kernel_cache[key] = (
                self._conv_kernel_real_cpu.to(device=device, dtype=dtype),
                self._conv_kernel_imag_cpu.to(device=device, dtype=dtype),
            )
        return self._conv_kernel_cache[key]

    # -- hologram ------------------------------------------------------------

    def _propagate_between_layers(self, xr: torch.Tensor, xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Complex depthwise convolution with the RSD kernel, on separate real/imag [N, C, H, W] parts."""
        channels = xr.shape[1]
        kr, ki = self._conv_kernels(xr.device, self.compute_dtype)
        xr, xi = xr.to(self.compute_dtype), xi.to(self.compute_dtype)

        def conv(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
            return F.conv2d(x, k, padding=self._conv_padding, groups=channels)

        real, imag = conv(xr, kr) - conv(xi, ki), conv(xi, kr) + conv(xr, ki)
        return real.to(torch.float32), imag.to(torch.float32)

    def _build_field_impl(self, rgb_vol: torch.Tensor, alpha_vol: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Composite the MPI into the field at the nearest layer, as real/imag parts [B*tmx, C, H, W]."""
        if rgb_vol.ndim == 5:  # [B, C, D, H, W]: the same amplitude in every frame
            batch_size, color_channels, num_layers, height, width = rgb_vol.shape
            rgb_per_tmx = rgb_vol.unsqueeze(1).expand(-1, self.tmx, -1, -1, -1, -1)
        elif rgb_vol.ndim == 6 and rgb_vol.shape[1] == self.tmx:  # [B, tmx, C, D, H, W] from an encoder
            batch_size, _, color_channels, num_layers, height, width = rgb_vol.shape
            rgb_per_tmx = rgb_vol
        else:
            raise ValueError(f'Expected rgb_vol [B,C,D,H,W] or [B,{self.tmx},C,D,H,W], got {tuple(rgb_vol.shape)}')

        n = batch_size * self.tmx
        field_r = torch.zeros((n, color_channels, height, width), device=rgb_vol.device, dtype=torch.float32)
        field_i = torch.zeros_like(field_r)
        phase_layers = int(self.phase_tensor.shape[0])

        for layer_index in range(num_layers - 1, -1, -1):
            if self.layer_spacing_m != 0.0:
                field_r, field_i = self._propagate_between_layers(field_r, field_i)

            amp_flat = rgb_per_tmx[:, :, :, layer_index].clamp(min=0.0).reshape(n, color_channels, height, width)
            opa = alpha_vol[:, 0, layer_index].clamp(0.0, 1.0)
            opa_flat = opa.unsqueeze(1).expand(-1, self.tmx, -1, -1).reshape(n, 1, height, width)

            phase_idx = min(self.layer_id + layer_index, phase_layers - 1)
            layer_phase = self.phase_tensor[phase_idx, :self.tmx].to(device=rgb_vol.device, dtype=torch.float32)
            phase_flat = layer_phase.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).reshape(n, color_channels, height, width)

            layer_r = amp_flat * torch.cos(phase_flat)
            layer_i = amp_flat * torch.sin(phase_flat)
            sqrt_opa = torch.sqrt(opa_flat)
            sqrt_one_minus = torch.sqrt(1.0 - opa_flat)
            field_r = layer_r * sqrt_opa + field_r * sqrt_one_minus
            field_i = layer_i * sqrt_opa + field_i * sqrt_one_minus

        return field_r, field_i

    def render_field(self, rgb_vol: torch.Tensor, alpha_vol: torch.Tensor) -> torch.Tensor:
        """Phase-only SLM field [B, tmx, C, H, W] for an MPI amplitude volume and its alpha [B, 1, D, H, W]."""
        autocast = self.use_autocast and rgb_vol.device.type == 'cuda'
        with torch.autocast('cuda', dtype=self.compute_dtype) if autocast else nullcontext():
            field_r, field_i = self._build_field(rgb_vol, alpha_vol)
        field = torch.complex(field_r.to(torch.float32), field_i.to(torch.float32))
        field = asm_propagate(field, self._slm_transfer(rgb_vol.device))
        field = field.view(-1, self.tmx, *field.shape[1:])
        return field / (field.abs() + 1e-8)

    # -- focal stack ---------------------------------------------------------

    def focal_planes(self, field: torch.Tensor):
        """Yield the tmx-averaged intensity [B, C, H, W] of an SLM field at each focus plane."""
        batch_size, tmx, channels, height, width = field.shape
        for distance_m in self.focus_distances_m:
            flat = field.reshape(batch_size * tmx, channels, height, width)
            intensity = asm_propagate(flat, self._transfer(distance_m, field.device)).abs() ** 2
            yield intensity.view(batch_size, tmx, channels, height, width).mean(dim=1)

    def render_stack(self, rgb_vol: torch.Tensor, alpha_vol: torch.Tensor) -> torch.Tensor:
        """Focal stack [B, S, C, H, W] (not color-balanced)."""
        return torch.stack(list(self.focal_planes(self.render_field(rgb_vol, alpha_vol))), dim=1)

    def render_stack_loss(
        self, rgb_vol: torch.Tensor, alpha_vol: torch.Tensor, target_stack: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean L1 loss and PSNR over the color-balanced focus planes, one plane in memory at a time."""
        return self.field_stack_loss(self.render_field(rgb_vol, alpha_vol), target_stack)

    def field_stack_loss(self, field: torch.Tensor, target_stack: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """render_stack_loss for any SLM field [B, tmx, C, H, W], e.g. from another method such as ARH."""
        if target_stack.shape[1] != len(self.focus_distances_m):
            raise ValueError(f'Target stack has {target_stack.shape[1]} planes, the renderer {len(self.focus_distances_m)}')
        losses, psnrs = [], []
        for index, pred in enumerate(self.focal_planes(field)):
            target = target_stack[:, index]
            pred = colorbalance(pred, target)
            losses.append(F.l1_loss(pred, target))
            psnrs.append(psnr(pred, target).mean())
        return torch.stack(losses).mean(), torch.stack(psnrs).mean()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    renderer: LayeredFocalStackRenderer,
    loader,
    channels: Sequence[int],
    device: torch.device,
) -> tuple[float, float, int]:
    """Summed per-sample focal-stack L1 loss and PSNR over a loader, and the number of samples."""
    model.eval()
    total_loss = total_psnr = 0.0
    count = 0
    for batch in loader:
        rgb_vol, alpha_vol = prepare_volume(batch['rgb_vol'], batch['alpha_vol'], device, channels)
        target_stack = select_stack_channels(batch['focal_stack'].to(device=device, dtype=torch.float32), channels)
        loss, psnr_value = renderer.render_stack_loss(model(rgb_vol), alpha_vol, target_stack)
        total_loss += loss.item() * rgb_vol.shape[0]
        total_psnr += psnr_value.item() * rgb_vol.shape[0]
        count += rgb_vol.shape[0]
    return total_loss, total_psnr, count
