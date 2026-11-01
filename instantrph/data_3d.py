"""
Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
Conference: SIGGRAPH Asia 2026
Project Page: https://opueyociutad.github.io/InstantRPH_Web/

3D dataset: pairs of an MPI volume (RGBA layers) and its target focal stack.

Supported layouts under the data root:
  * diopter_mpi_dataset/<scene>/ and focal_mpi_dataset/<scene>/ holding image sequences;
  * one folder per scene holding an MPI file (name contains mpi/volume/diopter/layer) and a
    focal-stack file (name contains focal/stack/target), as .pt/.pth/.npy/.npz or images;
  * flat files paired by name, e.g. <scene>_mpi.pt and <scene>_focal_stack.pt.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from instantrph.common import DATASET_3D_DIR

TENSOR_SUFFIXES = {'.pt', '.pth', '.npy', '.npz'}
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
SPLIT_ROOTS = ('diopter_mpi_dataset', 'focal_mpi_dataset')
_TENSOR_KEYS = ('rgb_vol', 'mpi', 'volume', 'focal_stack', 'stack', 'target')


@dataclass(frozen=True)
class ScenePair:
    scene_dir: Path
    mpi_path: Path
    focal_stack_path: Path


def _load_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        if image.mode == 'P':
            image = image.convert('RGBA' if 'transparency' in image.info else 'RGB')
        elif image.mode in ('RGBA', 'LA'):
            image = image.convert('RGBA')
        elif image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    return torch.from_numpy(arr).permute(2, 0, 1)


def _load_tensor_file(path: Path) -> torch.Tensor:
    if path.suffix.lower() == '.npy':
        return torch.from_numpy(np.asarray(np.load(path)))
    if path.suffix.lower() == '.npz':
        data = np.load(path)
        key = next((k for k in (*_TENSOR_KEYS, 'arr_0') if k in data), data.files[0])
        return torch.from_numpy(np.asarray(data[key]))
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        candidates = [obj.get(key) for key in (*_TENSOR_KEYS, 'image')] + list(obj.values())
        for value in candidates:
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, np.ndarray):
                return torch.from_numpy(value)
    raise ValueError(f'Unsupported tensor container in {path}')


def load_any_tensor(path: Path) -> torch.Tensor:
    """A tensor file, a single image ([C,H,W]) or a folder of images ([N,C,H,W])."""
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise ValueError(f'No image files found in {path}')
        return torch.stack([_load_image(p) for p in files], dim=0)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return _load_image(path)
    if path.suffix.lower() in TENSOR_SUFFIXES:
        return _load_tensor_file(path)
    raise ValueError(f'Unsupported file type: {path}')


def _is_data_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in TENSOR_SUFFIXES | IMAGE_SUFFIXES


def _is_mpi_name(name: str) -> bool:
    return any(token in name.lower() for token in ('mpi', 'rgb_vol', 'volume', 'diopter', 'layer'))


def _is_focal_stack_name(name: str) -> bool:
    return any(token in name.lower() for token in ('focal_stack', 'focal', 'stack', 'target'))


def _split_root_pairs(root: Path) -> list[ScenePair]:
    if root.name == SPLIT_ROOTS[0]:
        mpi_root, focal_root = root, root.parent / SPLIT_ROOTS[1]
    elif root.name == SPLIT_ROOTS[1]:
        mpi_root, focal_root = root.parent / SPLIT_ROOTS[0], root
    else:
        mpi_root, focal_root = root / SPLIT_ROOTS[0], root / SPLIT_ROOTS[1]
    if not (mpi_root.is_dir() and focal_root.is_dir()):
        return []
    mpi_scenes = {p.name: p for p in mpi_root.iterdir() if p.is_dir()}
    focal_scenes = {p.name: p for p in focal_root.iterdir() if p.is_dir()}
    return [
        ScenePair(mpi_scenes[name], mpi_scenes[name], focal_scenes[name])
        for name in sorted(mpi_scenes.keys() & focal_scenes.keys())
        if any(map(_is_data_file, mpi_scenes[name].iterdir())) and any(map(_is_data_file, focal_scenes[name].iterdir()))
    ]


def discover_scene_pairs(root: Path) -> list[ScenePair]:
    if not root.exists():
        raise FileNotFoundError(f'Data root does not exist: {root}')
    pairs = _split_root_pairs(root)
    if pairs:
        return pairs

    for scene_dir in sorted([p for p in root.rglob('*') if p.is_dir()] + [root]):
        files = [p for p in sorted(scene_dir.iterdir()) if _is_data_file(p)]
        mpi_files = [p for p in files if _is_mpi_name(p.stem)]
        stack_files = [p for p in files if _is_focal_stack_name(p.stem)]
        if mpi_files and stack_files:
            pairs.append(ScenePair(scene_dir, mpi_files[0], stack_files[0]))
    if pairs:
        return pairs

    files = [p for p in root.rglob('*') if _is_data_file(p)]
    mpi_files = [p for p in files if _is_mpi_name(p.stem)]
    stack_files = [p for p in files if _is_focal_stack_name(p.stem)]
    if not mpi_files or not stack_files:
        raise ValueError(
            f'Could not find MPI/focal-stack pairs under {root}. Expected {SPLIT_ROOTS[0]}/<scene> and '
            f'{SPLIT_ROOTS[1]}/<scene>, or filenames containing mpi/diopter/volume and focal/stack.'
        )

    def scene_key(path: Path) -> str:
        stem = path.stem.lower()
        for token in ('_mpi', '-mpi', '_rgb_vol', '-rgb_vol', '_diopter', '-diopter', '_focal_stack', '-focal_stack', '_stack', '-stack'):
            stem = stem.replace(token, '')
        return stem

    stacks = {scene_key(p): p for p in stack_files}
    pairs = [ScenePair(p.parent, p, stacks[scene_key(p)]) for p in mpi_files if scene_key(p) in stacks]
    if not pairs:
        raise ValueError(f'Could not pair MPI files with focal stacks under {root}')
    return pairs


def _normalize_mpi(tensor: torch.Tensor, *, input_is_intensity: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Any RGB(A) MPI layout -> amplitude [3, D, H, W] and alpha [1, D, H, W] (depth = smallest dim)."""
    if tensor.ndim == 5 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    channel_axes = [i for i, size in enumerate(tensor.shape) if size in (3, 4)]
    if tensor.ndim < 4 or not channel_axes:
        raise ValueError(f'Could not identify an RGB(A) MPI tensor of shape {tuple(tensor.shape)}')
    tensor = tensor.movedim(channel_axes[-1], 0).contiguous()
    rgb = tensor[:3]
    alpha = tensor[3:4] if tensor.shape[0] >= 4 else torch.ones((1, *tensor.shape[1:]), dtype=tensor.dtype)
    rgb = torch.clamp(rgb, min=0.0)
    if input_is_intensity:
        rgb = rgb.sqrt()
    alpha = torch.clamp(alpha, 0.0, 1.0)
    depth_axis = int(torch.tensor(rgb.shape[1:]).argmin().item()) + 1
    return rgb.movedim(depth_axis, 1).contiguous(), alpha.movedim(depth_axis, 1).contiguous()


def _normalize_focal_stack(tensor: torch.Tensor) -> torch.Tensor:
    """Any focal-stack layout -> [S, 3, H, W] in [0, 1] (S = smallest non-channel dim)."""
    if tensor.ndim == 4 and tensor.shape[-1] in (1, 3):
        tensor = tensor.movedim(-1, 1)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim < 4:
        raise ValueError(f'Expected a focal stack with at least 3 dimensions, got {tuple(tensor.shape)}')
    channel_axes = [i for i, size in enumerate(tensor.shape) if size in (1, 3)]
    tensor = tensor.movedim(channel_axes[-1] if channel_axes else 1, 1).contiguous()
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    other_axes = [i for i in range(tensor.ndim) if i != 1]
    stack_axis = other_axes[int(torch.tensor([tensor.shape[i] for i in other_axes]).argmin().item())]
    return tensor.movedim(stack_axis, 0).contiguous().clamp(0.0, 1.0)


class MPIFocalStackDataset(Dataset):
    """Samples: {'scene_name', 'rgb_vol' [3,D,H,W] amplitude, 'alpha_vol' [1,D,H,W], 'focal_stack' [S,3,H,W]}.

    With augmentation, applies random horizontal/vertical flips and a random RGB permutation
    (same permutation for the MPI and its focal stack).
    """

    def __init__(
        self,
        root: str | Path,
        *,
        input_is_intensity: bool = True,
        enable_augmentation: bool = False,
        flip_h_prob: float = 0.5,
        flip_v_prob: float = 0.5,
        shuffle_rgb_prob: float = 1.0,
    ) -> None:
        self.input_is_intensity = input_is_intensity
        self.enable_augmentation = enable_augmentation
        self.flip_h_prob = flip_h_prob
        self.flip_v_prob = flip_v_prob
        self.shuffle_rgb_prob = shuffle_rgb_prob
        self.scenes = discover_scene_pairs(Path(root))

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> dict[str, object]:
        pair = self.scenes[index]
        rgb_vol, alpha_vol = _normalize_mpi(load_any_tensor(pair.mpi_path).float(), input_is_intensity=self.input_is_intensity)
        focal_stack = _normalize_focal_stack(load_any_tensor(pair.focal_stack_path).float())
        if self.enable_augmentation:
            for prob, dim in ((self.flip_h_prob, -1), (self.flip_v_prob, -2)):
                if np.random.rand() < prob:
                    rgb_vol, alpha_vol, focal_stack = (torch.flip(x, dims=[dim]) for x in (rgb_vol, alpha_vol, focal_stack))
            if np.random.rand() < self.shuffle_rgb_prob:
                permutation = torch.randperm(3)
                rgb_vol, focal_stack = rgb_vol[permutation], focal_stack[:, permutation]
        return {'scene_name': pair.scene_dir.name, 'rgb_vol': rgb_vol, 'alpha_vol': alpha_vol, 'focal_stack': focal_stack}


def split_dataset_indices(num_items: int, *, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Seeded random (train, val) index split."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f'val_fraction must be in [0, 1), got {val_fraction}')
    indices = list(range(num_items))
    np.random.default_rng(seed).shuffle(indices)
    val_count = int(round(num_items * val_fraction))
    return indices[val_count:], indices[:val_count]


def collate_scene_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {'scene_name': [str(item['scene_name']) for item in batch]}
    for key in ('rgb_vol', 'alpha_vol', 'focal_stack'):
        out[key] = torch.stack([item[key] for item in batch], dim=0)  # type: ignore[misc]
    return out


def run_eval_loader(
    config: dict, *, data_root: str | None = None, split: str = 'auto', batch_size: int = 1, num_workers: int = 0
) -> tuple[DataLoader, str]:
    """Scenes of a train_3d.py run's split ('train', 'val', 'all', or 'auto': val if any), in order.
    Without data_root, the run's data root, or data/dataset_3d when that is not on this machine."""
    if not data_root:
        data_root = config['data_root']
        if not Path(data_root).exists():
            print(f'[Data] {data_root} not found, using {DATASET_3D_DIR}')
            data_root = DATASET_3D_DIR
    dataset = MPIFocalStackDataset(data_root, input_is_intensity=not config.get('input_is_amplitude', False))
    if config.get('debug_single_sample'):
        train_indices = val_indices = [int(config.get('debug_sample_index', 0)) % len(dataset)]
    else:
        train_indices, val_indices = split_dataset_indices(
            len(dataset), val_fraction=float(config.get('val_fraction', 0.1)), seed=int(config.get('seed', 42))
        )
    if split == 'auto':
        split = 'val' if val_indices else 'train'
    if split == 'val' and not val_indices:
        raise ValueError('The run has no validation scenes; use --eval-split train or all')
    indices = {'train': train_indices, 'val': val_indices, 'all': list(range(len(dataset)))}[split]
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=torch.cuda.is_available(), collate_fn=collate_scene_batch)
    return loader, split
