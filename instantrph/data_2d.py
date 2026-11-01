"""
Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
Conference: SIGGRAPH Asia 2026
Project Page: https://opueyociutad.github.io/InstantRPH_Web/

2D image datasets (DIV2K by default). Images are bilinearly resized to the SLM resolution.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def load_rgb(path: Path, image_hw: Sequence[int]) -> torch.Tensor:
    """[3, H, W] intensity image in [0, 1]."""
    arr = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    image = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return F.interpolate(image, size=tuple(image_hw), mode='bilinear', align_corners=False).squeeze(0)


class ImageFolder(Dataset):
    """All images below a folder (recursively), sorted by path. Each image is its own target."""

    def __init__(self, root: str | Path, image_hw: Sequence[int]) -> None:
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f'Image directory does not exist: {root}')
        self.files = sorted(p for p in root.rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.files:
            raise ValueError(f'No images found in {root}')
        self.image_hw = tuple(image_hw)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        return load_rgb(self.files[index], self.image_hw)


class SingleSample(Dataset):
    """One fixed image, for smoke tests (--debug-single-sample)."""

    def __init__(self, sample: torch.Tensor) -> None:
        self.sample = sample

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _: int) -> torch.Tensor:
        return self.sample


def build_loaders(
    datasets_root: str | Path,
    train_subdir: str,
    val_subdir: str,
    *,
    image_hw: Sequence[int],
    batch_size: int,
    num_workers: int,
    debug_sample_index: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Train/val loaders. With `debug_sample_index`, both serve that one training image."""
    root = Path(datasets_root)
    if not root.exists():
        raise FileNotFoundError(f'Datasets root does not exist: {root}')
    train_set: Dataset = ImageFolder(root / train_subdir, image_hw)
    val_set: Dataset = ImageFolder(root / val_subdir, image_hw)
    if debug_sample_index is not None:
        train_set = val_set = SingleSample(train_set[debug_sample_index % len(train_set)])

    def loader(dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          pin_memory=torch.cuda.is_available())

    return loader(train_set, True), loader(val_set, False)
