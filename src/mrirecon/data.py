"""fastMRI single-coil knee dataset.

Each fastMRI HDF5 file is one volume (one patient exam) and contains:

* ``kspace``: complex64 array (num_slices, H, W), typically H = 640, W ~ 368,
  oversampled by 2 along the readout (H) direction;
* ``reconstruction_esc``: float32 array (num_slices, 320, 320), the ground-truth
  magnitude image (absent from the official test set).

Two modes are provided:

* ``mode="image"`` for image-domain networks (U-Net): returns 320 x 320 magnitude
  images normalised by the zero-filled mean/std, so batches can be stacked.
* ``mode="kspace"`` for model-based methods (TV, unrolled networks): returns the
  full undersampled k-space and mask, scaled by a per-slice constant. Widths differ
  between volumes, so use ``batch_size=1`` in this mode.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .fft import center_crop, ifft2c
from .masks import CartesianMask

TARGET_SIZE = (320, 320)


def volume_seed(fname: str) -> int:
    """Deterministic seed per volume, so evaluation masks are reproducible."""
    return zlib.crc32(Path(fname).name.encode()) & 0xFFFFFFFF


def load_split(split_file: str | Path, split: str, data_root: str | Path) -> list[Path]:
    """Read ``splits.json`` (written by scripts/split_data.py) and return absolute paths."""
    with open(split_file) as f:
        splits = json.load(f)
    root = Path(data_root)
    return [root / name for name in splits[split]]


@dataclass
class SliceRef:
    path: Path
    slice_idx: int
    num_slices: int


class FastMRISliceDataset(Dataset):
    def __init__(
        self,
        files: list[Path],
        mask_func: CartesianMask,
        mode: str = "image",
        deterministic_mask: bool = False,
        skip_edge_slices: int = 0,
        max_volumes: int | None = None,
    ):
        if mode not in {"image", "kspace"}:
            raise ValueError(f"mode must be 'image' or 'kspace', got {mode}")
        self.mask_func = mask_func
        self.mode = mode
        self.deterministic_mask = deterministic_mask
        self.slices: list[SliceRef] = []

        files = sorted(Path(f) for f in files)
        if max_volumes is not None:
            files = files[:max_volumes]
        for path in files:
            with h5py.File(path, "r") as hf:
                n = hf["kspace"].shape[0]
            for s in range(skip_edge_slices, n - skip_edge_slices):
                self.slices.append(SliceRef(path, s, n))
        if not self.slices:
            raise RuntimeError("No slices found: check the data path and split file.")

    def __len__(self) -> int:
        return len(self.slices)

    @staticmethod
    def _read(ref: SliceRef) -> tuple[torch.Tensor, torch.Tensor | None]:
        with h5py.File(ref.path, "r") as hf:
            kspace = torch.from_numpy(hf["kspace"][ref.slice_idx])
            target = None
            if "reconstruction_esc" in hf:
                target = torch.from_numpy(hf["reconstruction_esc"][ref.slice_idx].astype(np.float32))
        return kspace, target

    def __getitem__(self, idx: int) -> dict:
        ref = self.slices[idx]
        kspace, target = self._read(ref)
        if target is None:  # fall back to the fully sampled reconstruction
            target = center_crop(ifft2c(kspace).abs(), TARGET_SIZE)

        seed = volume_seed(ref.path.name) if self.deterministic_mask else None
        mask = self.mask_func(tuple(kspace.shape[-2:]), seed=seed)
        masked_kspace = kspace * mask
        zero_filled = ifft2c(masked_kspace)

        meta = {"fname": ref.path.name, "slice_idx": ref.slice_idx}

        if self.mode == "image":
            zf = center_crop(zero_filled.abs(), TARGET_SIZE)
            mean, std = zf.mean(), zf.std() + 1e-11
            image = ((zf - mean) / std).clamp(-6, 6)
            target_n = ((target - mean) / std).clamp(-6, 6)
            return {
                "image": image.unsqueeze(0),  # (1, 320, 320)
                "target": target_n.unsqueeze(0),
                "target_raw": target.unsqueeze(0),
                "mean": mean,
                "std": std,
                **meta,
            }

        scale = zero_filled.abs().max() + 1e-11
        return {
            "masked_kspace": masked_kspace / scale,  # complex (H, W)
            "mask": mask,  # (1, W)
            "target": target / scale,  # (320, 320)
            "target_raw": target,
            "scale": scale,
            **meta,
        }
