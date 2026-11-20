"""Cartesian undersampling masks, following the fastMRI conventions.

fastMRI knee data are acquired line by line along the phase-encoding direction
(the last dimension, W). A mask keeps a fully sampled band of low frequencies
(``center_fraction``) and a subset of the remaining columns so that, on average,
one column in ``acceleration`` is kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class MaskConfig:
    kind: str = "random"  # "random" or "equispaced"
    acceleration: int = 4
    center_fraction: float = 0.08


class CartesianMask:
    """Generate 1D column masks broadcast to (H, W).

    A ``seed`` can be given at call time so that validation and test masks are
    identical across runs and methods (fair comparison), while training masks
    are drawn freshly at every epoch (data augmentation).
    """

    def __init__(self, kind: str = "random", acceleration: int = 4, center_fraction: float = 0.08):
        if kind not in {"random", "equispaced"}:
            raise ValueError(f"Unknown mask kind: {kind}")
        if acceleration < 1:
            raise ValueError("acceleration must be >= 1")
        if not 0 < center_fraction < 1:
            raise ValueError("center_fraction must be in (0, 1)")
        self.kind = kind
        self.acceleration = acceleration
        self.center_fraction = center_fraction

    @classmethod
    def from_config(cls, cfg: MaskConfig | dict) -> "CartesianMask":
        if isinstance(cfg, dict):
            cfg = MaskConfig(**cfg)
        return cls(cfg.kind, cfg.acceleration, cfg.center_fraction)

    def column_mask(self, num_cols: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        num_low = int(round(num_cols * self.center_fraction))
        mask = np.zeros(num_cols, dtype=bool)
        pad = (num_cols - num_low + 1) // 2
        mask[pad : pad + num_low] = True

        if self.acceleration == 1:
            mask[:] = True
            return mask

        if self.kind == "random":
            # Probability chosen so that the expected total is num_cols / acceleration.
            prob = (num_cols / self.acceleration - num_low) / (num_cols - num_low)
            prob = float(np.clip(prob, 0.0, 1.0))
            mask |= rng.uniform(size=num_cols) < prob
        else:  # equispaced, with a random offset
            offset = int(rng.integers(0, self.acceleration))
            mask[offset :: self.acceleration] = True
        return mask

    def __call__(self, shape: tuple[int, int], seed: int | None = None) -> torch.Tensor:
        """Return a float mask of shape (1, W) that broadcasts over (H, W)."""
        _, num_cols = shape
        cols = self.column_mask(num_cols, seed)
        return torch.from_numpy(cols.astype(np.float32)).unsqueeze(0)

    def __repr__(self) -> str:
        return f"CartesianMask(kind={self.kind!r}, acceleration={self.acceleration}, center_fraction={self.center_fraction})"
