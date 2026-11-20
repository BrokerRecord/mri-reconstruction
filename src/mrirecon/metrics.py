"""Evaluation metrics, following the fastMRI challenge conventions.

Metrics are computed per *volume* (all slices of an exam stacked together), with
the data range set to the maximum of the ground-truth volume, then averaged over
volumes. Computing them per slice would give optimistic and less comparable numbers.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def nmse(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2)


def psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(gt, pred, data_range=gt.max()))


def ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    """Mean SSIM over slices of a (S, H, W) volume, data range = volume max."""
    if gt.ndim == 2:
        gt, pred = gt[None], pred[None]
    max_val = gt.max()
    values = [structural_similarity(g, p, data_range=max_val) for g, p in zip(gt, pred)]
    return float(np.mean(values))


METRICS = {"NMSE": nmse, "PSNR": psnr, "SSIM": ssim}


class VolumeMetrics:
    """Accumulate slice predictions, then compute metrics per volume."""

    def __init__(self):
        self.preds: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
        self.targets: dict[str, dict[int, np.ndarray]] = defaultdict(dict)

    def add(self, fname: str, slice_idx: int, pred: np.ndarray, target: np.ndarray) -> None:
        self.preds[fname][int(slice_idx)] = pred.astype(np.float64)
        self.targets[fname][int(slice_idx)] = target.astype(np.float64)

    def per_volume(self) -> dict[str, dict[str, float]]:
        out = {}
        for fname in sorted(self.preds):
            idx = sorted(self.preds[fname])
            pred = np.stack([self.preds[fname][i] for i in idx])
            gt = np.stack([self.targets[fname][i] for i in idx])
            out[fname] = {name: fn(gt, pred) for name, fn in METRICS.items()}
        return out

    def summary(self) -> dict[str, tuple[float, float]]:
        """Mean and standard deviation of each metric across volumes."""
        vols = self.per_volume()
        return {
            name: (float(np.mean([v[name] for v in vols.values()])), float(np.std([v[name] for v in vols.values()])))
            for name in METRICS
        }
