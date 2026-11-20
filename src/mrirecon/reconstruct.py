"""One interface for every method: k-space batch in, 320 x 320 magnitude images out.

All functions take a batch produced by ``FastMRISliceDataset(mode="kspace")`` and
return the reconstruction in the original intensity units (same scale as the
ground truth), as a tensor of shape (B, 320, 320). This makes the comparison
between classical and learned methods strictly identical.
"""

from __future__ import annotations

import torch

from .classical import magnitude_crop, tv_fista, zero_filled
from .data import TARGET_SIZE


def _rescale(img: torch.Tensor, batch: dict) -> torch.Tensor:
    return img * batch["scale"].to(img.device).view(-1, 1, 1)


@torch.no_grad()
def recon_zero_filled(batch: dict, device: torch.device) -> torch.Tensor:
    y = batch["masked_kspace"].to(device)
    return _rescale(magnitude_crop(zero_filled(y), TARGET_SIZE), batch)


@torch.no_grad()
def recon_tv(batch: dict, device: torch.device, lam: float, n_iter: int = 50, n_inner: int = 10) -> torch.Tensor:
    y = batch["masked_kspace"].to(device)
    mask = batch["mask"].to(device)
    x = tv_fista(y, mask, lam=lam, n_iter=n_iter, n_inner=n_inner)
    return _rescale(magnitude_crop(x, TARGET_SIZE), batch)


def unet_input(masked_kspace: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalised U-Net input from k-space (identical to the training transform)."""
    zf = magnitude_crop(zero_filled(masked_kspace), TARGET_SIZE)  # (B, 320, 320)
    mean = zf.mean(dim=(-2, -1), keepdim=True)
    std = zf.std(dim=(-2, -1), keepdim=True) + 1e-11
    return ((zf - mean) / std).clamp(-6, 6).unsqueeze(1), mean, std


@torch.no_grad()
def recon_unet(model: torch.nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
    y = batch["masked_kspace"].to(device)
    inp, mean, std = unet_input(y)
    out = model(inp).squeeze(1) * std + mean
    return _rescale(out.clamp(min=0), batch)


@torch.no_grad()
def recon_unrolled(model: torch.nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
    y = batch["masked_kspace"].to(device)
    mask = batch["mask"].to(device)
    return _rescale(magnitude_crop(model(y, mask), TARGET_SIZE), batch)
