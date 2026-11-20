"""Unrolled proximal-gradient network for single-coil MRI reconstruction.

The network unrolls K iterations of a gradient method on the same objective as
TV-FISTA, but replaces the hand-crafted TV prior with a small learned CNN:

    x_{k+1} = x_k - eta_k * A^H (A x_k - y) - R_k(x_k)

* ``A^H (A x_k - y)`` is the exact gradient of the data-fidelity term, so the
  physics of the acquisition (FFT + mask) is built into the network;
* ``eta_k`` are learned step sizes;
* ``R_k`` is a small U-Net acting on the real and imaginary channels.

This "physics-informed" design (in the spirit of variational networks and MoDL)
typically needs far fewer parameters than an image-domain U-Net for the same quality.
"""

from __future__ import annotations

import torch
from torch import nn

from ..fft import adjoint_op, channels_to_complex, complex_to_channels, forward_op
from .unet import UNet


class NormUNet(nn.Module):
    """U-Net on (real, imag) channels with per-sample normalisation for stability."""

    def __init__(self, chans: int = 16, num_pools: int = 3):
        super().__init__()
        self.unet = UNet(in_chans=2, out_chans=2, chans=chans, num_pools=num_pools)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = complex_to_channels(x)  # (B, 2, H, W)
        mean = c.mean(dim=(-2, -1), keepdim=True)
        std = c.std(dim=(-2, -1), keepdim=True) + 1e-8
        out = self.unet((c - mean) / std) * std + mean
        return channels_to_complex(out)


class UnrolledNet(nn.Module):
    def __init__(self, num_cascades: int = 6, chans: int = 16, num_pools: int = 3, share_weights: bool = False):
        super().__init__()
        self.num_cascades = num_cascades
        n_reg = 1 if share_weights else num_cascades
        self.regularisers = nn.ModuleList([NormUNet(chans, num_pools) for _ in range(n_reg)])
        self.step_sizes = nn.Parameter(torch.ones(num_cascades))

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Args: masked_kspace complex (B, H, W); mask (B, 1, W). Returns complex image (B, H, W)."""
        x = adjoint_op(masked_kspace, mask)
        for k in range(self.num_cascades):
            reg = self.regularisers[k % len(self.regularisers)]
            dc_grad = adjoint_op(forward_op(x, mask) - masked_kspace, mask)
            x = x - self.step_sizes[k] * dc_grad - reg(x)
        return x
