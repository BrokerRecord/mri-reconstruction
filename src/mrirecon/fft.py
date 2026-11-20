"""Centred, orthonormal 2D Fourier transforms and the single-coil MRI forward model.

All functions work on complex tensors and transform the last two dimensions.
With ``norm="ortho"`` the transforms are unitary, so ``ifft2c(fft2c(x)) == x`` and
the adjoint of ``fft2c`` is ``ifft2c``. This matters for the optimisation methods,
whose step sizes rely on the operator norm being 1.
"""

from __future__ import annotations

import torch


def fft2c(image: torch.Tensor) -> torch.Tensor:
    """Image (complex, ..., H, W) -> centred k-space (complex, ..., H, W)."""
    x = torch.fft.ifftshift(image, dim=(-2, -1))
    x = torch.fft.fft2(x, norm="ortho")
    return torch.fft.fftshift(x, dim=(-2, -1))


def ifft2c(kspace: torch.Tensor) -> torch.Tensor:
    """Centred k-space (complex, ..., H, W) -> image (complex, ..., H, W)."""
    x = torch.fft.ifftshift(kspace, dim=(-2, -1))
    x = torch.fft.ifft2(x, norm="ortho")
    return torch.fft.fftshift(x, dim=(-2, -1))


def forward_op(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Undersampled single-coil acquisition A x = M F x."""
    return mask * fft2c(image)


def adjoint_op(kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Adjoint A^H y = F^H M y (M is a 0/1 mask, so M^H = M)."""
    return ifft2c(mask * kspace)


def center_crop(x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Crop the last two dimensions of ``x`` around the centre to ``shape``."""
    h, w = x.shape[-2:]
    ch, cw = shape
    if ch > h or cw > w:
        raise ValueError(f"Crop {shape} larger than input {(h, w)}")
    top = (h - ch) // 2
    left = (w - cw) // 2
    return x[..., top : top + ch, left : left + cw]


def complex_to_channels(x: torch.Tensor) -> torch.Tensor:
    """Complex (B, H, W) -> real (B, 2, H, W) with real and imaginary channels."""
    return torch.stack((x.real, x.imag), dim=1)


def channels_to_complex(x: torch.Tensor) -> torch.Tensor:
    """Real (B, 2, H, W) -> complex (B, H, W)."""
    return torch.complex(x[:, 0], x[:, 1])
