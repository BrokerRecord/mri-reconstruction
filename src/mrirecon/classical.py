"""Model-based reconstruction: zero-filling and total-variation regularisation with FISTA.

We solve the compressed-sensing problem

    min_x  1/2 || M F x - y ||_2^2  +  lambda * TV(x)

where x is the complex image, F the unitary 2D Fourier transform, M the binary
sampling mask, y the measured k-space and TV the isotropic total variation.

* The data term is smooth with Lipschitz constant L = ||M F||^2 = 1, so the
  gradient step size is 1.
* The proximal operator of lambda * TV (a TV denoising problem) has no closed
  form; we compute it with the Fast Gradient Projection (FGP) algorithm of
  Beck & Teboulle (2009) on the dual problem, warm-started across outer iterations.
"""

from __future__ import annotations

import math

import torch

from .fft import adjoint_op, center_crop, forward_op, ifft2c


# --------------------------------------------------------------------------- #
# Discrete gradient (forward differences, Neumann boundary) and its adjoint
# --------------------------------------------------------------------------- #
def grad(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward differences along W (horizontal) and H (vertical)."""
    dh = torch.zeros_like(x)
    dv = torch.zeros_like(x)
    dh[..., :, :-1] = x[..., :, 1:] - x[..., :, :-1]
    dv[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    return dh, dv


def grad_adjoint(ph: torch.Tensor, pv: torch.Tensor) -> torch.Tensor:
    """Adjoint of :func:`grad` (equal to minus the discrete divergence)."""
    ph = ph.clone()
    pv = pv.clone()
    ph[..., :, -1] = 0
    pv[..., -1, :] = 0
    out = -ph - pv
    out[..., :, 1:] += ph[..., :, :-1]
    out[..., 1:, :] += pv[..., :-1, :]
    return out


def tv_norm(x: torch.Tensor) -> torch.Tensor:
    """Isotropic total variation, summed over the last two dimensions."""
    dh, dv = grad(x)
    return torch.sqrt(dh.abs() ** 2 + dv.abs() ** 2).sum(dim=(-2, -1))


def _project_unit_ball(ph: torch.Tensor, pv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    norm = torch.sqrt(ph.abs() ** 2 + pv.abs() ** 2).clamp(min=1.0)
    return ph / norm, pv / norm


def prox_tv(
    b: torch.Tensor,
    lam: float,
    n_iter: int = 20,
    dual: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Solve min_x 1/2 ||x - b||^2 + lam * TV(x) with FGP.

    Returns the solution and the final dual variables (for warm starts).
    Works for real or complex ``b``.
    """
    if lam <= 0:
        return b, dual if dual is not None else (torch.zeros_like(b), torch.zeros_like(b))
    if dual is None:
        ph, pv = torch.zeros_like(b), torch.zeros_like(b)
    else:
        ph, pv = dual
    rh, rv = ph, pv
    t = 1.0
    step = 1.0 / (8.0 * lam)  # ||grad||^2 <= 8 in 2D
    for _ in range(n_iter):
        x = b - lam * grad_adjoint(rh, rv)
        gh, gv = grad(x)
        ph_new, pv_new = _project_unit_ball(rh + step * gh, rv + step * gv)
        t_new = (1 + math.sqrt(1 + 4 * t * t)) / 2
        beta = (t - 1) / t_new
        rh = ph_new + beta * (ph_new - ph)
        rv = pv_new + beta * (pv_new - pv)
        ph, pv, t = ph_new, pv_new, t_new
    x = b - lam * grad_adjoint(ph, pv)
    return x, (ph, pv)


# --------------------------------------------------------------------------- #
# Reconstructions
# --------------------------------------------------------------------------- #
def zero_filled(masked_kspace: torch.Tensor) -> torch.Tensor:
    """Inverse FFT of the zero-filled k-space (complex image, full size)."""
    return ifft2c(masked_kspace)


def data_fidelity(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return 0.5 * (forward_op(x, mask) - y).abs().pow(2).sum(dim=(-2, -1))


def tv_fista(
    masked_kspace: torch.Tensor,
    mask: torch.Tensor,
    lam: float = 0.005,
    n_iter: int = 50,
    n_inner: int = 10,
    return_history: bool = False,
):
    """TV-regularised reconstruction with FISTA.

    Args:
        masked_kspace: complex (..., H, W) undersampled k-space y.
        mask: (..., 1, W) or (..., H, W) binary mask.
        lam: regularisation weight (data are scaled so that the zero-filled
            image has maximum 1; values between 1e-3 and 2e-2 are typical).
        n_iter: outer FISTA iterations.
        n_inner: FGP iterations for each TV proximal step.
        return_history: also return the objective value at every iteration.
    """
    y = masked_kspace
    x = adjoint_op(y, mask)
    z = x
    t = 1.0
    dual = None
    history = []
    for _ in range(n_iter):
        g = z - adjoint_op(forward_op(z, mask) - y, mask)  # gradient step, L = 1
        x_new, dual = prox_tv(g, lam, n_inner, dual)
        t_new = (1 + math.sqrt(1 + 4 * t * t)) / 2
        z = x_new + ((t - 1) / t_new) * (x_new - x)
        x, t = x_new, t_new
        if return_history:
            history.append(float((data_fidelity(x, y, mask) + lam * tv_norm(x)).sum()))
    return (x, history) if return_history else x


def magnitude_crop(x: torch.Tensor, size=(320, 320)) -> torch.Tensor:
    """Magnitude of a complex image, centre-cropped to the fastMRI target size."""
    return center_crop(x.abs(), size)
