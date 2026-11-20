import torch

from mrirecon.classical import grad, grad_adjoint, prox_tv, tv_fista, tv_norm, zero_filled
from mrirecon.fft import fft2c
from mrirecon.masks import CartesianMask


def test_gradient_adjoint():
    """<grad x, p> == <x, grad^T p>, required for the FGP dual algorithm."""
    x = torch.randn(24, 30, dtype=torch.complex64)
    ph, pv = torch.randn(24, 30, dtype=torch.complex64), torch.randn(24, 30, dtype=torch.complex64)
    dh, dv = grad(x)
    lhs = torch.vdot(dh.flatten(), ph.flatten()) + torch.vdot(dv.flatten(), pv.flatten())
    rhs = torch.vdot(x.flatten(), grad_adjoint(ph, pv).flatten())
    assert torch.allclose(lhs, rhs, atol=1e-3)


def piecewise_constant(h=64, w=64):
    img = torch.zeros(h, w)
    img[16:48, 16:48] = 1.0
    img[24:40, 28:36] = 0.5
    return img.to(torch.complex64)


def test_prox_tv_denoises():
    torch.manual_seed(0)
    clean = piecewise_constant()
    noisy = clean + 0.1 * torch.randn_like(clean)
    den, _ = prox_tv(noisy, lam=0.1, n_iter=100)
    assert (den - clean).abs().norm() < 0.6 * (noisy - clean).abs().norm()
    assert tv_norm(den) < tv_norm(noisy)


def test_tv_fista_beats_zero_filled_and_decreases_objective():
    clean = piecewise_constant(64, 64)
    mask = CartesianMask("random", 4, 0.1)((64, 64), seed=3)
    y = mask * fft2c(clean)
    zf = zero_filled(y)
    rec, hist = tv_fista(y, mask, lam=0.01, n_iter=60, n_inner=10, return_history=True)
    assert (rec - clean).abs().norm() < (zf - clean).abs().norm()
    assert hist[-1] < hist[0]
