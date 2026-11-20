import numpy as np
import torch


from mrirecon.fft import adjoint_op, center_crop, fft2c, forward_op, ifft2c
from mrirecon.masks import CartesianMask


def rand_complex(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def test_fft_roundtrip_and_unitary():
    x = rand_complex(2, 64, 48)
    assert torch.allclose(ifft2c(fft2c(x)), x, atol=1e-5)
    # Parseval: orthonormal transform preserves the energy
    assert torch.allclose(fft2c(x).abs().pow(2).sum(), x.abs().pow(2).sum(), rtol=1e-4)


def test_forward_adjoint_dot_product():
    """<A x, y> == <x, A^H y> for the masked Fourier operator."""
    x, y = rand_complex(32, 40), rand_complex(32, 40)
    mask = CartesianMask("random", 4, 0.08)((32, 40), seed=0)
    lhs = torch.vdot(forward_op(x, mask).flatten(), y.flatten())
    rhs = torch.vdot(x.flatten(), adjoint_op(y, mask).flatten())
    assert torch.allclose(lhs, rhs, atol=1e-3)


def test_center_crop_shape():
    assert center_crop(torch.zeros(1, 640, 368), (320, 320)).shape == (1, 320, 320)


def test_mask_acceleration_and_center():
    for kind in ["random", "equispaced"]:
        m = CartesianMask(kind, acceleration=4, center_fraction=0.08)
        cols = m.column_mask(368, seed=1)
        assert abs(cols.mean() - 0.25) < 0.06
        c = 368 // 2
        assert cols[c - 10 : c + 10].all(), "low frequencies must be fully sampled"


def test_mask_is_deterministic_with_seed():
    m = CartesianMask("random", 8, 0.04)
    assert np.array_equal(m.column_mask(368, seed=5), m.column_mask(368, seed=5))
