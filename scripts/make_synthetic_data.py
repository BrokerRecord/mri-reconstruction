"""Create small synthetic volumes in the fastMRI single-coil format.

Useful to check that the whole pipeline runs (tests, debugging) without the real
data. The images are random ellipse phantoms, not anatomy: results on them say
nothing about real MRI performance.

Example:
    python scripts/make_synthetic_data.py --out data/synthetic --volumes 8
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def phantom(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[-1 : 1 : h * 1j, -1 : 1 : w * 1j]
    img = np.zeros((h, w))
    for _ in range(rng.integers(6, 12)):
        cy, cx = rng.uniform(-0.5, 0.5, 2)
        ay, ax = rng.uniform(0.05, 0.45, 2)
        theta = rng.uniform(0, np.pi)
        c, s = np.cos(theta), np.sin(theta)
        u = ((xx - cx) * c + (yy - cy) * s) / ax
        v = (-(xx - cx) * s + (yy - cy) * c) / ay
        img[u**2 + v**2 <= 1] += rng.uniform(0.2, 1.0)
    return img


def make_volume(path: Path, n_slices: int, width: int, rng: np.random.Generator) -> None:
    h = 640
    base = phantom(320, 320, rng)
    kspace = np.zeros((n_slices, h, width), dtype=np.complex64)
    target = np.zeros((n_slices, 320, 320), dtype=np.float32)
    for s in range(n_slices):
        img = np.zeros((h, width), dtype=np.complex128)
        sl = base * (0.8 + 0.4 * np.sin(np.pi * s / n_slices)) + 0.02 * rng.standard_normal(base.shape)
        phase = np.exp(1j * 0.3 * rng.standard_normal())
        top, left = (h - 320) // 2, (width - 320) // 2
        img[top : top + 320, left : left + 320] = sl * phase
        k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img), norm="ortho")) * 1e-4
        kspace[s] = k.astype(np.complex64)
        rec = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k), norm="ortho"))
        target[s] = np.abs(rec[top : top + 320, left : left + 320])
    with h5py.File(path, "w") as hf:
        hf.create_dataset("kspace", data=kspace)
        hf.create_dataset("reconstruction_esc", data=target)
        hf.attrs["max"] = float(target.max())
        hf.attrs["acquisition"] = "SYNTHETIC"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--volumes", type=int, default=8)
    p.add_argument("--slices", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for i in range(args.volumes):
        width = int(rng.choice([368, 372]))  # real fastMRI widths vary between volumes
        make_volume(out / f"synthetic{i:04d}.h5", args.slices, width, rng)
    print(f"Wrote {args.volumes} synthetic volumes to {out}")


if __name__ == "__main__":
    main()
