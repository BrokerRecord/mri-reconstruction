"""Create the figures for the README from the evaluation and training outputs.

Example:
    python scripts/make_figures.py --results results/test --runs runs/unet runs/unrolled
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def comparison_figure(npz_path: Path, out_path: Path) -> None:
    data = dict(np.load(npz_path))
    target = data.pop("target")
    methods = list(data)
    vmax = np.percentile(target, 99.5)
    fig, axes = plt.subplots(2, len(methods) + 1, figsize=(3 * (len(methods) + 1), 6.9))
    axes[0, 0].imshow(target, cmap="gray", vmin=0, vmax=vmax)
    axes[0, 0].set_title("Ground truth")
    axes[1, 0].axis("off")
    err_max = 0.25 * vmax
    for j, name in enumerate(methods, start=1):
        pred = data[name]
        psnr = peak_signal_noise_ratio(target, pred, data_range=target.max())
        ssim = structural_similarity(target, pred, data_range=target.max())
        axes[0, j].imshow(pred, cmap="gray", vmin=0, vmax=vmax)
        axes[0, j].set_title(f"{name}\n{psnr:.2f} dB / SSIM {ssim:.3f}", fontsize=10)
        axes[1, j].imshow(np.abs(pred - target), cmap="inferno", vmin=0, vmax=err_max)
        axes[1, j].set_title("|error|", fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def learning_curves(run_dirs: list[Path], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for run in run_dirs:
        hist = run / "history.csv"
        if not hist.exists():
            continue
        with open(hist) as f:
            rows = list(csv.DictReader(f))
        ep = [int(r["epoch"]) for r in rows]
        axes[0].plot(ep, [float(r["train_loss"]) for r in rows], label=run.name)
        axes[1].plot(ep, [float(r["val_PSNR"]) for r in rows], label=run.name)
        axes[2].plot(ep, [float(r["val_SSIM"]) for r in rows], label=run.name)
    for ax, title in zip(axes, ["Training L1 loss", "Validation PSNR (dB)", "Validation SSIM"]):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def tv_tuning_figure(csv_path: Path, out_path: Path) -> None:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    lams = [float(r["lambda"]) for r in rows]
    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    ax1.semilogx(lams, [float(r["PSNR"]) for r in rows], "o-", color="tab:blue")
    ax1.set_xlabel("TV weight λ")
    ax1.set_ylabel("PSNR (dB)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.semilogx(lams, [float(r["SSIM"]) for r in rows], "s--", color="tab:orange")
    ax2.set_ylabel("SSIM", color="tab:orange")
    ax1.set_title("TV-FISTA: choice of λ on validation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results/test")
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--tv-tuning", default="results/tv_tuning.csv")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    examples = sorted((Path(args.results) / "examples").glob("*.npz"))
    for i, ex in enumerate(examples):
        comparison_figure(ex, out / f"comparison_{i}.png")
    if args.runs:
        learning_curves([Path(r) for r in args.runs], out / "learning_curves.png")
    if Path(args.tv_tuning).exists():
        tv_tuning_figure(Path(args.tv_tuning), out / "tv_tuning.png")
    print(f"Figures written to {out}/")


if __name__ == "__main__":
    main()
