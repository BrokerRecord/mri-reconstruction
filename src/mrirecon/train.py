"""Training loop for the U-Net and the unrolled network."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .classical import magnitude_crop
from .data import TARGET_SIZE, FastMRISliceDataset, load_split
from .masks import CartesianMask
from .metrics import VolumeMetrics
from .models import build_model
from .reconstruct import recon_unet, recon_unrolled
from .utils import count_parameters, get_device, save_checkpoint, save_json, set_seed


def make_loaders(cfg: dict, model_name: str):
    d = cfg["data"]
    mask = CartesianMask.from_config(cfg["mask"])
    train_files = load_split(d["split_file"], "train", d["root"])
    val_files = load_split(d["split_file"], "val", d["root"])

    train_mode = "image" if model_name == "unet" else "kspace"
    train_bs = cfg["train"]["batch_size"] if model_name == "unet" else 1
    train_ds = FastMRISliceDataset(
        train_files, mask, mode=train_mode, deterministic_mask=False,
        skip_edge_slices=d.get("skip_edge_slices", 0), max_volumes=d.get("max_train_volumes"),
    )
    val_ds = FastMRISliceDataset(
        val_files, mask, mode="kspace", deterministic_mask=True,
        skip_edge_slices=d.get("skip_edge_slices", 0), max_volumes=d.get("max_val_volumes"),
    )
    nw = d.get("num_workers", 2)
    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=nw, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=nw)
    return train_loader, val_loader


def training_step(model, batch, model_name: str, device) -> torch.Tensor:
    if model_name == "unet":
        out = model(batch["image"].to(device))
        return F.l1_loss(out, batch["target"].to(device))
    y = batch["masked_kspace"].to(device)
    mask = batch["mask"].to(device)
    pred = magnitude_crop(model(y, mask), TARGET_SIZE)
    return F.l1_loss(pred, batch["target"].to(device))


@torch.no_grad()
def validate(model, loader, model_name: str, device) -> dict:
    model.eval()
    metrics = VolumeMetrics()
    recon = recon_unet if model_name == "unet" else recon_unrolled
    for batch in tqdm(loader, desc="val", leave=False):
        pred = recon(model, batch, device).cpu().numpy()
        target = batch["target_raw"].numpy()
        for i in range(pred.shape[0]):
            metrics.add(batch["fname"][i], int(batch["slice_idx"][i]), pred[i], target[i])
    return {k: v[0] for k, v in metrics.summary().items()}


def _resume(out_dir: Path, model, optimizer, t: dict, device) -> tuple[int, float, object]:
    """Restore model, optimizer and LR schedule from ``last.pt``.

    Returns (last completed epoch, best validation SSIM so far, scheduler).
    The best score is read from ``best.pt`` because it is always up to date there.
    """
    ckpt = torch.load(out_dir / "last.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    done = int(ckpt["epoch"])
    if ckpt.get("scheduler"):
        scheduler = _make_scheduler(optimizer, t)
        scheduler.load_state_dict(ckpt["scheduler"])
    else:  # checkpoints written before the scheduler was saved
        scheduler = _make_scheduler(optimizer, t, last_epoch=done - 1)
    best_path = out_dir / "best.pt"
    best = ckpt.get("best_metric", -1.0)
    if best_path.exists():
        best = torch.load(best_path, map_location="cpu", weights_only=False)["best_metric"]
    return done, float(best), scheduler


def _make_scheduler(optimizer, t: dict, last_epoch: int = -1):
    return torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=t.get("lr_step", 40), gamma=t.get("lr_gamma", 0.1), last_epoch=last_epoch
    )


def train(cfg: dict, resume: bool = False) -> Path:
    """Train a model. With ``resume=True``, continue from ``output_dir/last.pt``
    up to ``train.epochs`` (the total number of epochs, not additional ones)."""
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))
    model_name = cfg["model"]["name"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = make_loaders(cfg, model_name)
    model = build_model(cfg["model"]).to(device)
    t = cfg["train"]
    optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 0.0))

    log_path = out_dir / "history.csv"
    if resume:
        if not (out_dir / "last.pt").exists():
            raise FileNotFoundError(f"Cannot resume: {out_dir / 'last.pt'} not found")
        start_epoch, best_ssim, scheduler = _resume(out_dir, model, optimizer, t, device)
        start_epoch += 1
        print(f"Resuming from epoch {start_epoch - 1} (best val SSIM so far {best_ssim:.4f}, "
              f"lr {optimizer.param_groups[0]['lr']:g})")
    else:
        scheduler = _make_scheduler(optimizer, t)
        start_epoch, best_ssim = 1, -1.0
        save_json(cfg, out_dir / "config.json")
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_NMSE", "val_PSNR", "val_SSIM", "lr", "seconds"])

    if start_epoch > t["epochs"]:
        print(f"Already trained for {start_epoch - 1} epochs; increase train.epochs to continue.")
        return out_dir / "best.pt"

    print(f"Model: {model_name} | parameters: {count_parameters(model):,} | device: {device}")
    print(f"Train slices: {len(train_loader.dataset)} | val slices: {len(val_loader.dataset)}")

    for epoch in range(start_epoch, t["epochs"] + 1):
        model.train()
        start = time.time()
        running, n = 0.0, 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{t['epochs']}", leave=False):
            loss = training_step(model, batch, model_name, device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t.get("grad_clip", 1.0))
            optimizer.step()
            running += loss.item()
            n += 1
        scheduler.step()

        val = validate(model, val_loader, model_name, device)
        secs = time.time() - start
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:3d} | loss {running / n:.4f} | val PSNR {val['PSNR']:.2f} dB "
            f"| SSIM {val['SSIM']:.4f} | NMSE {val['NMSE']:.4f} | {secs:.0f}s"
        )
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, running / n, val["NMSE"], val["PSNR"], val["SSIM"], lr, round(secs, 1)])

        if val["SSIM"] > best_ssim:
            best_ssim = val["SSIM"]
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, best_ssim, cfg, scheduler)
        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, best_ssim, cfg, scheduler)

    print(f"Best validation SSIM: {best_ssim:.4f} -> {out_dir / 'best.pt'}")
    return out_dir / "best.pt"
