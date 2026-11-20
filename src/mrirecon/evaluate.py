"""Evaluate all methods on the same test volumes and masks, and save examples."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import FastMRISliceDataset, load_split
from .masks import CartesianMask
from .metrics import VolumeMetrics
from .reconstruct import recon_tv, recon_unet, recon_unrolled, recon_zero_filled
from .utils import get_device, load_model_from_checkpoint, save_json


def build_methods(cfg: dict, device) -> dict:
    """Return {name: callable(batch) -> (B, 320, 320) tensor} for the requested methods."""
    methods = {"Zero-filled": lambda b: recon_zero_filled(b, device)}
    tv = cfg.get("tv")
    if tv:
        methods["TV-FISTA"] = lambda b: recon_tv(b, device, tv["lam"], tv.get("n_iter", 50), tv.get("n_inner", 10))
    for name, ckpt_path in (cfg.get("checkpoints") or {}).items():
        if not ckpt_path or not Path(ckpt_path).exists():
            print(f"[skip] {name}: checkpoint not found ({ckpt_path})")
            continue
        model, ckpt = load_model_from_checkpoint(ckpt_path, device)
        if ckpt["config"]["model"]["name"] == "unet":
            methods[name] = lambda b, m=model: recon_unet(m, b, device)
        else:
            methods[name] = lambda b, m=model: recon_unrolled(m, b, device)
    return methods


def evaluate(cfg: dict) -> dict:
    device = get_device(cfg.get("device", "auto"))
    d = cfg["data"]
    split = cfg.get("split", "test")
    files = load_split(d["split_file"], split, d["root"])
    ds = FastMRISliceDataset(
        files, CartesianMask.from_config(cfg["mask"]), mode="kspace", deterministic_mask=True,
        skip_edge_slices=d.get("skip_edge_slices", 0), max_volumes=d.get("max_eval_volumes"),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=d.get("num_workers", 2))
    methods = build_methods(cfg, device)
    trackers = {name: VolumeMetrics() for name in methods}
    timings = {name: 0.0 for name in methods}

    out_dir = Path(cfg["output_dir"])
    examples_dir = out_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    example_slices = set(cfg.get("example_slices", []))  # e.g. ["file1000000.h5:18"]
    n_examples = cfg.get("n_examples", 3)
    saved = 0

    print(f"Evaluating {list(methods)} on {len(ds)} slices ({split} split) | device: {device}")
    for batch in tqdm(loader, desc="eval"):
        target = batch["target_raw"].numpy()
        fname, sl = batch["fname"][0], int(batch["slice_idx"][0])
        preds = {}
        for name, fn in methods.items():
            start = time.time()
            pred = fn(batch).cpu().numpy()
            timings[name] += time.time() - start
            trackers[name].add(fname, sl, pred[0], target[0])
            preds[name] = pred[0]

        key = f"{fname}:{sl}"
        is_middle = sl == batch_middle_slice(ds, fname)
        if key in example_slices or (not example_slices and is_middle and saved < n_examples):
            np.savez_compressed(examples_dir / f"{Path(fname).stem}_s{sl}.npz", target=target[0], **preds)
            saved += 1

    results = {}
    for name, tracker in trackers.items():
        summary = tracker.summary()
        results[name] = {
            **{f"{m}_mean": v[0] for m, v in summary.items()},
            **{f"{m}_std": v[1] for m, v in summary.items()},
            "seconds_per_slice": timings[name] / len(ds),
        }
        save_json(tracker.per_volume(), out_dir / f"per_volume_{name}.json")

    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "PSNR", "SSIM", "NMSE", "s/slice"])
        for name, r in results.items():
            w.writerow([name, f"{r['PSNR_mean']:.2f} ± {r['PSNR_std']:.2f}", f"{r['SSIM_mean']:.4f} ± {r['SSIM_std']:.4f}",
                        f"{r['NMSE_mean']:.4f}", f"{r['seconds_per_slice']:.3f}"])
    save_json({"config": cfg, "results": results}, out_dir / "results.json")
    print_table(results)
    return results


_middle_cache: dict[str, int] = {}


def batch_middle_slice(ds: FastMRISliceDataset, fname: str) -> int:
    if fname not in _middle_cache:
        n = next(r.num_slices for r in ds.slices if r.path.name == fname)
        _middle_cache[fname] = n // 2
    return _middle_cache[fname]


def print_table(results: dict) -> None:
    print(f"\n{'Method':<16}{'PSNR (dB)':>16}{'SSIM':>18}{'NMSE':>10}{'s/slice':>10}")
    for name, r in results.items():
        print(f"{name:<16}{r['PSNR_mean']:>9.2f} ± {r['PSNR_std']:<4.2f}{r['SSIM_mean']:>10.4f} ± {r['SSIM_std']:<5.4f}"
              f"{r['NMSE_mean']:>10.4f}{r['seconds_per_slice']:>10.3f}")
