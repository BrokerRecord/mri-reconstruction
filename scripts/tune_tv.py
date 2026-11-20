"""Choose the TV regularisation weight lambda on the *validation* set.

Tuning on validation (never on test) keeps the final comparison fair. The script
runs TV-FISTA for each lambda on a few validation volumes and reports PSNR / SSIM.

Example:
    python scripts/tune_tv.py --config configs/evaluate.yaml --lams 0.001 0.003 0.005 0.01 0.02
"""

import argparse
import sys
import csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


from torch.utils.data import DataLoader
from tqdm import tqdm

from mrirecon.data import FastMRISliceDataset, load_split
from mrirecon.masks import CartesianMask
from mrirecon.metrics import VolumeMetrics
from mrirecon.reconstruct import recon_tv
from mrirecon.utils import apply_overrides, get_device, load_config


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--lams", type=float, nargs="+", default=[0.001, 0.003, 0.005, 0.01, 0.02])
    p.add_argument("--volumes", type=int, default=3, help="Number of validation volumes to use")
    p.add_argument("--n-iter", type=int, default=50)
    p.add_argument("--out", default="results/tv_tuning.csv")
    p.add_argument("--set", nargs="*", default=[], help="Config overrides, e.g. data.root=/path")
    args = p.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    d = cfg["data"]
    device = get_device(cfg.get("device", "auto"))
    ds = FastMRISliceDataset(
        load_split(d["split_file"], "val", d["root"]), CartesianMask.from_config(cfg["mask"]),
        mode="kspace", deterministic_mask=True, skip_edge_slices=d.get("skip_edge_slices", 0), max_volumes=args.volumes,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    rows = []
    for lam in args.lams:
        tracker = VolumeMetrics()
        for batch in tqdm(loader, desc=f"lambda={lam}", leave=False):
            pred = recon_tv(batch, device, lam, args.n_iter).cpu().numpy()
            tracker.add(batch["fname"][0], int(batch["slice_idx"][0]), pred[0], batch["target_raw"][0].numpy())
        s = tracker.summary()
        rows.append((lam, s["PSNR"][0], s["SSIM"][0]))
        print(f"lambda={lam:<8g} PSNR={s['PSNR'][0]:.2f} dB  SSIM={s['SSIM'][0]:.4f}")

    best = max(rows, key=lambda r: r[2])
    print(f"\nBest lambda (by SSIM): {best[0]} -> set tv.lam in configs/evaluate.yaml")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lambda", "PSNR", "SSIM"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
