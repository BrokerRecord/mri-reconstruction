"""Split fastMRI volumes into train / val / test sets *by volume* and write splits.json.

Splitting by volume (not by slice) guarantees that no patient appears in two sets,
which would otherwise leak information and inflate test scores.

Example:
    python scripts/split_data.py --data-root data/singlecoil_val --out data/splits.json
"""

import argparse
import json
import random
from pathlib import Path

import h5py


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, help="Folder containing the .h5 volumes (searched recursively)")
    p.add_argument("--out", default="data/splits.json")
    p.add_argument("--val", type=float, default=0.15, help="Fraction of volumes for validation")
    p.add_argument("--test", type=float, default=0.15, help="Fraction of volumes for testing")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    root = Path(args.data_root)
    files, skipped = [], []
    for f in sorted(root.rglob("*.h5")):
        try:
            with h5py.File(f, "r") as hf:
                if "kspace" in hf and "reconstruction_esc" in hf:
                    files.append(f.relative_to(root).as_posix())
                else:
                    skipped.append(f.name)
        except OSError:
            skipped.append(f.name)  # incomplete download / corrupted file
    if not files:
        raise SystemExit(f"No valid fastMRI single-coil volumes found in {root}")

    random.Random(args.seed).shuffle(files)
    n = len(files)
    n_test = max(1, round(n * args.test))
    n_val = max(1, round(n * args.val))
    splits = {
        "test": sorted(files[:n_test]),
        "val": sorted(files[n_test : n_test + n_val]),
        "train": sorted(files[n_test + n_val :]),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({**splits, "seed": args.seed, "data_root": str(root)}, f, indent=2)

    print(f"{n} volumes -> train {len(splits['train'])}, val {len(splits['val'])}, test {len(splits['test'])}")
    if skipped:
        print(f"Skipped {len(skipped)} files without kspace/reconstruction_esc or unreadable: {skipped[:5]}...")
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
