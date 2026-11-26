"""Train a U-Net or an unrolled network.

Example:
    python scripts/train.py --config configs/unet.yaml
    python scripts/train.py --config configs/unrolled.yaml --set train.epochs=5 data.max_train_volumes=10
    python scripts/train.py --config configs/unrolled.yaml --resume --set train.epochs=10
"""

import argparse

from mrirecon.train import train
from mrirecon.utils import apply_overrides, load_config


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], help="Overrides, e.g. train.epochs=10")
    p.add_argument("--resume", action="store_true",
                   help="Continue from output_dir/last.pt up to train.epochs (total, not extra)")
    args = p.parse_args()
    train(apply_overrides(load_config(args.config), args.set), resume=args.resume)


if __name__ == "__main__":
    main()
