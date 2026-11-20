"""Train a U-Net or an unrolled network.

Example:
    python scripts/train.py --config configs/unet.yaml
    python scripts/train.py --config configs/unrolled.yaml --set train.epochs=5 data.max_train_volumes=10
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


from mrirecon.train import train
from mrirecon.utils import apply_overrides, load_config


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], help="Overrides, e.g. train.epochs=10")
    args = p.parse_args()
    train(apply_overrides(load_config(args.config), args.set))


if __name__ == "__main__":
    main()
