"""Compare zero-filling, TV-FISTA and trained networks on the test set.

Example:
    python scripts/evaluate.py --config configs/evaluate.yaml
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


from mrirecon.evaluate import evaluate
from mrirecon.utils import apply_overrides, load_config


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[])
    args = p.parse_args()
    evaluate(apply_overrides(load_config(args.config), args.set))


if __name__ == "__main__":
    main()
