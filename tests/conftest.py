import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def synthetic_data(tmp_path_factory):
    """A few small synthetic volumes plus a splits.json."""
    root = tmp_path_factory.mktemp("data")
    subprocess.run([sys.executable, str(ROOT / "scripts/make_synthetic_data.py"), "--out", str(root),
                    "--volumes", "6", "--slices", "3"], check=True)
    split = root / "splits.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/split_data.py"), "--data-root", str(root),
                    "--out", str(split), "--val", "0.2", "--test", "0.2"], check=True)
    return root, split
