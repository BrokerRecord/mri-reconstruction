import torch

from mrirecon.data import FastMRISliceDataset, load_split
from mrirecon.evaluate import evaluate
from mrirecon.masks import CartesianMask
from mrirecon.models import UNet, UnrolledNet
from mrirecon.train import train


def test_unet_handles_any_size():
    net = UNet(1, 1, chans=8, num_pools=3)
    for h, w in [(320, 320), (100, 70)]:
        assert net(torch.randn(2, 1, h, w)).shape == (2, 1, h, w)


def test_unet_default_size_matches_fastmri():
    n = sum(p.numel() for p in UNet().parameters())
    assert 7.0e6 < n < 8.5e6


def test_unrolled_forward_shape():
    net = UnrolledNet(num_cascades=2, chans=8, num_pools=2)
    y = torch.randn(1, 64, 52, dtype=torch.complex64)
    mask = torch.ones(1, 1, 52)
    out = net(y, mask)
    assert out.shape == (1, 64, 52) and out.is_complex()


def test_dataset_modes(synthetic_data):
    root, split = synthetic_data
    files = load_split(split, "train", root)
    mask = CartesianMask("random", 4, 0.08)
    img = FastMRISliceDataset(files, mask, mode="image")[0]
    assert img["image"].shape == (1, 320, 320) and img["target"].shape == (1, 320, 320)
    ks = FastMRISliceDataset(files, mask, mode="kspace", deterministic_mask=True)[0]
    assert ks["masked_kspace"].is_complex() and ks["target"].shape == (320, 320)


def base_cfg(root, split, out, model):
    return {
        "seed": 0, "device": "cpu", "output_dir": str(out),
        "data": {"root": str(root), "split_file": str(split), "num_workers": 0},
        "mask": {"kind": "random", "acceleration": 4, "center_fraction": 0.08},
        "model": model,
        "train": {"epochs": 1, "batch_size": 2, "lr": 1e-3},
    }


def test_end_to_end_train_and_evaluate(synthetic_data, tmp_path):
    root, split = synthetic_data
    unet_ckpt = train(base_cfg(root, split, tmp_path / "unet", {"name": "unet", "chans": 8, "num_pools": 2}))
    unr_ckpt = train(base_cfg(root, split, tmp_path / "unr",
                              {"name": "unrolled", "num_cascades": 2, "chans": 8, "num_pools": 2}))
    cfg = base_cfg(root, split, tmp_path / "eval", None)
    cfg.update({"tv": {"lam": 0.005, "n_iter": 5, "n_inner": 3},
                "checkpoints": {"U-Net": str(unet_ckpt), "Unrolled": str(unr_ckpt)}})
    results = evaluate(cfg)
    assert set(results) == {"Zero-filled", "TV-FISTA", "U-Net", "Unrolled"}
    assert all(r["PSNR_mean"] > 0 for r in results.values())
    assert (tmp_path / "eval" / "results.csv").exists()


def test_resume_continues_training_and_history(synthetic_data, tmp_path):
    import csv

    root, split = synthetic_data
    out = tmp_path / "unr"
    model = {"name": "unrolled", "num_cascades": 1, "chans": 8, "num_pools": 2}
    cfg = base_cfg(root, split, out, model)
    cfg["train"].update({"epochs": 1, "lr_step": 1, "lr_gamma": 0.5})
    train(cfg)
    cfg["train"]["epochs"] = 2
    train(cfg, resume=True)
    with open(out / "history.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert float(rows[1]["lr"]) == float(rows[0]["lr"]) / 2  # schedule continued, not restarted
    ckpt = torch.load(out / "last.pt", weights_only=False)
    assert ckpt["epoch"] == 2 and ckpt["scheduler"] is not None
