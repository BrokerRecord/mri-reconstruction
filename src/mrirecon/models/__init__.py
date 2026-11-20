from .unet import UNet
from .unrolled import UnrolledNet


def build_model(cfg: dict):
    """Build a model from the ``model`` section of a config file."""
    cfg = dict(cfg)
    name = cfg.pop("name")
    if name == "unet":
        return UNet(**cfg)
    if name == "unrolled":
        return UnrolledNet(**cfg)
    raise ValueError(f"Unknown model: {name}")


__all__ = ["UNet", "UnrolledNet", "build_model"]
