"""U-Net (Ronneberger et al., 2015) in the configuration used by the fastMRI baseline.

With ``chans=32`` and ``num_pools=4`` the model has about 7.8 M parameters.
Inputs whose height or width is not a multiple of 2**num_pools are padded and
the output is cropped back, so the network accepts any image size.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans, 2, stride=2, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, in_chans: int = 1, out_chans: int = 1, chans: int = 32, num_pools: int = 4, drop_prob: float = 0.0):
        super().__init__()
        self.num_pools = num_pools
        self.down = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pools - 1):
            self.down.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.bottleneck = ConvBlock(ch, ch * 2, drop_prob)

        self.up_trans = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        for _ in range(num_pools):
            self.up_trans.append(UpBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2
        self.head = nn.Conv2d(chans, out_chans, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        m = 2**self.num_pools
        pad_h, pad_w = (-h) % m, (-w) % m
        if pad_h or pad_w:
            x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode="reflect")

        skips = []
        out = x
        for block in self.down:
            out = block(out)
            skips.append(out)
            out = F.avg_pool2d(out, 2)
        out = self.bottleneck(out)
        for trans, conv in zip(self.up_trans, self.up_conv):
            out = trans(out)
            out = conv(torch.cat([out, skips.pop()], dim=1))
        out = self.head(out)

        if pad_h or pad_w:
            out = out[..., pad_h // 2 : pad_h // 2 + h, pad_w // 2 : pad_w // 2 + w]
        return out
