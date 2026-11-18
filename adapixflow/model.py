"""The dynamics operator G_theta (Sec. 3.2): a 3D U-Net that maps a context of fused states to velocity node fields."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import CHANNELS

TIME_UNIT = 10.0  # the spacing s enters the network as s / TIME_UNIT


class ChannelLN(nn.Module):
    """LayerNorm over the channels of each voxel. Unlike GroupNorm, its statistics do not depend on the window size, so a
    model trained on crops applies unchanged to whole scenes."""

    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.movedim(1, -1)).movedim(-1, 1)


def _block(cin, cout):
    return nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1), ChannelLN(cout), nn.SiLU(),
                         nn.Conv3d(cout, cout, 3, padding=1), ChannelLN(cout), nn.SiLU())


class VelocityUNet(nn.Module):
    """Velocity node fields V_0..V_H of one context in a single forward pass.

    Input (B, 3 * 6 + 1 + 2, X, Y, Z): the fused states at k0, kp, k1 (scaled), the flag e, a time channel (zero) and s / 10.
    Output (B, 3 * (H + 1), X, Y, Z): the velocity at the H + 1 integer times after k0, in voxels per time step along the grid
    axes. The output layer is zero-initialized, so an untrained operator predicts no motion. X, Y, Z must be multiples of 8.
    """

    def __init__(self, widths=(32, 64, 96, 128), horizon=12, amp=True):
        super().__init__()
        cin = 3 * CHANNELS + 1 + 2
        self.horizon, self.amp = horizon, amp
        self.enc = nn.ModuleList([_block(cin, widths[0])] + [_block(widths[i], widths[i + 1]) for i in range(len(widths) - 1)])
        self.down = nn.ModuleList([nn.Conv3d(widths[i], widths[i], 2, stride=2) for i in range(len(widths) - 1)])
        self.dec = nn.ModuleList([_block(widths[i + 1] + widths[i], widths[i]) for i in range(len(widths) - 1)])
        self.out = nn.Conv3d(widths[0], 3 * (horizon + 1), 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, ctx, tau, s):
        sh = (-1, 1, *ctx.shape[2:])
        x = torch.cat([ctx, tau.view(-1, 1, 1, 1, 1).expand(sh), s.view(-1, 1, 1, 1, 1).expand(sh)], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and x.is_cuda):
            skips = []
            for i, enc in enumerate(self.enc):
                x = enc(x)
                if i < len(self.down):
                    skips.append(x)
                    x = self.down[i](x)
            for i in reversed(range(len(self.dec))):
                x = self.dec[i](torch.cat([F.interpolate(x, scale_factor=2, mode="trilinear"), skips[i]], 1))
            return self.out(x).float()


def make_context(z0, zp, z1):
    """Context tensor (B, 3 * 6 + 1, ...) from the scaled fused states at k0 and kp, and at k1 (None in future prediction,
    where the succeeding state is zero and the flag e is 0)."""
    e = torch.zeros_like(z0[:, :1]) if z1 is None else torch.ones_like(z0[:, :1])
    return torch.cat([z0, zp, torch.zeros_like(z0) if z1 is None else z1, e], 1)


def load_checkpoint(path, device="cpu", amp=True):
    """(model in eval mode, feature scale (1, 6, 1, 1, 1), checkpoint config) from a checkpoint written by train.py or
    the released model file."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck.get("cfg", {})
    widths = cfg.get("widths", "32,64,96,128")
    widths = tuple(int(v) for v in widths.split(",")) if isinstance(widths, str) else tuple(widths)
    model = VelocityUNet(widths, int(cfg.get("horizon", 12)), amp).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    scale = torch.as_tensor(ck["scale"], dtype=torch.float32, device=device).view(1, CHANNELS, 1, 1, 1)
    return model, scale, cfg
