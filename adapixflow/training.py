"""The training loss of one batch (Sec. 3.4)."""

from __future__ import annotations

import numpy as np
import torch

from .flow import advect_trajectory, substep_displacements
from .fusion import OBJ, OCC
from .losses import distance_preservation, neighbors, relative_chamfer
from .model import TIME_UNIT, make_context


def batch_loss(model, scale, raw, s, h, use_end, step, tc, dev):
    """Loss of one batch of crops raw (B, h + 2, 6, X, Y, Z) (frames kp, k0, k0 + 1, ..., k0 + h; unscaled).

    The object points of k0, and of k0 + h when the next observation is given, are advected forward (backward) through the
    predicted field and compared with the object points of every target step by the relative Chamfer distance; the local
    distance-preservation term is added from rigid_start_frac of the training on. Returns (loss, per-sample sum of the
    relative point losses, per-sample count) for the adaptive scene sampling."""
    x = torch.from_numpy(raw).to(dev) / scale.unsqueeze(1)
    obj = torch.from_numpy((raw[:, :, OCC] > 0.5) & (raw[:, :, OBJ] > 0.5)).to(dev)
    ctx = make_context(x[:, 1], x[:, 0], x[:, -1] if use_end else None)
    B, crop, substeps, npts = x.shape[0], tc["crop"], tc["substeps"], tc["points"]
    V = model(ctx, torch.zeros(B, device=dev), torch.full((B,), s / TIME_UNIT, device=dev))
    D = substep_displacements([V[:, 3 * m:3 * m + 3] for m in range(h + 1)], h, substeps)

    def points(b, fi):  # up to `npts` object points of crop frame fi
        q = torch.nonzero(obj[b, fi])
        return q[torch.randperm(len(q), device=dev)[:npts]].float()

    lp, n_lp, lr, n_lr = 0.0, 0, 0.0, 0
    rel, cnt = np.zeros(B), np.zeros(B)
    rigid_on = tc["rigid_weight"] > 0 and step >= tc["rigid_start_frac"] * tc["steps"]
    for b in range(B):
        Db = [d[b:b + 1] for d in D]
        tgt = {j: points(b, j + 1) for j in range(0 if use_end else 1, h + 1)}  # step j after k0 -> target points
        for fi, sign in [(1, +1)] + ([(h + 1, -1)] if use_end else []):  # forward from k0, backward from k1
            q = points(b, fi)
            if len(q) == 0:
                continue
            traj = advect_trajectory(Db, q, substeps, sign)
            if rigid_on and len(q) > tc["rigid_neighbors"] + 1:
                nb = neighbors(q, tc["rigid_neighbors"])
                for pj in (traj if sign > 0 else list(traj.values())):
                    lr = lr + distance_preservation(pj, q, nb)
                    n_lr += 1
            for j in (range(1, h + 1) if sign > 0 else range(0, h)):
                pj = traj[j - 1] if sign > 0 else traj.get(j)
                if pj is None or len(tgt[j]) == 0:
                    continue
                lj = relative_chamfer(pj, tgt[j], q, crop, tc["loss_floor"])
                lp = lp + lj
                n_lp += 1
                rel[b] += float(lj)
                cnt[b] += 1
    loss = (lp / n_lp if n_lp else 0.0) + (tc["rigid_weight"] * lr / n_lr if n_lr else 0.0)
    return loss, rel, cnt
