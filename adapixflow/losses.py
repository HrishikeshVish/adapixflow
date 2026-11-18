"""Training objective (Sec. 3.4): relative Chamfer point loss and local distance preservation."""

from __future__ import annotations

import torch


def relative_chamfer(pred, target, source, crop, floor=0.5):
    """Symmetric Chamfer distance (voxels) between advected points `pred` and target points, normalized by the distance of
    a static prediction (the unadvected `source` points), floored at `floor` voxels. Predicted points carried out of the
    crop have no observable target and are left out of the prediction-to-target term."""
    inside = ((pred.detach() > -0.5) & (pred.detach() < crop - 0.5)).all(1).float()
    d = torch.cdist(pred, target)
    loss = 0.5 * ((d.min(1)[0] * inside).sum() / inside.sum().clamp_min(1) + d.min(0)[0].mean())
    with torch.no_grad():
        d0 = torch.cdist(source, target)
        static = (0.5 * (d0.min(1)[0].mean() + d0.min(0)[0].mean())).clamp_min(floor)
    return loss / static


def neighbors(source, k=6):
    """Indices (N, k) of the k nearest neighbours of each source point among the source points."""
    return torch.cdist(source, source).topk(k + 1, largest=False).indices[:, 1:]


def distance_preservation(pred, source, nb):
    """Mean change (voxels) of the distances between each point and its source neighbours nb after advection."""
    d_src = (source[:, None] - source[nb]).norm(dim=-1)
    return ((pred[:, None] - pred[nb]).norm(dim=-1) - d_src).abs().mean()
