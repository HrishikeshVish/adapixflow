"""Metrics on object points (voxel coordinates): F1 at 1 and 2 voxels and Chamfer distance."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def point_metrics(pred, gt, voxel, chamfer="mean"):
    """F1 at 1 and 2 voxels (precision: predicted points within the threshold of a GT point; recall: the converse) and the
    symmetric Chamfer distance in mm, as the mean (held-out evaluation) or the median (comparison scenes) of the
    nearest-neighbour distances. An empty prediction scores F1 0."""
    if len(gt) == 0:
        return {"f1_1": np.nan, "f1_2": np.nan, "chamfer_mm": np.nan}
    if len(pred) == 0:
        return {"f1_1": 0.0, "f1_2": 0.0, "chamfer_mm": np.nan}
    da, db = cKDTree(gt).query(pred)[0], cKDTree(pred).query(gt)[0]
    agg = np.mean if chamfer == "mean" else np.median
    out = {"chamfer_mm": float(1000 * voxel * 0.5 * (agg(da) + agg(db)))}
    for tv in (1, 2):
        pr, rc = (da <= tv).mean(), (db <= tv).mean()
        out[f"f1_{tv}"] = float(2 * pr * rc / max(pr + rc, 1e-9))
    return out


def moved_since(ref_pts, pts):
    """Mean symmetric nearest-neighbour distance (voxels) between the object points of two steps: a target is scored only
    if this exceeds one voxel, i.e. the object moved since its reference observation."""
    if len(ref_pts) == 0 or len(pts) == 0:
        return 0.0
    return 0.5 * (cKDTree(pts).query(ref_pts)[0].mean() + cKDTree(ref_pts).query(pts)[0].mean())
