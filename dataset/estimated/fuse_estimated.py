"""Fuse a scene from estimated depth and masks (the estimated-input experiment): the training views' RGB and poses
(training_views.py), Depth Anything 3 depth (da3_depth.py) and SAM 2.1 masks (sam_masks.py), on the grid of the
dataset fusion of the same scene (bounds_like), with the settings of the main experiments.

Depth beyond `far` is set invalid. `far` is the largest depth that plane-sweep stereo on the same RGB measures at the
task's observed frames (no ground truth and no target frame is read). The paper's values are in splits/estimated_far.json,
per scene and task. The masks are task-specific (in-between: the s4 observations, future: the observations before the
forecast), so each task has its own fusion.

Usage: python dataset/estimated/fuse_estimated.py <scene> --task inbetween|future --views <training-view cache dir>
       --depth <da3 depth dir> --masks <sam mask dir> --bounds-like <dataset fusion dir> --out <dir>
       [--far splits/estimated_far.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adapixflow.fusion import fuse_scene  # noqa: E402


def arg(k, d=None):
    if k in sys.argv:
        return sys.argv[sys.argv.index(k) + 1]
    if d is None:
        sys.exit(f"{k} is required\n\n{__doc__}")
    return d


def fuse_estimated(name, task, views_dir, depth_dir, mask_dir, bounds_like, out_dir, far_file):
    src = np.load(views_dir / f"{name}.npz")
    meta = json.loads(str(src["meta"]))
    dz, mz = np.load(depth_dir / f"{name}.npz"), np.load(mask_dir / f"{name}.npz")
    dm, mm = json.loads(str(dz["meta"])), json.loads(str(mz["meta"]))
    for what, m in (("depth", dm), ("masks", mm)):
        assert m["views"] == meta["views"], f"{what} views {m['views']} != the training views {meta['views']}"
    D, S = dz["depth"], mz["seg"]
    assert D.shape == S.shape == src["depth"].shape, (D.shape, S.shape, src["depth"].shape)
    far = json.load(open(far_file))[name][task]
    D = np.where(D > far, 0, D).astype(D.dtype)
    meta = dict(meta, depth=dm["model"], objects=mm["model"], far=far, task=task)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"{name}.npz"
    np.savez(tmp, depth=D, seg=S, rgb_jpeg=src["rgb_jpeg"], rgb_offsets=src["rgb_offsets"], c2w=src["c2w"], f=src["f"],
             W=src["W"], meta=json.dumps(meta))
    try:
        print(fuse_scene(tmp, out_dir, voxel=0.0125, stride=1, max_extent=4.0, occ_min=3, holdout=(),
                         bounds_like=bounds_like), flush=True)
    finally:
        tmp.unlink()


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
        sys.exit(__doc__)
    task = arg("--task")
    assert task in ("inbetween", "future"), task
    fuse_estimated(sys.argv[1], task, Path(arg("--views")), Path(arg("--depth")), Path(arg("--masks")),
                   Path(arg("--bounds-like")), Path(arg("--out")), Path(arg("--far", "splits/estimated_far.json")))
