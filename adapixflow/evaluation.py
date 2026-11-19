"""Evaluation protocols of the paper.

held-out scenes (Tab. 3): in-between prediction at spacings 2, 4, 8 and future prediction of 10 steps on the Test and Val
    scenes, inside a motion window capped at 128 voxels, with all views as input. Mean Chamfer distance.
comparison scenes (Tab. 1): in-between prediction at spacing 4 and future prediction of 10 steps, with views 0 and 1 held
    out of the input fusion and the full-view fusion as ground truth, over the whole moving region. Median Chamfer.

Both score F1 at 1 and 2 voxels on the target steps at which the ground-truth object moved by more than one voxel since
the reference observation (the observation before the gap, or the last observation), average over the scored steps of a
scene, then over scenes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data import motion_window
from .fusion import FusedScene, object_points
from .flow import advect, node_fields
from .metrics import moved_since, point_metrics
from .model import make_context


class _Dense:
    """Dense frames of one scene window, cached."""

    def __init__(self, S, center, size, keep=24):
        self.S, self.center, self.size, self.keep, self.cache = S, center, size, keep, {}

    def __call__(self, k):
        if k not in self.cache:
            if len(self.cache) > self.keep:
                self.cache.pop(next(iter(self.cache)))
            self.cache[k] = self.S.crop([k], self.center, self.size)[0][0]
        return self.cache[k]


def _nodes(model, scale, dense, k0, kp, k1, s, n, tiling):
    g = lambda k: torch.from_numpy(dense(k))[None] / scale.cpu()  # noqa: E731
    ctx = make_context(g(k0), g(kp), None if k1 is None else g(k1))
    return node_fields(model, ctx, s, n, tiling)


def _advect_objects(Vn, pts, t0, t1, size, substeps):
    """Object points advected from time t0 to t1, without those carried out of the window."""
    if len(pts):
        dev = Vn[0].device
        pts = advect(Vn, torch.from_numpy(pts).to(dev), t0, t1, substeps).cpu().numpy()
    return pts[np.all((pts > -0.5) & (pts < np.asarray(size) - 0.5), 1)]


def predictions(model, scale, dense, T, size, task, s=4, horizon=10, substeps=8, tiling=None, objects=None):
    """{target step: (predicted object voxels (unique, rounded), reference observation)} for one task on one window.

    task "inbetween": observations every s steps, each target advected from the nearer observation (ties at j = s / 2
    from the succeeding one). task "future": observations up to the middle of the clip, the last one advected forward."""
    objects = objects or (lambda k: object_points(dense(k)))
    out = {}
    if task == "future":
        kobs = (T // 2) // 2 * 2
        n = min(horizon, T - 1 - kobs)
        Vn = _nodes(model, scale, dense, kobs, kobs - 2, None, 2, n, tiling)
        src = objects(kobs)
        for j in range(1, n + 1):
            p = _advect_objects(Vn, src, 0, j, size, substeps)
            out[kobs + j] = (np.unique(np.round(p), axis=0), kobs)
        return out
    obs = list(range(0, T, s))
    for i, k0 in enumerate(obs[:-1]):
        k1, kp = obs[i + 1], obs[i - 1] if i > 0 else k0
        Vn = _nodes(model, scale, dense, k0, kp, k1, s, s, tiling)
        for j in range(1, s):
            near = k0 if j < s / 2 else k1
            p = _advect_objects(Vn, objects(near), 0 if near == k0 else s, j, size, substeps)
            out[k0 + j] = (np.unique(np.round(p), axis=0), k0)
    return out


def score(preds, gt_points, voxel, chamfer, shift=0.0):
    """Mean metrics over the scored targets of one scene and task; gt_points(k) gives the GT object points of step k in
    the prediction's window coordinates plus `shift`."""
    vals, per_step = [], {}
    for k, (P, kref) in sorted(preds.items()):
        g, r = gt_points(k), gt_points(kref)
        if not (len(g) and len(r) and moved_since(r, g) > 1):
            continue  # the object moved by at most one voxel since the reference observation: not scored
        m = point_metrics(P + shift if len(P) else P, g, voxel, chamfer)
        vals.append([m["f1_1"], m["f1_2"], m["chamfer_mm"]])
        per_step[int(k)] = m["f1_2"]
    if not vals:
        return {"steps": 0}
    v = np.nanmean(np.array(vals, float), 0)
    return {"steps": len(vals), "f1_1": float(v[0]), "f1_2": float(v[1]), "chamfer_mm": float(v[2]), "per_step_f1_2": per_step}


@torch.no_grad()
def evaluate_heldout(model, scale, scenes, cfg, log=print):
    """Held-out evaluation (Tab. 3) of a SceneSet. Returns {task: {"scenes": {name: metrics}, "mean": {...}}}."""
    tasks = [(f"inbetween_s{s}", "inbetween", s) for s in cfg["spacings"]] + [("future10", "future", 2)]
    res = {t: {"scenes": {}} for t, _, _ in tasks}
    for it in scenes.items:
        S = it["S"]
        center, size = motion_window(S, cfg["max_side"])
        dense = _Dense(S, center, size, keep=6)
        objp = {k: object_points(dense(k)) for k in range(S.T)}
        for name, task, s in tasks:
            P = predictions(model, scale, dense, S.T, size, task, s, cfg["horizon"], cfg["substeps"], tuple(cfg["tiling"]),
                            objp.__getitem__)
            m = score(P, objp.__getitem__, S.voxel, "mean")
            res[name]["scenes"][it["name"]] = {"tier": it["tier"], **m}
        log(f"  {it['name'][:48]:48s} " + "  ".join(
            f"{n} F1@2 {res[n]['scenes'][it['name']].get('f1_2', np.nan) * 100:5.1f}" for n, _, _ in tasks))
    for name in res:
        rows = [m for m in res[name]["scenes"].values() if m["steps"]]
        res[name]["mean"] = {k: float(np.nanmean([m[k] for m in rows])) for k in ("f1_1", "f1_2", "chamfer_mm")} if rows else {}
    return res


@torch.no_grad()
def evaluate_comparison(model, scale, names, gt_dir, input_dir, cfg, log=print):
    """Comparison-scene evaluation (Tab. 1): inputs fused from the training views (input_dir), ground truth from all views
    (gt_dir), on the whole moving region. input_dir may map each task to its own fusion ({"inbetween_s4": dir,
    "future10": dir}, as for estimated masks); a scene without a fusion for a task is not scored on it.
    Returns {task: {"scenes": {name: metrics}, "mean": {...}}}."""
    tasks = [("inbetween_s4", "inbetween", 4), ("future10", "future", 2)]
    dirs = input_dir if isinstance(input_dir, dict) else {t: input_dir for t, _, _ in tasks}
    tasks = [t for t in tasks if dirs.get(t[0])]
    res = {t: {"scenes": {}} for t, _, _ in tasks}
    for name in names:
        Sg = FusedScene(gt_dir, name)
        voxel, lo_g = Sg.voxel, Sg.lo
        cg, size = motion_window(Sg, cfg["max_side"])
        corner_g = cg - size // 2
        gt_objects = {k: object_points(Sg.crop([k], cg, size)[0][0]) for k in range(Sg.T)}
        cache = {}
        for tname, task, s in tasks:
            d = Path(dirs[tname])
            if not (d / f"{name}.json").exists():
                continue
            if d not in cache:
                Si = FusedScene(d, name)
                ci = np.round(((cg + 0.5) * voxel + lo_g - Si.lo) / voxel - 0.5).astype(int)  # the same window in the input grid
                corner_i = ci - size // 2
                shift = ((corner_i + 0.5) * voxel + Si.lo - ((corner_g + 0.5) * voxel + lo_g)) / voxel
                cache = {d: (_Dense(Si, ci, size, keep=6), shift)}
            in_dense, shift = cache[d]
            P = predictions(model, scale, in_dense, Sg.T, size, task, s, cfg["horizon"], cfg["substeps"], tuple(cfg["tiling"]))
            res[tname]["scenes"][name] = score(P, gt_objects.__getitem__, voxel, "median", shift)
        log(f"  {name[:48]:48s} window {size.tolist()}  " + "  ".join(
            f"{t} F1@2 {res[t]['scenes'][name].get('f1_2', np.nan) * 100:5.1f} (n={res[t]['scenes'][name]['steps']})"
            for t, _, _ in tasks if name in res[t]["scenes"]))
    for tname in res:
        rows = [m for m in res[tname]["scenes"].values() if m["steps"]]
        res[tname]["mean"] = {k: float(np.nanmean([m[k] for m in rows])) for k in ("f1_1", "f1_2", "chamfer_mm")} if rows else {}
    return res


def motion_split_scores(res, split_file, scenes=None):
    """F1@2 (%) pooled over the targets of each GT-defined motion subset (Tab. 2 and Tab. 9): in-between targets that
    the oracle linear interpolation predicts with F1 >= threshold (linear) or below (non-linear), non-linear without
    PanelSpin, future targets 1-3 and 4-10 steps ahead, and future targets before or after the first ground contact.
    res: evaluate_comparison output; scenes: restrict to these scenes (e.g. the common subset of a benchmark)."""
    import json
    S = json.load(open(split_file))
    thr, contact = S.get("threshold", 90.0), S.get("contact_step", {})
    groups = {}

    def add(name, v):
        groups.setdefault(name, []).append(v)

    for key, targets in S["targets"].items():
        scene, task = key.rsplit("/", 1)
        if scenes is not None and scene not in scenes:
            continue
        per_step = res.get(task, {}).get("scenes", {}).get(scene, {}).get("per_step_f1_2", {})
        per_step = {int(k): v for k, v in per_step.items()}
        for k, t in targets.items():
            k = int(k)
            if k not in per_step:
                continue
            f1 = 100 * per_step[k]
            if task.startswith("inbetween"):
                cls = "linear" if t["oracle"] >= thr else "non-linear"
                add(cls, f1)
                if cls == "non-linear" and "PanelSpin" not in scene:
                    add("non-linear w/o PanelSpin", f1)
            else:
                add("future, j<=3" if t["j"] <= 3 else "future, j>=4", f1)
                if contact:
                    kc = contact.get(scene)
                    add("future, before contact" if kc is None or k <= kc else "future, after contact", f1)
    order = ["linear", "non-linear", "non-linear w/o PanelSpin", "future, j<=3", "future, j>=4", "future, before contact",
             "future, after contact"]
    return {g: {"f1_2": float(np.mean(groups[g])), "n": len(groups[g])} for g in order if g in groups}
