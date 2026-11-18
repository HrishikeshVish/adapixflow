"""Scenes, splits, moving voxels, evaluation windows, and the training sampler."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .fusion import FusedScene


def load_split(path):
    """A split file: {"train": {tier: [scene, ...]}, "test": {tier: [...]}, "val": [...], "comparison": [...]}."""
    return json.load(open(path))


class SceneSet:
    """Fused scenes of a split part, looked up in one or more fused directories."""

    def __init__(self, names_by_tier, fused_dirs):
        self.items = []
        for tier, names in names_by_tier.items():
            for n in names:
                for d in fused_dirs:
                    if (Path(d) / f"{n}.json").exists():
                        S = FusedScene(d, n)
                        self.items.append({"name": n, "tier": tier, "S": S, "meta": S.meta})
                        break
        self.by_tier = {t: [i for i, it in enumerate(self.items) if it["tier"] == t] for t in names_by_tier}
        self.by_tier = {t: v for t, v in self.by_tier.items() if v}
        missing = sum(len(v) for v in names_by_tier.values()) - len(self.items)
        self.missing = missing

    def __len__(self):
        return len(self.items)


def moved_voxels(S, k0, k1, r=2):
    """Object voxels (linear indices) of step k0 with no object voxel of step k1 within L-inf distance r, and vice versa:
    the voxels that the motion between the two steps changes."""
    G = np.asarray(S.G, np.int64)
    A, B = S.object_voxels(k0), S.object_voxels(k1)
    offs = np.array([dx * G[1] * G[2] + dy * G[2] + dz for dx in range(-r, r + 1) for dy in range(-r, r + 1)
                     for dz in range(-r, r + 1)], np.int64)
    out = []
    for X, Y in ((A, B), (B, A)):
        X = np.setdiff1d(X, Y, assume_unique=True)
        if len(X) and len(Y):
            near = np.isin((X[:, None] + offs[None]).ravel(), Y).reshape(len(X), -1).any(1)
            out.append(X[~near])
        elif len(X):
            out.append(X)
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def motion_window(S, max_side=128):
    """Evaluation window around the moving region: the 1-99 percentile box of the voxels that move between steps k and
    k + 2, plus 8 voxels, sides rounded up to multiples of 8 and capped at max_side. Returns (center, size)."""
    G, pts = np.asarray(S.G), []
    for k in range(0, S.T - 2):
        mv = moved_voxels(S, k, k + 2)
        if len(mv):
            pts.append(np.stack(np.unravel_index(mv, G), 1))
    if not pts:
        pts = [np.stack(np.unravel_index(S.object_voxels(k), G), 1) for k in range(S.T)]
    P = np.concatenate(pts)
    lo, hi = np.percentile(P, 1, 0), np.percentile(P, 99, 0)
    size = np.minimum(((hi - lo + 1 + 16) / 8).astype(int) * 8 + 8, max_side)
    return np.round((lo + hi) / 2).astype(int), size


def feature_scale(scenes, rng, n_scenes=40):
    """Per-channel input scale: the standard deviation of RGB and objectness over the occupied voxels of a sample of
    training scenes (shared by the four color/objectness channels; occupancy and unseen are not scaled)."""
    vals = []
    for it in [scenes.items[i] for i in rng.choice(len(scenes), min(n_scenes, len(scenes)), replace=False)]:
        vals.append(np.asarray(it["S"].rgbo[:: max(1, len(it["S"].rgbo) // 200000)], np.float32).ravel() / 255.0)
    scale = np.ones(6, np.float32)
    scale[:4] = np.concatenate(vals).std() + 1e-6
    return scale


class TrainSampler:
    """Training windows (Sec. 3.4 and Tab. 7): a spacing and horizon per batch, crops centred on moving voxels, and a redraw
    until enough object voxels move. Half the samples come from scenes drawn by the squared moving average of their
    loss relative to a static prediction (adaptive sampling), the rest uniformly over tiers."""

    def __init__(self, scenes, cfg, rng):
        self.S, self.c, self.rng = scenes, cfg, rng
        self.ema_rel = np.ones(len(scenes.items))  # per-scene moving average of the loss relative to no motion

    def horizon_cap(self, step):
        c = self.c
        return int(min(c["horizon_max"], 1 + (c["horizon_max"] - 1) * step / max(1, c["curriculum_frac"] * c["steps"])))

    def sample(self, step):
        """Returns (crops (B, h + 2, 6, X, Y, Z) unscaled, masks (B, X, Y, Z), s, h, use_end, scene indices); frame 0 of each
        crop is the preceding observation kp, frame 1 the initial observation k0, frame j + 1 the step k0 + j."""
        c, rng = self.c, self.rng
        hcap = self.horizon_cap(step)
        s = int(rng.choice([x for x in c["spacings"] if x <= hcap]))
        use_end = s >= 2 and rng.random() < c["p_end"]  # s = 1 with the end given is trivial (step 1 is the end)
        h = s if (use_end or rng.random() < 0.5) else int(rng.integers(1, hcap + 1))
        if rng.random() < c["p_eval_settings"]:
            s, h, use_end = [tuple(x) for x in c["eval_settings"]][int(rng.integers(len(c["eval_settings"])))]
            use_end = bool(use_end)
        crop = c["crop"]
        xs, ms, idx = [], [], []
        for _ in range(c["batch"]):
            for _try in range(20):
                if rng.random() < c["p_adaptive"]:
                    w = self.ema_rel ** 2
                    ii = int(rng.choice(len(self.S.items), p=w / w.sum()))
                else:
                    t = list(self.S.by_tier)[int(rng.integers(len(self.S.by_tier)))]
                    ii = int(rng.choice(self.S.by_tier[t]))
                it = self.S.items[ii]
                T = it["S"].T
                if T - 1 - h < 0:
                    continue
                k0 = int(rng.choice(np.arange(0, T - h, s)))
                movers = [p for k in range(k0, k0 + h + 1) for p in it["meta"]["movers"][k]]
                mvx = moved_voxels(it["S"], k0, k0 + h)
                if movers and len(mvx) >= c["min_moved"]:
                    break
            base = np.array(movers[int(rng.integers(len(movers)))]) if movers else it["S"].G // 2
            if len(mvx):
                base = np.array(np.unravel_index(mvx[int(rng.integers(len(mvx)))], it["S"].G))
            center = base + rng.integers(-crop // 4, crop // 4 + 1, 3)
            kp = max(k0 - s, 0)
            arr, m = it["S"].crop([kp] + list(range(k0, k0 + h + 1)), center, crop)
            xs.append(arr)
            ms.append(m)
            idx.append(ii)
        return np.stack(xs), np.stack(ms), s, h, use_end, idx
