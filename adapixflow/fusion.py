"""Volumetric fusion of multi-view RGB-D observations (Sec. 3.1 of the paper).

Every pixel with valid depth is back-projected with the pinhole model and the points of all cameras are fused into one
voxel grid in world coordinates. Each voxel stores six channels:

    RGB (3)   color in [0, 1], trilinear splat over all cameras, normalized at each occupied voxel
    OBJ (1)   fraction of the splat weight that comes from moving-object pixels
    OCC (1)   occupied if at least `occ_min` points round to the voxel (nearest-voxel binning)
    UNSEEN(1) unoccupied, and in every camera outside the image or more than half a voxel behind the observed depth

A fused scene is stored sparsely (dense grids are ~99% empty):

    <scene>.occ.npy   int32 (n,)                     linear (C-order) indices of the occupied voxels, all steps concatenated
    <scene>.rgbo.npy  uint8 (n, 4)                   RGB and OBJ of those voxels, times 255
    <scene>.uns.npy   uint8 (T, Gx, Gy, ceil(Gz/8))  UNSEEN, bits packed along z
    <scene>.json      grid origin, voxel size, grid size, per-step offsets, cameras, moving labels, movers (written last)

The input is a scene cache produced by dataset/fetch_physinone.py (or a dataset converter with the same layout).
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

CHANNELS = 6  # RGB, OBJ, OCC, UNSEEN
OCC = 4       # index of the occupancy channel
OBJ = 3       # index of the objectness channel


def moving_labels(meta: dict) -> set:
    """Segmentation labels of the moving objects of a PhysInOne scene: objects tagged as moving, or with a stencil label
    >= 128 (PhysInOne's dynamic range: tagged rigid movers, MPM materials, liquids and what they carry). A cache may
    instead list the labels directly in `moving_labels_override` (inputs whose masks were estimated)."""
    if meta.get("moving_labels_override"):
        return set(meta["moving_labels_override"])
    rec = meta.get("recorder_stats") or {}
    labels = set()
    for group in ("solid_object", "interactable_object", "breakable_object"):
        for obj in rec.get(group) or []:
            for spec in obj.values():
                stencils = [c.get("custom_stencil_value") for c in (spec.get("components") or [])
                            if c.get("custom_stencil_value") is not None]
                if (spec.get("tags") or {}).get("has_moving_tag") or any(s >= 128 for s in stencils):
                    labels.update(int(s) for s in stencils)
    return labels - {0}  # 0 = unlabeled pixels (background), even when a moving object is recorded with stencil 0


def fuse_scene(cache_file: Path, out_dir: Path, voxel: float = 0.0125, stride: int = 1, max_extent: float = 4.0,
               occ_min: int = 3, holdout: tuple = (), bounds_like: Path | None = None, frame_step: int = 2) -> str:
    """Fuse every step of one scene cache into a sparse voxel grid (see the module docstring). Step k uses video frame
    frame_step * k: a PhysInOne step is two video frames, and the benchmark converters write every native frame twice, so
    the default of 2 serves both. `holdout` lists camera indices left out entirely (novel views). `bounds_like` takes the
    grid origin and size from another fusion of the same scene, so that both cover the same region."""
    t_start = time.time()
    name = cache_file.name[:-4]
    out_json = out_dir / f"{name}.json"
    if out_json.exists():
        return f"skip {name}"
    d = np.load(cache_file)
    meta = json.loads(str(d["meta"]))
    depth, seg, c2w, f, W = d["depth"], d["seg"], d["c2w"], float(d["f"]), int(d["W"])
    jpeg, offs = d["rgb_jpeg"], d["rgb_offsets"]
    C, T = depth.shape[:2]
    cams = [c for c in range(C) if c not in holdout]
    mov = moving_labels(meta)
    if not mov:
        return f"FAIL {name}: no moving labels"
    lut = np.zeros(256, bool)
    lut[list(mov)] = True
    v, u = np.mgrid[0:W:stride, 0:W:stride] + 0.5

    def unproject(ci, t):
        """World points of camera ci at frame t (OpenGL camera: x right, y up, looking along -z; planar depth)."""
        z = depth[ci, t, ::stride, ::stride].astype(np.float64)
        pc = np.stack([(u - W / 2) / f * z, -(v - W / 2) / f * z, -z], -1).reshape(-1, 3)
        M = c2w[ci, t]
        return pc @ M[:3, :3].T + M[:3, 3], (z.reshape(-1) > 0) & (z.reshape(-1) < 1e3)

    # grid bounds: the 0.5-99.5 percentile box of the moving-object points over the clip (every 5th frame) plus 5 cm. An
    # axis longer than max_extent is cut to a window centred on the 2-98 percentile box.
    if bounds_like is not None:
        ref = json.load(open(bounds_like / f"{name}.json"))
        assert abs(ref["VOX"] - voxel) < 1e-9, "bounds_like must use the same voxel size"
        lo, G = np.array(ref["lo"], np.float64), np.array(ref["G"], int)
    else:
        pts = []
        for t in range(0, T, 5):
            for ci in cams:
                p, ok = unproject(ci, t)
                pts.append(p[lut[seg[ci, t, ::stride, ::stride].reshape(-1)] & ok])
        pts = np.concatenate(pts)
        if len(pts) < 100:
            return f"FAIL {name}: too few moving-object points"
        lo, hi = np.percentile(pts, 0.5, 0) - 0.05, np.percentile(pts, 99.5, 0) + 0.05
        mid = (np.percentile(pts, 2, 0) + np.percentile(pts, 98, 0)) / 2
        lo = np.where(hi - lo > max_extent, mid - max_extent / 2, lo)
        hi = np.minimum(hi, lo + max_extent)
        G = np.ceil((hi - lo) / voxel).astype(int)
    N, Tl = int(np.prod(G)), T // frame_step
    rng = np.random.default_rng(0)
    corners = np.array([[dx, dy, dz] for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)])

    # occupancy (nearest voxel) and color / objectness (trilinear splat, normalized at occupied voxels), per step
    occ_parts, rgbo_parts, offsets, movers = [], [], [0], []
    for k in range(Tl):
        t, Q, COL = frame_step * k, [], []
        for ci in cams:
            p, ok = unproject(ci, t)
            q = (p - lo) / voxel - 0.5  # grid coordinates g(y)
            keep = ok & np.all((q > -1) & (q < G), axis=1)
            if not keep.any():
                continue
            rgb = np.asarray(Image.open(io.BytesIO(jpeg[offs[ci * T + t]:offs[ci * T + t + 1]].tobytes())).convert("RGB"),
                             np.float64)[::stride, ::stride].reshape(-1, 3)[keep] / 255
            ob = lut[seg[ci, t, ::stride, ::stride].reshape(-1)][keep].astype(np.float64)
            Q.append(q[keep])
            COL.append(np.concatenate([rgb, ob[:, None]], 1))
        occ_lin = np.zeros(0, np.int64)
        if Q:
            q, col = np.concatenate(Q), np.concatenate(COL)
            nb = np.round(q).astype(np.int64)
            okb = np.all((nb >= 0) & (nb < G), axis=1)
            lin_nb, cnt = np.unique(np.ravel_multi_index(nb[okb].T, G), return_counts=True)
            occ_lin = lin_nb[cnt >= occ_min]
        M_occ = len(occ_lin)
        w_acc, c_acc = np.zeros(M_occ), np.zeros((4, M_occ))
        if M_occ:
            i0 = np.floor(q).astype(np.int64)
            for cc in corners:
                idx = i0 + cc
                w = np.prod(1 - np.abs(q - idx), axis=1)
                okc = np.all((idx >= 0) & (idx < G), axis=1) & (w > 0)
                lin = np.ravel_multi_index(idx[okc].T, G)
                pos = np.searchsorted(occ_lin, lin).clip(max=M_occ - 1)
                hit = occ_lin[pos] == lin
                pos, wh, ch_v = pos[hit], w[okc][hit], col[okc][hit]
                w_acc += np.bincount(pos, weights=wh, minlength=M_occ)
                for ch in range(4):
                    c_acc[ch] += np.bincount(pos, weights=wh * ch_v[:, ch], minlength=M_occ)
        colr = c_acc / np.maximum(w_acc, 1e-8)
        occ_parts.append(occ_lin.astype(np.int32))
        rgbo_parts.append(np.round(colr.T * 255).clip(0, 255).astype(np.uint8))
        offsets.append(offsets[-1] + M_occ)
        mv = occ_lin[colr[3] > 0.5]
        movers.append(np.stack(np.unravel_index(rng.choice(mv, min(64, len(mv)), replace=False), G), 1).tolist()
                      if len(mv) else [])

    # UNSEEN: not occupied and hidden from every camera. Camera coordinates are separable over the grid axes, so the voxel
    # centres are projected once per camera pose and tested against the depth of every step; a moving camera (PhysGaia's
    # training views) is re-projected whenever its pose changes, a static one once.
    seen = np.zeros((Tl, G[0], G[1], (G[2] + 7) // 8), np.uint8)
    ax = [(np.arange(n) + 0.5) * voxel + lo[i] for i, n in enumerate(G)]
    for ci in cams:
        last = None
        for k in range(Tl):
            Mk = c2w[ci, frame_step * k]
            if last is None or not np.array_equal(Mk, last):
                R, tc = Mk[:3, :3], Mk[:3, 3]
                a = [np.multiply.outer(ax[i] - tc[i], R[i]) for i in range(3)]

                def cam(j):
                    return (a[0][:, None, None, j] + a[1][None, :, None, j] + a[2][None, None, :, j]).astype(np.float32).ravel()

                zc = -cam(2)
                zs = np.maximum(zc, 1e-6)
                uu, vv = cam(0) / zs * f + W / 2, -cam(1) / zs * f + W / 2
                sel = np.flatnonzero((zc > 1e-6) & (uu >= 0) & (uu < W) & (vv >= 0) & (vv < W)).astype(np.int32)
                pix, zin = vv[sel].astype(np.int32) * W + uu[sel].astype(np.int32), zc[sel]
                del zc, zs, uu, vv
                last = np.array(Mk, copy=True)
            in_front = zin <= depth[ci, frame_step * k].ravel().astype(np.float32)[pix] + 0.5 * voxel
            fr = np.zeros(N, bool)
            fr[sel[in_front]] = True
            seen[k] |= np.packbits(fr.reshape(G), axis=-1)
    for k in range(Tl):
        fr = np.zeros(N, bool)
        fr[occ_parts[k]] = True
        seen[k] = ~(seen[k] | np.packbits(fr.reshape(G), axis=-1))  # now UNSEEN

    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix, arr in (("occ", np.concatenate(occ_parts)), ("rgbo", np.concatenate(rgbo_parts)), ("uns", seen)):
        tmp = out_dir / f"{name}.{suffix}.tmp.npy"
        np.save(tmp, arr)
        tmp.rename(out_dir / f"{name}.{suffix}.npy")
    info = {"format": "sparse1", "lo": lo.tolist(), "VOX": voxel, "G": G.tolist(), "offsets": offsets,
            "c2w": c2w[:, 0].tolist(), "f": f, "W": W, "stride": stride, "occ_min": occ_min, "per_latent": 1,
            "holdout_views": list(holdout), "moving_labels": sorted(mov), "movers": movers,
            **{k: meta[k] for k in meta if k != "recorder_stats"}}
    tmp = out_dir / f"{name}.json.tmp"
    json.dump(info, open(tmp, "w"))
    tmp.rename(out_json)
    return f"ok {name}: grid {G.tolist()}, {Tl} steps, {offsets[-1] / max(Tl, 1):.0f} occupied voxels/step, {time.time() - t_start:.0f} s"


class FusedScene:
    """A fused scene (fuse_scene output), read back as dense float32 crops with channels RGB, OBJ, OCC, UNSEEN."""

    def __init__(self, fused_dir: Path, name: str):
        fused_dir = Path(fused_dir)
        self.name = name
        self.meta = json.load(open(fused_dir / f"{name}.json"))
        self.occ = np.load(fused_dir / f"{name}.occ.npy", mmap_mode="r")
        self.rgbo = np.load(fused_dir / f"{name}.rgbo.npy", mmap_mode="r")
        self.uns = np.load(fused_dir / f"{name}.uns.npy", mmap_mode="r")
        self.off = np.asarray(self.meta["offsets"])
        self.G = np.asarray(self.meta["G"])
        self.T = len(self.off) - 1
        self.voxel = float(self.meta["VOX"])
        self.lo = np.asarray(self.meta["lo"], np.float64)

    def crop(self, steps, center, size):
        """Steps `steps` in a window of `size` voxels (int or 3-tuple) centred at `center`, zero outside the grid.
        Returns (array (len(steps), 6, *size), validity mask (*size))."""
        size = np.broadcast_to(np.asarray(size, int), (3,))
        G, lo = self.G, np.asarray(center) - size // 2
        out = np.zeros((len(steps), CHANNELS, *size), np.float32)
        mask = np.zeros(tuple(size), np.float32)
        s0, s1 = np.maximum(lo, 0), np.minimum(lo + size, G)
        if not np.all(s1 > s0):
            return out, mask
        d0, d1 = s0 - lo, s1 - lo
        win = (slice(d0[0], d1[0]), slice(d0[1], d1[1]), slice(d0[2], d1[2]))
        mask[win] = 1
        b0, b1 = s0[2] // 8, (s1[2] + 7) // 8
        for j, k in enumerate(steps):
            a, b = self.off[k], self.off[k + 1]
            xyz = np.stack(np.unravel_index(np.asarray(self.occ[a:b]), G), 1) - lo
            inb = np.all((xyz >= 0) & (xyz < size), axis=1)
            x, y, z = xyz[inb].T
            fr = out[j]
            fr[:4, x, y, z] = np.asarray(self.rgbo[a:b])[inb].T / 255.0
            fr[OCC, x, y, z] = 1
            bits = np.unpackbits(np.asarray(self.uns[k, s0[0]:s1[0], s0[1]:s1[1], b0:b1]), axis=-1)
            fr[5][win] = bits[..., s0[2] - 8 * b0:s1[2] - 8 * b0]
        return out, mask

    def object_voxels(self, k):
        """Linear grid indices of the moving-object voxels of step k."""
        a, b = self.off[k], self.off[k + 1]
        return np.asarray(self.occ[a:b]).astype(np.int64)[np.asarray(self.rgbo[a:b, OBJ]) > 127]


def object_points(Z):
    """Moving-object voxels (occupied, objectness > 0.5) of a dense frame (6, X, Y, Z), as float32 points (n, 3)."""
    return np.argwhere((Z[OCC] > 0.5) & (Z[OBJ] > 0.5)).astype(np.float32)
