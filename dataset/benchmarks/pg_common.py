"""Shared PhysGaia helpers (numpy / scipy / PIL only). Conventions, as checked in check_geometry.py:

  cameras   camera_info_*.json: per frame "transform_matrix" = camera-to-world, OpenGL / NeRF axes (x right, y up, looking
            along -z), "time" = frame / 24 s, "file_path" = ./<split>/<view>_<frame:03d> (frame 1..240). One global
            "camera_angle_x". Training views (render/train, camera_info_train_multi.json or camera_info_train.json for the
            2-camera scenes) move along spiral paths; test views 1, 2 (render/test, camera_info_test.json) are static.
  intrinsics images are 640 wide x 720 tall (RGBA, alpha = 255 everywhere: no masks). Pinhole, square pixels, principal
            point at the image centre; the focal length is a hypothesis tested in check_geometry.py:
              "H"  f = 0.5 * 720 / tan(camera_angle_x / 2)  (the PhysGaia MoSca loader, use_width_for_focal=False)
              "W"  f = 0.5 * 640 / tan(camera_angle_x / 2)  (the NeRF-synthetic reading of camera_angle_x)
  particles particles/<object>/particles_frame_XXXX.json (frame 1..240), converted by particles_to_npy.py to
            data/physgaia/_particles/<scene>/<object>.npy, (240, N, 3) float32, world frame, Y up, metres (free fall
            fits g = 9.80-9.81 m/s^2 at 24 fps in visco_bouncing_balls).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import os  # noqa: E402

DATA = Path(os.environ.get("PHYSGAIA_ROOT", "data/physgaia"))  # extracted scenes and data/physgaia/_particles
W_IMG, H_IMG = 640, 720
FPS = 24


def focal(angle_x: float, hyp: str = "H") -> float:
    return 0.5 * (H_IMG if hyp == "H" else W_IMG) / np.tan(angle_x / 2)


def load_cams(scene: str):
    """{view: (240, 4, 4) c2w}, {view: split}, camera_angle_x. Training views from the multi-view file if present."""
    base = DATA / scene
    cams, split = {}, {}
    trains = [base / "camera_info_train_multi.json", base / "camera_info_train.json"]
    angle = None
    for fn, sp in ((next(p for p in trains if p.exists()), "train"), (base / "camera_info_test.json", "test")):
        cj = json.load(open(fn))
        angle = cj["camera_angle_x"] if angle is None else angle
        assert abs(cj["camera_angle_x"] - angle) < 1e-12
        per = {}
        for fr in cj["frames"]:
            v, t = Path(fr["file_path"]).name.split("_")
            per.setdefault(int(v), {})[int(t)] = np.array(fr["transform_matrix"], float)
        for v, d in per.items():
            assert sorted(d) == list(range(1, len(d) + 1)), f"{fn} view {v}: frames not 1..T"
            cams[v] = np.stack([d[t] for t in sorted(d)])
            split[v] = sp
    return cams, split, angle


def image(scene: str, view: int, frame: int, split: str) -> np.ndarray:
    """RGB uint8 (720, 640, 3) of 1-based frame."""
    return np.asarray(Image.open(DATA / scene / "render" / split / f"{view}_{frame:03d}.png").convert("RGB"))


def project(P: np.ndarray, c2w: np.ndarray, f: float, W: int = W_IMG, H: int = H_IMG, cx=None, cy=None):
    """World points -> (u, v) continuous pixel coordinates (pixel i covers [i, i+1), centre i + 0.5) and planar depth z
    (> 0 in front). OpenGL camera: x right, y up, looking along -z."""
    cx = W / 2 if cx is None else cx
    cy = H / 2 if cy is None else cy
    Xc = (P - c2w[:3, 3]) @ c2w[:3, :3]
    z = -Xc[:, 2]
    zs = np.where(z > 1e-6, z, 1e-6)
    return f * Xc[:, 0] / zs + cx, -f * Xc[:, 1] / zs + cy, z


def zbuffer(u, v, z, W, H, labels=None, radius=0):
    """Nearest-point depth image (inf where empty) and the label of the nearest point (-1 where empty); every point is
    splatted to a (2 radius + 1)^2 pixel square."""
    ok = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    ui, vi, zz = np.floor(u[ok]).astype(np.int64), np.floor(v[ok]).astype(np.int64), z[ok]
    lab = labels[ok] if labels is not None else np.zeros(len(zz), np.int64)
    zb = np.full(H * W, np.inf)
    lb = np.full(H * W, -1, np.int64)
    LIN, ZZ, LB = [], [], []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            uu, vv = ui + dx, vi + dy
            m = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            LIN.append(vv[m] * W + uu[m])
            ZZ.append(zz[m])
            LB.append(lab[m])
    lin, zz, lab = np.concatenate(LIN), np.concatenate(ZZ), np.concatenate(LB)
    np.minimum.at(zb, lin, zz)
    win = zz <= zb[lin]  # the nearest point(s) of each pixel
    lb[lin[win]] = lab[win]
    return zb.reshape(H, W), lb.reshape(H, W)


def surface_mask(P: np.ndarray, cell: float) -> np.ndarray:
    """Particles on the boundary of a solid particle cloud (MPM particles fill the volume): occupancy at `cell` (>= ~3x the
    particle spacing so the interior has no holes), holes filled, eroded once; a particle is on the surface if its cell is
    occupied but not interior."""
    from scipy import ndimage
    lo = P.min(0) - 2 * cell
    q = np.floor((P - lo) / cell).astype(np.int64)
    G = q.max(0) + 3
    occ = np.zeros(G, bool)
    occ[tuple(q.T)] = True
    solid = ndimage.binary_fill_holes(occ)
    interior = ndimage.binary_erosion(solid, iterations=1)
    return ~interior[tuple(q.T)]


def lathe_dense(P: np.ndarray, centre_y: float, step: float = 0.01, dtheta: float = 1.0):
    """Dense outer surface of an axisymmetric static object about the world y axis through x = z = 0 (the
    visco_bouncing_balls glass bowl ships as ~7.9k mesh vertices, 5 deg apart on the equator: holes at 560 px).
    Meridian = per dtheta-degree bin of the polar angle around (r = 0, y = centre_y), the vertex farthest from that
    centre (outer wall); resampled every `step` along the meridian and swept around the axis every `step` of arc.
    Returns (dense points, per-vertex radial residual |r - r_meridian(angle)| in units)."""
    r, y = np.hypot(P[:, 0], P[:, 2]), P[:, 1]
    th = np.degrees(np.arctan2(y - centre_y, r))
    rho = np.hypot(r, y - centre_y)
    bins = np.floor(th / dtheta).astype(int)
    mer = []
    for b in np.unique(bins):
        m = bins == b
        i = np.flatnonzero(m)[np.argmax(rho[m])]
        mer.append((th[i], r[i], y[i]))
    mer = np.array(sorted(mer))
    seg = np.hypot(np.diff(mer[:, 1]), np.diff(mer[:, 2]))
    s = np.concatenate([[0], np.cumsum(seg)])
    ss = np.arange(0, s[-1], step)
    rr, yy = np.interp(ss, s, mer[:, 1]), np.interp(ss, s, mer[:, 2])
    pts = []
    for ri, yi in zip(rr, yy):
        n = max(8, int(np.ceil(2 * np.pi * ri / step)))
        a = np.arange(n) * 2 * np.pi / n
        pts.append(np.stack([ri * np.cos(a), np.full(n, yi), ri * np.sin(a)], 1))
    resid = np.abs(r - np.interp(th, mer[:, 0], mer[:, 1]))
    return np.concatenate(pts).astype(np.float32), resid
