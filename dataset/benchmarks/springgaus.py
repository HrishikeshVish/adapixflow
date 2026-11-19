"""One Spring-Gaus (Zhong et al., ECCV 2024) mpm_synthetic scene -> our cache layout (read by adapixflow/fusion.py).

Sources (read as data only; nothing from the dataset is executed):
  <--sim>/<scene>/NNN.ply                                                    MPM particles (x y z red green blue), 30 frames,
                                                                          particle order fixed over frames
  <--render>/<scene>/                                                     camera_XXX/NNN.png (512 x 512 RGBA, Blender;
                                                                          alpha = object, no ground or background drawn),
                                                                          camera.json (c2w Blender/OpenGL, K), frame.json
                                                                          (t = 0.03 s * frame), physical.json (E, nu, ground)
Output <--out>/<scene>[_r2].npz with the PhysInOne cache keys:
  depth (C, T, W, W) float16   planar metres along camera -z: object = the nearest GT particle (3 x 3 px z-buffer splat)
                               inside the rendered alpha mask (alpha pixels without a particle hit take the nearest hit);
                               ground (optional) = the analytic plane of physical.json's ground BC, one-sided (seen only
                               by cameras above it, as the renders: the two cameras below the plane see the object through
                               it); else BG_DEPTH (>= 1e3: invalid for the fusion, and rays pass through for UNSEEN)
  seg (C, T, W, W) uint8       OBJ_LABEL = 200 (object, alpha > 127), GROUND_LABEL = 1, 0 elsewhere
  c2w (C, T, 4, 4) float64     OpenGL, in OUR world frame (z up): x' = x - 0.5, y' = -(z - 0.5), z' = y - ground_y
                               (a proper rotation about x; gravity -y -> -z as PhysInOne, ground at z' = 0)
  f, W                         711.11 px at the native W = 512 (principal point 256 = W / 2; pixel centres +0.5);
                               no resampling to 560 (every reader takes f and W from the file)
  rgb_jpeg, rgb_offsets        RGBA composited over flat grey BG_RGB (JPEG q95), offsets per (view, frame) c * T + t
  meta (JSON)                  scene, fps, frames, cameras, moving_labels_override [200], world transform, ground, physics,
                               frame times, latent convention, sources
  gt_pts (T, n, 3) float32     GT particles in our world frame (a fixed random subset of n <= --npts, same particles every
  gt_idx (n,) int64            frame: correspondences preserved), and their indices in the PLY
Usage: python dataset/benchmarks/springgaus.py <scene> --repeat 2 [--sim <dir>] [--render <dir>] [--out data/springgaus/cache]
  --repeat 2 writes every native frame twice, so that the fusion's step k (video frame 2k) is native frame k (the paper's
  setting); default 1 keeps the native frames (step k = frame 2k, the two-frames-per-step stress setting).
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plyio import read_ply  # noqa: E402


def _arg(k, d):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d


SIM = Path(_arg("--sim", "data/springgaus/raw/mpm_synthetic/simulation"))   # NNN.ply MPM particles per scene
REN = Path(_arg("--render", "data/springgaus/raw/mpm_synthetic/render"))    # camera_XXX/NNN.png renders per scene
OBJ_LABEL, GROUND_LABEL = 200, 1
BG_DEPTH = 6.0e4          # float16-finite, > 1e3 (invalid for unproject; UNSEEN rays pass)
BG_RGB = 128
GROUND_HALF = 2.0         # ground plane drawn for |x'|, |y'| <= 2 m


def arg(k, d, c=str):
    return c(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d


def world_transform(ground_y):
    """4x4 A: Spring-Gaus world (y up, ground y = ground_y, object near x = z = 0.5) -> ours (z up, ground z = 0)."""
    A = np.eye(4)
    A[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    A[:3, 3] = [-0.5, 0.5, -ground_y]
    return A


def project(X, M, f, W):
    """World points (n, 3) -> (u, v, z) for an OpenGL c2w M: u = f x / z + W / 2, v = -f y / z + W / 2 (pixel index
    floor(u)), z planar along -z."""
    pc = (X - M[:3, 3]) @ M[:3, :3]
    z = -pc[:, 2]
    zs = np.maximum(z, 1e-9)
    return f * pc[:, 0] / zs + W / 2, -f * pc[:, 1] / zs + W / 2, z


def zbuffer(u, v, z, W, r=1):
    """Nearest depth per pixel from points splatted as (2r+1)^2 squares; np.inf where no point lands."""
    ui, vi = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    lins, zs = [], []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            uu, vv = ui + dx, vi + dy
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < W) & (z > 1e-6)
            lins.append(vv[ok] * W + uu[ok])
            zs.append(z[ok])
    lin, zz = np.concatenate(lins), np.concatenate(zs)
    out = np.full(W * W, np.inf)
    np.minimum.at(out, lin, zz)
    return out.reshape(W, W)


def plane_depth(M, f, W):
    """Planar depth of the one-sided plane z = 0 (|x|, |y| <= GROUND_HALF) seen from above; np.inf elsewhere."""
    if M[2, 3] <= 0:
        return np.full((W, W), np.inf)
    v, u = np.mgrid[0:W, 0:W] + 0.5
    d_cam = np.stack([(u - W / 2) / f, -(v - W / 2) / f, -np.ones_like(u)], -1)  # planar depth 1 along -z
    d = d_cam @ M[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -M[2, 3] / d[..., 2]
    P = M[:3, 3] + s[..., None] * d
    ok = (s > 0) & (np.abs(P[..., 0]) <= GROUND_HALF) & (np.abs(P[..., 1]) <= GROUND_HALF)
    return np.where(ok, s, np.inf)


def convert(scene, out_dir, npts=120000, ground=True, repeat=1):
    t0 = time.time()
    ren = REN / scene
    phys = json.load(open(ren / "physical.json"))
    ground_y = float(phys[1]["BC"]["GROUND"][0][1])
    assert phys[1]["BC"]["GROUND"][1] == [0, 1, 0] and phys[1]["GRAVITY"][1] < 0 == phys[1]["GRAVITY"][0] == phys[1]["GRAVITY"][2]
    times = [float(list(e.values())[0]) for e in json.load(open(ren / "frame.json"))]
    cams = json.load(open(ren / "camera.json"))
    C = len(cams)
    K = np.array(cams[0]["K"], np.float64)
    assert all(np.allclose(np.array(c["K"]), K) for c in cams) and K[0, 0] == K[1, 1]
    f, W = float(K[0, 0]), int(round(2 * K[0, 2]))
    assert abs(K[0, 2] - W / 2) < 1e-9 and abs(K[1, 2] - W / 2) < 1e-9, "principal point must be W / 2"
    A = world_transform(ground_y)
    c2w = np.stack([A @ np.array(c["c2w"], np.float64) for c in cams])  # (C, 4, 4), static cameras
    plies = sorted((SIM / scene).glob("*.ply"))
    T = len(plies)
    assert T == len(times) and [p.stem for p in plies] == [f"{t:03d}" for t in range(T)]
    rng = np.random.default_rng(0)
    gt_idx = gt_pts = None
    Tw = T * repeat
    depth = np.full((C, Tw, W, W), BG_DEPTH, np.float16)
    seg = np.zeros((C, Tw, W, W), np.uint8)
    jpegs = [[None] * Tw for _ in range(C)]
    planes = [plane_depth(c2w[c], f, W) if ground else np.full((W, W), np.inf) for c in range(C)]
    stats = {"alpha_px": 0, "alpha_px_filled_nearest": 0}
    for t, ply in enumerate(plies):
        props, a = read_ply(ply)
        X = a[:, :3].astype(np.float64) @ A[:3, :3].T + A[:3, 3]
        if gt_idx is None:
            n = len(X)
            gt_idx = np.sort(rng.choice(n, min(npts, n), replace=False))
            gt_pts = np.zeros((T, len(gt_idx), 3), np.float32)
            n_particles = n
        assert len(X) == n_particles
        gt_pts[t] = X[gt_idx]
        for c in range(C):
            rgba = np.asarray(Image.open(ren / f"camera_{c:03d}/{t:03d}.png").convert("RGBA"), np.float64)
            assert rgba.shape[:2] == (W, W)
            alpha = rgba[..., 3] / 255
            obj = alpha > 0.5
            zb = zbuffer(*project(X, c2w[c], f, W), W, r=1)
            miss = obj & ~np.isfinite(zb)
            if miss.any():  # alpha pixels no particle reached: nearest hit
                idx = ndimage.distance_transform_edt(~np.isfinite(zb), return_distances=False, return_indices=True)
                zb[miss] = zb[idx[0][miss], idx[1][miss]]
            stats["alpha_px"] += int(obj.sum())
            stats["alpha_px_filled_nearest"] += int(miss.sum())
            d = np.where(obj, zb, planes[c])
            s = np.where(obj, OBJ_LABEL, np.where(np.isfinite(planes[c]), GROUND_LABEL, 0)).astype(np.uint8)
            d = np.where(np.isfinite(d), d, BG_DEPTH)
            rgb = rgba[..., :3] * alpha[..., None] + BG_RGB * (1 - alpha[..., None])
            buf = io.BytesIO()
            Image.fromarray(np.round(rgb).clip(0, 255).astype(np.uint8)).save(buf, format="JPEG", quality=95)
            for r_ in range(repeat):
                depth[c, t * repeat + r_] = d.astype(np.float16)
                seg[c, t * repeat + r_] = s
                jpegs[c][t * repeat + r_] = buf.getvalue()
    flat = [jpegs[c][t] for c in range(C) for t in range(Tw)]
    offs = np.zeros(len(flat) + 1, np.int64)
    offs[1:] = np.cumsum([len(b) for b in flat])
    dt = times[1] - times[0]
    meta = {
        "scene": scene, "source": "Spring-Gaus mpm_synthetic (ECCV 2024), Google Drive release",
        "fps": round(repeat / dt, 4), "native_fps": round(1 / dt, 4), "frames": Tw, "native_frames": T, "repeat": repeat,
        "frame_times_s": times, "cameras": [c["camera"] for c in cams], "moving_labels_override": [OBJ_LABEL],
        "ground_label": GROUND_LABEL if ground else None, "ground": ("analytic one-sided plane z'=0 from physical.json BC, "
                                                                    "depth/seg only (not in the RGB)") if ground else None,
        "world_from_springgaus": A.tolist(), "ground_y_springgaus": ground_y, "physics": phys,
        "n_particles": int(n_particles), "bg_depth": BG_DEPTH, "bg_rgb": BG_RGB,
        "latent_convention": "fusion step k = cache frame 2k" + (" = native frame k" if repeat == 2 else " = native frame 2k"),
        "depth_source": "GT MPM particles, 3x3 px z-buffer splat, inside the Blender alpha mask", **stats,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{scene}{'_r' + str(repeat) if repeat != 1 else ''}.npz"
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, depth=depth, seg=seg, c2w=np.repeat(c2w[:, None], Tw, 1), f=np.float64(f), W=np.int64(W),
             rgb_jpeg=np.frombuffer(b"".join(flat), np.uint8), rgb_offsets=offs, meta=np.array(json.dumps(meta)),
             gt_pts=gt_pts, gt_idx=gt_idx)
    tmp.rename(out)
    print(f"ok {out}: C={C} T={Tw} W={W} f={f:.2f} particles={n_particles} (kept {len(gt_idx)}), "
          f"alpha px filled by nearest {stats['alpha_px_filled_nearest']}/{stats['alpha_px']}, "
          f"{out.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f} s", flush=True)
    return out


if __name__ == "__main__":
    convert(sys.argv[1], Path(arg("--out", "data/springgaus/cache")), arg("--npts", 120000, int), "--no-ground" not in sys.argv,
            arg("--repeat", 1, int))
