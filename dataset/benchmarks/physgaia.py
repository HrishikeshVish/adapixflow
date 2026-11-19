"""PhysGaia scene -> our cache layout (as dataset/fetch_physinone.py writes for PhysInOne), so that the fusion, the
evaluation and the baseline adapters can read it, plus GT object points from the particles.

PhysGaia ships RGB only (no depth, no masks; alpha = 255). Depth and labels are therefore RENDERED here from the GT
geometry (a particle z-buffer), which makes this a GT-geometry cache, not a renderer's depth pass:
  moving objects  GT particles (visco_bouncing_balls: 5 balls, 166,608 MPM particles), each ball its own label 128 + i
                  (meta moving_labels_override, read by adapixflow.fusion.moving_labels)
  static          floor = plane y = 0 (checked: check_geometry floor test), label 0, out to |x|, |z| <= FLOOR_EXTENT;
                  bowl (glass, transparent in the renders): its mesh vertices (particles/bowl, static) densified to the outer wall of
                  the surface of revolution (pg.lathe_dense; the shell is ~10 cm thick in PhysGaia units), label 1, --bowl:
                    xray (default)  bowl surface only where no ball is behind it in that view (the RGB shows the balls
                                    through the glass, so ball pixels keep the ball's depth); the union over views gives
                                    the bowl wall and the balls in one lift
                    transparent     no bowl in depth (depth = what is behind the glass)
                    opaque          plain z-buffer (balls inside the bowl hidden)
                  anything else (grey backdrop): depth 0 = invalid (the fusion keeps 0 < z < 1e3)
Conventions of the output (as dataset/fetch_physinone.py): square W = 560 px; PhysGaia's 640 x 720 frames are
padded with black to 720 x 720 (40 px left and right, principal point stays at the centre) and BOX-resized to 560;
f = 0.5 * 720 / tan(camera_angle_x / 2) * 560 / 720 (check_geometry.py: IoU 0.93 vs 0.53 for the width reading);
depth planar metres float16; c2w (C, T, 4, 4) OpenGL; pixel centres at +0.5. World frame: PhysGaia's Y-up frame rotated
to Z-up (x, y, z) -> (x, -z, y) (PhysInOne and our model: gravity along -z) and scaled by --scale (metres per PhysGaia
unit; PhysGaia units are metres: g = 9.81 at 24 fps). Views: cache 0, 1 = PhysGaia test views 1, 2 (static, held out as
our novel views, as PhysGaia's own test split); cache 2.. = training views 0, 3, 4, 5 (moving spiral paths).
Time: latent step k = cache frame 2k = PhysGaia frame first + k * step (--step PhysGaia frames per latent step; odd cache
frames = first + k * step + step // 2, so --step 1 duplicates frames); T_cache = 2 * n_latent.

GT object points (<name>_gtpoints.npz, our world frame, metres): per latent step k (cache frame 2k): every ball
particle surface point (pg.surface_mask at 3 cm PhysGaia units; MPM particles fill the volume), with its ball id, and
whether some cache view sees it (particle z-buffer of the balls, tolerance 5 cm PhysGaia units); voxelise with a lift's
lo / VOX to compare with lifted object voxels (gt_compare.py).
Usage: python -I convert_cache.py [scene] [--scale 0.1] [--step 2] [--first 1] [--n-latent 45] [--bowl xray]
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pg_common as pg  # noqa: E402

RES = 560
FLOOR_EXTENT = 25.0  # PhysGaia units; the rendered checker floor reaches the horizon in most views (extent not given)
ROT = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # Y-up -> Z-up


def arg(k, d, c=float):
    return c(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d


def to_ours(P, s):
    return (P @ ROT.T) * s


def c2w_ours(M, s):
    out = np.eye(4)
    out[:3, :3] = ROT @ M[:3, :3]
    out[:3, 3] = s * (ROT @ M[:3, 3])
    return out


def pad_resize(img):
    H, W = img.shape[:2]
    S = max(H, W)
    canvas = np.zeros((S, S, 3), np.uint8)
    y0, x0 = (S - H) // 2, (S - W) // 2
    canvas[y0:y0 + H, x0:x0 + W] = img
    return np.asarray(Image.fromarray(canvas).resize((RES, RES), Image.BOX))


def floor_depth(M, f):
    """Planar depth of the plane y = 0 (PhysGaia frame) at RES x RES pixel centres; 0 where missed / beyond extent."""
    v, u = np.mgrid[0:RES, 0:RES] + 0.5
    d = np.stack([(u - RES / 2) / f, -(v - RES / 2) / f, -np.ones_like(u)], -1)  # camera ray with planar z = 1
    dw = d @ M[:3, :3].T
    o = M[:3, 3]
    t = -o[1] / np.where(np.abs(dw[..., 1]) > 1e-9, dw[..., 1], 1e-9)
    P = o + t[..., None] * dw
    ok = (t > 0) & (np.abs(P[..., 0]) <= FLOOR_EXTENT) & (np.abs(P[..., 2]) <= FLOOR_EXTENT)
    return np.where(ok, t, 0.0)  # t is the planar depth because the ray has camera z = -1 per unit t


def main():
    scene = next((a for a in sys.argv[1:] if not a.startswith("--") and not a.replace(".", "").isdigit()), "visco_bouncing_balls")
    s, step, first = arg("--scale", 0.1), arg("--step", 2, int), arg("--first", 1, int)
    n_lat, bowl_mode = arg("--n-latent", 45, int), arg("--bowl", "xray", str)
    name = arg("--name", f"{scene}__s{s:g}_n{step}", str)
    out_dir = Path(arg("--out", str(pg.DATA / "cache"), str))
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cams, split, angle = pg.load_cams(scene)
    order = sorted(v for v in cams if split[v] == "test") + sorted(v for v in cams if split[v] == "train")
    f_nat = pg.focal(angle, "H")
    f = f_nat * RES / max(pg.W_IMG, pg.H_IMG)
    pdir = pg.DATA / "_particles" / scene
    B = np.load(pdir / "balls.npy", mmap_mode="r")
    comp = np.load(pdir / "balls_component.npy").astype(np.int64)
    bowl = None
    if (pdir / "bowl.npy").exists():  # static (checked: max displacement 0); mesh vertices 5 deg apart -> dense lathed outer wall
        bw = np.load(pdir / "bowl.npy", mmap_mode="r")
        assert float(np.abs(np.asarray(bw) - np.asarray(bw[0])).max()) == 0.0
        bowl, _ = pg.lathe_dense(np.asarray(bw[0]), centre_y=0.5 * float(np.asarray(bw[0])[:, 1].max()), step=0.01, dtheta=1.0)
    n_cache = 2 * n_lat
    x0 = (max(pg.W_IMG, pg.H_IMG) - pg.W_IMG) / 2 * RES / max(pg.W_IMG, pg.H_IMG)  # native columns span [x0, RES - x0)
    native_cols = ((np.arange(RES) + 0.5) >= x0) & ((np.arange(RES) + 0.5) < RES - x0)
    pg_frame = [first + (j // 2) * step + (j % 2) * (step // 2) for j in range(n_cache)]
    assert max(pg_frame) <= B.shape[0], f"frames up to {max(pg_frame)} > {B.shape[0]}"
    C = len(order)
    depth = np.zeros((C, n_cache, RES, RES), np.float16)
    seg = np.zeros((C, n_cache, RES, RES), np.uint8)
    c2w = np.zeros((C, n_cache, 4, 4))
    jpeg, offs = [], [0]
    rend_cache = {}
    gt = {"xyz": [], "ball": [], "seen_any": [], "seen_train": [], "seen_test": [], "off": [0]}
    for ci, v in enumerate(order):
        for j, t in enumerate(pg_frame):
            M = cams[v][t - 1]
            c2w[ci, j] = c2w_ours(M, s)
            key = (v, t)
            if key not in rend_cache:
                P = np.asarray(B[t - 1])
                u, vv, z = pg.project(P, M, f, RES, RES)
                zb, lb = pg.zbuffer(u, vv, z, RES, RES, labels=comp, radius=1)
                fl = floor_depth(M, f)
                D = np.where(np.isfinite(zb), zb, 0.0)
                L = np.where(lb >= 0, 128 + lb, 0)
                fl_front = (fl > 0) & ((D == 0) | (fl < D))  # floor in front of a ball (never, here) or where no ball
                D = np.where(fl_front, fl, D)
                L = np.where(fl_front, 0, L)
                if bowl is not None and bowl_mode != "transparent":
                    ub, vb, zbw = pg.project(bowl, M, f, RES, RES)
                    zbb, _ = pg.zbuffer(ub, vb, zbw, RES, RES, radius=1)
                    hit = np.isfinite(zbb) & (zbb < np.where(D > 0, D, np.inf))
                    if bowl_mode == "xray":
                        hit &= lb < 0  # balls seen through the glass keep their depth
                    D = np.where(hit, zbb, D)
                    L = np.where(hit, 1, L)
                D[:, ~native_cols] = 0  # the black pad: no image there, so no depth / labels either
                L[:, ~native_cols] = 0
                img = pad_resize(pg.image(scene, v, t, split[v]))
                buf = io.BytesIO()
                Image.fromarray(img).save(buf, format="JPEG", quality=92)
                rend_cache = {key: (D * s, L, np.frombuffer(buf.getvalue(), np.uint8))}  # one entry (frames repeat only consecutively)
            D, L, jb = rend_cache[key]
            depth[ci, j], seg[ci, j] = D.astype(np.float16), L.astype(np.uint8)
            jpeg.append(jb)
            offs.append(offs[-1] + len(jb))
        print(f"  view {v} -> cache {ci} ({split[v]}): {n_cache} frames ({time.time() - t0:.0f} s)", flush=True)

    # GT object points per latent step: ball surface particles, visibility per cache view (balls-only z-buffer)
    for k in range(n_lat):
        t = pg_frame[2 * k]
        P = np.asarray(B[t - 1])
        sm = pg.surface_mask(P, 0.03)
        Ps, bs = P[sm], comp[sm]
        vis = np.zeros((C, len(Ps)), bool)
        for ci, v in enumerate(order):
            u, vv, z = pg.project(Ps, cams[v][t - 1], f, RES, RES)
            zb, _ = pg.zbuffer(u, vv, z, RES, RES, radius=1)
            ok = (z > 0) & (u >= x0) & (u < RES - x0) & (vv >= 0) & (vv < RES)
            vis[ci, ok] = z[ok] <= zb[np.floor(vv[ok]).astype(int), np.floor(u[ok]).astype(int)] + 0.05
        gt["xyz"].append(to_ours(Ps, s).astype(np.float32))
        gt["ball"].append(bs.astype(np.int8))
        gt["seen_any"].append(vis.any(0))
        gt["seen_train"].append(vis[2:].any(0))
        gt["seen_test"].append(vis[:2].any(0))
        gt["off"].append(gt["off"][-1] + len(Ps))
    meta = {"scene": name, "dataset": "PhysGaia", "physgaia_scene": scene, "source": "https://huggingface.co/datasets/mijeongkim/PhysGaia",
            "licence": "CC BY-NC 4.0", "fps": pg.FPS, "frames": n_cache, "latent_step_physgaia_frames": step,
            "latent_fps": pg.FPS / step, "cache_frame_to_physgaia_frame": pg_frame, "cameras": [str(v) for v in order],
            "camera_split": [split[v] for v in order], "holdout_views_are_physgaia_test_views": [0, 1],
            "moving_cameras": [ci for ci, v in enumerate(order) if split[v] == "train"],
            "scale_m_per_unit": s, "world_rotation_yup_to_zup": ROT.tolist(), "gravity_axis": "-z",
            "f_hypothesis": "0.5 * H / tan(camera_angle_x / 2), H = 720 (checked)", "f_native": f_nat, "pad": "640x720 -> 720x720 black, BOX resize to 560; depth / seg 0 outside the native columns",
            "bowl_mode": bowl_mode, "floor": "plane y = 0 (PhysGaia frame), |x|,|z| <= %g units" % FLOOR_EXTENT,
            "labels": {"0": "floor / background", "1": "bowl (glass, static)", **{str(128 + i): f"ball {i}" for i in range(int(comp.max()) + 1)}},
            "moving_labels_override": [128 + i for i in range(int(comp.max()) + 1)], "source_res": [pg.W_IMG, pg.H_IMG],
            "depth_source": "rendered from GT particles + floor plane (PhysGaia has no depth)", "recorder_stats": None}
    tmp = out_dir / f"{name}.tmp.npz"
    np.savez_compressed(tmp, rgb_jpeg=np.concatenate(jpeg), rgb_offsets=np.array(offs, np.int64), depth=depth, seg=seg, c2w=c2w,
                        f=np.float64(f), W=np.int64(RES), meta=np.array(json.dumps(meta)))
    tmp.rename(out_dir / f"{name}.npz")
    np.savez_compressed(out_dir / f"{name}_gtpoints.npz", xyz=np.concatenate(gt["xyz"]), ball=np.concatenate(gt["ball"]),
                        seen_any=np.concatenate(gt["seen_any"]), seen_train=np.concatenate(gt["seen_train"]),
                        seen_test=np.concatenate(gt["seen_test"]), off=np.array(gt["off"], np.int64),
                        latent_to_physgaia_frame=np.array(pg_frame[::2]), scale=np.float64(s))
    sz = (out_dir / f"{name}.npz").stat().st_size
    print(f"wrote {out_dir / name}.npz ({sz / 1e6:.0f} MB): {C} views x {n_cache} cache frames ({n_lat} latent steps, PhysGaia frames "
          f"{pg_frame[0]}..{pg_frame[-1]}), f {f:.2f} px at {RES}; {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
