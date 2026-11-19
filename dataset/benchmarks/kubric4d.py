"""One Kubric-4D scene (GCD, Van Hoorick et al. ECCV 2024; 16 synchronized static cameras rendered with Kubric) -> our cache
layout, plus a GT geometry check. Extracted raw Kubric worker output per camera v: frames_p0_v<v>/{rgba,depth,segmentation}
_<t:05d>.{png,tiff,png} and scn<id>_p0_v<v>.json (camera K / positions / quaternions, instances with poses).

Cache (data/kubric4d/cache/<scene>.npz, the PhysInOne layout): depth (16, T, 576, 576) float16 PLANAR depth (Kubric's
depth is the distance from the camera centre; converted), seg (16, T, 576, 576) uint8 (instance index + 1; the instance
order is the insertion order, identical in every camera's JSON), c2w (16, T, 4, 4) OpenGL, f = 576 px, W = 576. The
576 x 384 frames are padded to 576 x 576 (96 rows above and below; depth 0 = invalid, seg 0, black RGB), which keeps the
principal point at W / 2 as our unprojection assumes (pixels are square: K = [[1, 0, -.5], [0, -1.5, -.5]] normalized by
width / height). meta: moving_labels_override = instances that move > 0.05 units or rotate > 5 degrees.
GT check (<scene>_check.json):
  multiview   back-project view a (every 4th pixel) and project into view b at the same frame; |z_proj - z_b| for the
              points that land on b's image: inlier fraction (|dz| < 1 % of z) and inlier median, for the planar conversion
              and for the raw depth taken as planar (tests the depth convention and the 16 cameras' poses together)
  in-box      fraction of each instance's pixels (all views) inside its 3D box (bboxes_3d, +2 pixel footprints)
  floor       |world z| of background pixels within 6 units of the origin
Options (the paper's caches use --dup 2 --compress --moving d4 --suffix _f1):
  --dup N          frame duplication: cache video frame t' = native frame t' // N (depth, seg, c2w, JPEG bytes repeated),
                   so the fusion's step k (video frame 2k) = native frame k at N = 2
  --compress       np.savez_compressed (zlib) instead of np.savez
  --moving d4      moving_labels_override = D4 set: tossed (is_dynamic) instances + instances whose 3D-box corners move
                   > --box-thr (0.5) units from frame 0 at any frame; the inventory rule (> 0.05 units or 5 deg) is kept
                   in meta as moving_labels_inventory_rule
  --suffix S       cache name k4d_<scn><S> (e.g. _f1)
  --no-check       skip the GT geometry check (it does not depend on --dup / --moving except the in-box instance set)
Usage: python -I dataset/benchmarks/kubric4d.py <scene dir> --out data/kubric4d/cache --dup 2 --compress --moving d4 --suffix _f1
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import movi  # noqa: E402


def arg(k, d, c=str):
    return c(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d


def main():
    src = Path(sys.argv[1]).resolve()
    out_dir = Path(arg("--out", "data/kubric4d/cache"))
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sid = src.name
    views = sorted(int(p.name.split("_v")[1]) for p in src.glob("frames_p0_v*"))
    J = {v: json.load(open(src / f"{sid}_p0_v{v}.json")) for v in views}
    j0 = J[views[0]]
    T = int(j0["scene"]["num_frames"])
    Wd, Hd = (int(x) for x in j0["scene"]["resolution"])
    C, W, pad = len(views), Wd, (Wd - Hd) // 2
    K = np.array(j0["camera"]["K"])
    f = K[0, 0] * Wd
    assert abs(-K[1, 1] * Hd - f) < 1e-6, "non-square pixels"
    ins = j0["instances"]
    for v in views:  # same instances, same order in every camera
        assert [x["asset_id"] for x in J[v]["instances"]] == [x["asset_id"] for x in ins]
    depth = np.zeros((C, T, W, W), np.float16)
    seg = np.zeros((C, T, W, W), np.uint8)
    c2w = np.tile(np.eye(4), (C, T, 1, 1))
    jpeg, offs = [], [0]
    raw_keep = {}
    for ci, v in enumerate(views):
        cam = J[v]["camera"]
        c2w[ci, :, :3, :3] = movi.quat_to_R(np.array(cam["quaternions"]))
        c2w[ci, :, :3, 3] = np.array(cam["positions"])
        fd = src / f"frames_p0_v{v}"
        for t in range(T):
            d = np.asarray(Image.open(fd / f"depth_{t:05d}.tiff"), np.float64)
            z = movi.euclid_to_planar(d, f, Wd, Hd)
            depth[ci, t, pad:pad + Hd] = z
            seg[ci, t, pad:pad + Hd] = np.asarray(Image.open(fd / f"segmentation_{t:05d}.png"))
            rgb = np.zeros((W, W, 3), np.uint8)
            rgb[pad:pad + Hd] = np.asarray(Image.open(fd / f"rgba_{t:05d}.png").convert("RGB"))
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, format="JPEG", quality=92)
            jpeg.append(np.frombuffer(buf.getvalue(), np.uint8))
            offs.append(offs[-1] + len(jpeg[-1]))
            if t in (0, T // 2, T - 1):
                raw_keep[(ci, t)] = d
    pos = np.array([x["positions"] for x in ins])  # (K, T, 3)
    R = movi.quat_to_R(np.array([x["quaternions"] for x in ins]))
    disp = np.linalg.norm(pos - pos[:, :1], axis=2).max(1)
    rot = np.degrees(np.arccos(np.clip((np.einsum("ktij,kij->kt", R, R[:, 0]) - 1) / 2, -1, 1))).max(1)
    moving_inv = (disp > 0.05) | (rot > 5.0)
    bb = np.array([x["bboxes_3d"] for x in ins], np.float64)  # (K, T, 8, 3)
    box_disp = np.linalg.norm(bb - bb[:, :1], axis=3).max((1, 2))
    tossed = np.array([bool(x["is_dynamic"]) for x in ins])
    box_thr = arg("--box-thr", 0.5, float)
    moving_d4 = tossed | (box_disp > box_thr)
    rule = arg("--moving", "inventory")
    moving = moving_d4 if rule == "d4" else moving_inv
    labels = [int(k + 1) for k in np.flatnonzero(moving)]
    dup = arg("--dup", 1, int)
    scene = f"k4d_{sid}{arg('--suffix', '')}"
    meta = {"scene": scene, "dataset": "kubric-4d (GCD)", "source": src.name, "fps": int(j0["scene"]["frame_rate"]),
            "frames": T, "cameras": [str(v) for v in views], "source_res": [Wd, Hd], "pad_rows": pad, "static_camera": True,
            "units": "Kubric scene units (gravity 10 units/s^2 along -z)", "depth": "planar z (converted), 0 in the padding",
            "moving_labels_override": labels, "num_instances": len(ins), "asset_id": [x["asset_id"] for x in ins],
            "is_dynamic": [bool(x["is_dynamic"]) for x in ins], "instance_max_disp": disp.round(3).tolist(),
            "instance_max_rot_deg": rot.round(1).tolist(), "background": j0["scene"].get("background"),
            "moving_rule": rule, "moving_labels_inventory_rule": [int(k + 1) for k in np.flatnonzero(moving_inv)],
            "moving_labels_d4": [int(k + 1) for k in np.flatnonzero(moving_d4)], "d4_box_thr_units": box_thr,
            "instance_box_corner_max_disp": box_disp.round(3).tolist(), "tossed_labels": [int(k + 1) for k in np.flatnonzero(tossed)],
            "frame_dup": dup, "native_frames": T, "native_fps": int(j0["scene"]["frame_rate"])}
    if dup > 1:  # cache video frame t' = native frame t' // dup
        meta.update(frames=T * dup, fps=int(j0["scene"]["frame_rate"]) * dup, latent_fps=int(j0["scene"]["frame_rate"]) * dup / 2,
                    frame_dup_note=f"cache video frame t' = native frame t' // {dup}")
        src_t = np.repeat(np.arange(T), dup)
        jpeg_d, offs_d = [], [0]
        for ci in range(C):
            for t in src_t:
                jpeg_d.append(jpeg[ci * T + t])
                offs_d.append(offs_d[-1] + len(jpeg_d[-1]))
        arrs = dict(rgb_jpeg=np.concatenate(jpeg_d), rgb_offsets=np.array(offs_d, np.int64), depth=np.repeat(depth, dup, axis=1),
                    seg=np.repeat(seg, dup, axis=1), c2w=np.repeat(c2w, dup, axis=1))
    else:
        arrs = dict(rgb_jpeg=np.concatenate(jpeg), rgb_offsets=np.array(offs, np.int64), depth=depth, seg=seg, c2w=c2w)
    save = np.savez_compressed if "--compress" in sys.argv else np.savez
    save(out_dir / f"{scene}.npz", **arrs, f=np.float64(f), W=np.int64(W), meta=np.array(json.dumps(meta)))
    del arrs
    np.savez(out_dir / f"{scene}_gt.npz", positions=pos, quaternions_wxyz=np.array([x["quaternions"] for x in ins]),
             velocities=np.array([x["velocities"] for x in ins]), bboxes_3d=np.array([x["bboxes_3d"] for x in ins]),
             moving=moving, mass=np.array([x["mass"] for x in ins]), friction=np.array([x["friction"] for x in ins]),
             restitution=np.array([x["restitution"] for x in ins]), moving_inventory_rule=moving_inv, tossed=tossed,
             box_corner_max_disp=box_disp)
    t_cache = time.time() - t0
    if "--no-check" in sys.argv:
        print(json.dumps({"scene": scene, "moving_labels": labels, "seconds_cache": round(t_cache, 1)}))
        return

    # --- GT check ---
    v_, u_ = np.mgrid[0:W:4, 0:W:4] + 0.5
    mv = {"planar": [], "raw": []}
    for t in (0, T // 2, T - 1):
        for a in range(C):
            for b in range(C):
                if a == b:
                    continue
                for kind in ("planar", "raw"):
                    if kind == "planar":
                        za = depth[a, t, ::4, ::4].astype(np.float64)
                    else:
                        za = np.zeros((W, W))
                        za[pad:pad + Hd] = raw_keep[(a, t)]
                        za = za[::4, ::4]
                    ok = za > 0
                    pc = np.stack([(u_ - W / 2) / f * za, -(v_ - W / 2) / f * za, -za], -1)[ok]
                    P = pc @ c2w[a, t, :3, :3].T + c2w[a, t, :3, 3]
                    uv, zp = movi.project(P, c2w[b, t], f, W)
                    iu, iv = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
                    inb = (zp > 0) & (iu >= 0) & (iu < W) & (iv >= pad) & (iv < pad + Hd)
                    if kind == "planar":
                        zb = depth[b, t].astype(np.float64)[iv[inb], iu[inb]]
                    else:
                        zr = np.zeros((W, W))
                        zr[pad:pad + Hd] = raw_keep[(b, t)]
                        zb = zr[iv[inb], iu[inb]]
                    dz = np.abs(zp[inb] - zb)
                    rel = dz / zb
                    inl = rel < 0.01
                    mv[kind].append([inl.mean(), np.median(dz[inl]) if inl.any() else np.nan, np.median(rel)])
    inbox, floor = [], []
    for ci in range(C):
        for t in (0, T // 2, T - 1):
            z = depth[ci, t].astype(np.float64)
            for k in np.flatnonzero(moving):
                m = seg[ci, t] == k + 1
                if m.sum() < 20:
                    continue
                P = movi.unproject(z, c2w[ci, t], f, W, m)
                fp = np.median(z[m]) / f
                B = np.array(ins[k]["bboxes_3d"][t])
                ax = np.stack([B[4] - B[0], B[2] - B[0], B[1] - B[0]])
                ln = np.linalg.norm(ax, axis=1)
                q = (P - B[0]) @ (ax / ln[:, None]).T
                inbox.append(np.all((q >= -2 * fp) & (q <= ln + 2 * fp), 1).mean())
            m = (seg[ci, t] == 0) & (z > 0)
            P = movi.unproject(z, c2w[ci, t], f, W, m)
            floor.append(np.abs(P[np.linalg.norm(P[:, :2], axis=1) < 6, 2]))
    floor = np.concatenate(floor)
    ext = np.array([np.ptp(np.array(x["bboxes_3d"][0]), axis=0).max() for x in ins])
    step = np.linalg.norm(np.diff(pos, axis=1), axis=2)
    chk = {"scene": scene, "views": views, "T": T, "fps": meta["fps"], "W": W, "H_source": Hd, "f_px": f,
           "cam_dist": [float(np.linalg.norm(c2w[ci, 0, :3, 3])) for ci in range(C)],
           "cam_height": [float(c2w[ci, 0, 2, 3]) for ci in range(C)],
           "multiview_planar[inlier_frac,inlier_median_dz,median_rel_dz]": np.nanmedian(np.array(mv["planar"]), 0).round(5).tolist(),
           "multiview_RAWdepth_as_planar[inlier_frac,inlier_median_dz,median_rel_dz]": np.nanmedian(np.array(mv["raw"]), 0).round(5).tolist(),
           "inbox_frac_median_min": [float(np.median(inbox)), float(np.min(inbox))], "n_inbox_tests": len(inbox),
           "floor_abs_z_median_p95": [float(np.median(floor)), float(np.percentile(floor, 95))],
           "moving_labels": labels, "instance_max_disp": meta["instance_max_disp"], "is_dynamic": meta["is_dynamic"],
           "obj_extent_max_axis_frame0": ext.round(3).tolist(),
           "median_step_per_frame_moving": [float(np.median(s[s > 1e-3])) if (s > 1e-3).any() else 0.0 for s in step[moving]],
           "max_speed_moving": [float(np.linalg.norm(np.array(ins[k]["velocities"]), axis=1).max()) for k in np.flatnonzero(moving)],
           "seconds_cache": round(t_cache, 1), "seconds_total": round(time.time() - t0, 1)}
    (out_dir / f"{scene}_check.json").write_text(json.dumps(chk, indent=1))
    print(json.dumps(chk))


if __name__ == "__main__":
    main()
