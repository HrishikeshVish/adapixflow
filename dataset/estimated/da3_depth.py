"""Estimated depth for the estimated-input experiment: Depth Anything 3 (DA3, inference only), conditioned on our
intrinsics and poses. Code: https://github.com/ByteDance-Seed/Depth-Anything-3 (third_party/Depth-Anything-3, pinned),
weights: https://huggingface.co/depth-anything/DA3-GIANT-1.1 (downloaded on first use).

DA3 runs once per requested frame on all training views of that frame jointly (pose-conditioned mode). DA3 predicts its
own poses in a normalised frame. A Sim(3) Umeyama fit (camera centres, RANSAC for >= 10 views) of our poses onto them
gives the scale s, and depth / s is metric in our poses' scale. This is DA3's align_to_input_ext_scale=True path (api.py
inference), run step by step to record s and the residuals of the fit. Depth is planar (z along the optical axis).
Conventions: ours = OpenGL camera-to-world (x right, y up, looking along -z), pixel centres at +0.5, principal point at
W/2. DA3 = OpenCV world-to-camera: w2c = inv(c2w @ diag(1, -1, -1, 1)), cx = cy = W/2 - 0.5. DA3 never reads the
principal point and predicts depth on the input pixel grid, so output pixel (i, j) is the depth of image pixel (i, j).
On the GPU DA3 runs as released (bf16 autocast, fp32 camera encoder and heads), and the allocator is capped at
min(--mem, (free - 1.5 GiB) / total) of the card. On the CPU it runs in fp32, da3-large only.
Views: the training views only (the cache written by training_views.py). Frames: stored = every 2nd and every 5th video
frame (the paper's runs), s4 = the in-between observations (t % 8 == 0), future = the future observations (t <= 2 kobs),
or a list t1,t2,...

Usage: python dataset/estimated/da3_depth.py <training-view cache.npz> <out.npz> [--model da3-giant|da3-large]
       [--frames stored|s4|future|t1,t2,...] [--process-res 560] [--mem 0.8] [--device cuda|cuda:N|cpu]
"""
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

MODELS = {"da3-giant": "depth-anything/DA3-GIANT-1.1", "da3-large": "depth-anything/DA3-LARGE-1.1"}
NOVEL = (0, 1)  # GT cache indices of the novel views: never given to a model
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenGL <-> OpenCV camera axes
F16_MAX = 65000.0
GPU_SPARE = 1.5 * 2 ** 30  # bytes of the card's free memory left to other processes


def arg(k, d, c=str):
    return c(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d


def select_frames(spec, T):
    ts = sorted(set(range(0, T, 2)) | set(range(0, T, 5)))  # every 2nd and every 5th video frame (the paper's runs)
    if spec == "stored":
        return ts
    if spec == "s4":
        return [t for t in ts if t % 8 == 0]
    if spec == "future":
        kobs = (T // 2 // 2) // 2 * 2  # last observed latent of future10 (latent k = video frame 2k)
        return [t for t in ts if t <= 2 * kobs]
    return sorted(int(t) for t in spec.split(","))


def training_views(meta, C):
    """(array indices, GT cache view indices) of the training views of a cache with C views."""
    if "views" in meta:  # training-view cache: array index i is dataset view meta["views"][i]
        bad = set(meta["views"]) & (set(NOVEL) | set(meta.get("novel_views") or []))
        if bad:
            sys.exit(f"refusing a cache with the novel views {sorted(bad)}")
        return list(range(C)), list(meta["views"])
    idx = [c for c in range(C) if c not in NOVEL]
    return idx, idx


def load_rgb(jpeg, offs, T, ci, t):
    return np.asarray(Image.open(io.BytesIO(jpeg[offs[ci * T + t]:offs[ci * T + t + 1]].tobytes())).convert("RGB"))


def upsample_nn(a, W):
    """(N, h, w) -> (N, W, W) by nearest neighbour on pixel centres."""
    h, w = a.shape[-2:]
    iy = np.minimum(((np.arange(W) + 0.5) * h / W).astype(int), h - 1)
    ix = np.minimum(((np.arange(W) + 0.5) * w / W).astype(int), w - 1)
    return a[:, iy][:, :, ix]


def centres(E):
    """Camera centres -R^T t of (N, 3|4, 4) world-to-camera matrices."""
    return -np.einsum("nji,nj->ni", E[:, :3, :3], E[:, :3, 3])


def run(src, out):
    import torch
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils.pose_align import align_poses_umeyama
    from huggingface_hub import snapshot_download
    dev, name, res = arg("--device", "cuda"), arg("--model", "da3-giant"), arg("--process-res", 560, int)
    ray_pose = "--ray-pose" in sys.argv  # DA3 use_ray_pose: poses from its ray head instead of its camera head (dev test)
    gpu = dev.startswith("cuda")
    if not gpu and name == "da3-giant":
        sys.exit("refusing da3-giant on the CPU: its fp32 weights and activations exceed the host RAM budget")
    d = np.load(src)
    meta = json.loads(str(d["meta"]))
    scene = meta.get("scene", Path(src).stem)
    c2w, f, W = d["c2w"], float(d["f"]), int(d["W"])
    T = c2w.shape[1]
    idx, views = training_views(meta, c2w.shape[0])
    if "views" not in meta:
        sys.exit(f"{src} is not a training-view cache: write one with dataset/estimated/training_views.py")
    ts = select_frames(arg("--frames", "stored"), T)
    jpeg, offs = d["rgb_jpeg"], d["rgb_offsets"]
    mem = None
    if gpu:  # cap the allocator below what is free now, not a fraction of the whole card
        gi = torch.device(dev).index
        gi = torch.cuda.current_device() if gi is None else gi
        free, total = torch.cuda.mem_get_info(gi)
        mem = min(arg("--mem", 0.8, float), (free - GPU_SPARE) / total)
        if mem <= 0:
            sys.exit(f"only {free / 2 ** 30:.1f} GiB free on {dev}")
        torch.cuda.set_per_process_memory_fraction(float(mem), gi)
        print(f"{dev}: allocator capped at {mem * total / 2 ** 30:.1f} GiB ({free / 2 ** 30:.1f} of "
              f"{total / 2 ** 30:.1f} GiB free)", flush=True)
    # map_location: the state dict goes straight to the device, so the host holds one copy of the parameters
    model = DepthAnything3.from_pretrained(MODELS[name], map_location=dev).to(dev).eval()
    K = np.array([[f, 0, W / 2 - 0.5], [0, f, W / 2 - 0.5], [0, 0, 1]], np.float32)
    D = np.zeros((len(idx), T, W, W), np.float16)
    Cf = np.zeros_like(D)
    per_frame, t_all = {}, time.time()
    for t in ts:
        t0 = time.time()
        imgs = [load_rgb(jpeg, offs, T, ci, t) for ci in idx]
        w2c = np.linalg.inv(c2w[idx, t] @ FLIP).astype(np.float32)
        x, ex, ix = model._preprocess_inputs(imgs, w2c, np.repeat(K[None], len(idx), 0), res, "upper_bound_resize")
        x, ex_t, in_t = model._prepare_model_inputs(x, ex, ix)
        ex_n = model._normalize_extrinsics(ex_t.clone())
        if gpu:  # as released: bf16 autocast
            raw = model.forward(x, ex_n, in_t, [], False, ray_pose, "saddle_balanced")
        else:  # fp32 (CPU autocast would pick fp16)
            with torch.inference_mode():
                raw = model.model(x, ex_n, in_t, [], False, ray_pose, "saddle_balanced")
        p = model._convert_to_prediction(raw)
        # our poses (ex, OpenCV w2c) onto DA3's predicted ones: s maps our metres to DA3's units (api.py's alignment)
        _, _, s, ex_al = align_poses_umeyama(p.extrinsics, ex.numpy(), ransac=len(idx) >= 10, return_aligned=True,
                                             random_state=42)
        cam_err = np.linalg.norm(centres(ex_al) - centres(p.extrinsics), axis=1) / s
        cos = (np.einsum("nij,nij->n", ex_al[:, :3, :3], p.extrinsics[:, :3, :3]) - 1) / 2
        rot_err = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        depth, conf = p.depth / s, p.conf
        if depth.shape[-1] != W:
            depth, conf = upsample_nn(depth, W), upsample_nn(conf, W)
        D[:, t] = np.minimum(depth, F16_MAX)
        Cf[:, t] = np.minimum(conf, F16_MAX)
        per_frame[t] = dict(scale=float(s), cam_err_med_m=float(np.median(cam_err)), cam_err_max_m=float(cam_err.max()),
                            rot_err_med_deg=float(np.median(rot_err)), rot_err_max_deg=float(rot_err.max()),
                            focal_ratio=float(np.median(p.intrinsics[:, 0, 0] / ix[:, 0, 0].numpy())),
                            seconds=round(time.time() - t0, 2))
        print(f"t {t:3d}: scale {s:.4f}  camera centres med {1000 * np.median(cam_err):.1f} max {1000 * cam_err.max():.1f} mm  "
              f"rotations med {np.median(rot_err):.2f} deg  DA3 focal / ours {per_frame[t]['focal_ratio']:.3f}  "
              f"depth med {np.median(depth):.3f} m  ({time.time() - t0:.1f} s)", flush=True)
    out_meta = dict(scene=scene, source_cache=str(Path(src).resolve()), model=name, hf_repo=MODELS[name],
                    hf_revision=Path(snapshot_download(MODELS[name], local_files_only=True)).name, frames=ts, views=views,
                    novel_views_dropped="views" not in meta, process_res=res, use_ray_pose=ray_pose, device=dev, gpu_mem_fraction=mem,
                    precision="bf16 autocast" if gpu else "fp32",
                    align="Umeyama Sim(3) of our camera centres onto DA3's (RANSAC if >= 10 views), depth / scale",
                    pixel_centres="no correction: DA3 ignores the principal point, depth on the input pixel grid",
                    per_frame=per_frame)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, depth=D, conf=Cf, meta=json.dumps(out_meta))
    print(f"ok {out_meta['scene']}: {name} on {len(idx)} views x {len(ts)} frames at {res} px, {time.time() - t_all:.0f} s",
          flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    run(Path(sys.argv[1]), Path(sys.argv[2]))
