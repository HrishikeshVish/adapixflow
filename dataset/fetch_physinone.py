"""Download PhysInOne scenes and convert each one to a compact cache (one .npz per scene); the raw zip is deleted.

    python dataset/fetch_physinone.py splits/physinone_sources.json data/physinone/cache [--parts train,test] [--workers 4]
    python dataset/fetch_physinone.py --local <extracted scene dir> data/physinone/cache     (an already extracted scene)

Cache layout (fusion re-reads it at any resolution):
  rgb_jpeg      uint8 buffer of JPEG frames (quality 92), all static cameras x all frames, 560 x 560
  rgb_offsets   int64 (n_cam * T + 1) byte offsets into rgb_jpeg (camera-major)
  depth         float16 (n_cam, T, 560, 560) planar depth (every second pixel of the 1120 px source, no averaging)
  seg           uint8   (n_cam, T, 560, 560) segmentation labels (every second pixel)
  c2w           float64 (n_cam, T, 4, 4) camera-to-world (OpenGL convention); f, W  focal length and width at 560 px
  meta          JSON: scene, source repository and path, phenomena, fps, frames, cameras, PhysInOne's recorder statistics
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

RES = 560


def free_gb(path: Path) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def convert_scene(src: Path, out_file: Path, meta: dict) -> dict:
    cams = sorted([p.stem.replace("blender_CineCamera_", "") for p in src.glob("blender_CineCamera_*.json")
                   if p.stem.replace("blender_CineCamera_", "").isdigit()], key=int)
    c0 = json.load(open(src / f"blender_CineCamera_{cams[0]}.json"))
    W0, T = c0["img_w"], c0["total_frames"]
    step = W0 // RES
    f = 0.5 * W0 / np.tan(0.5 * c0["camera_angle_x"]) / step
    jpeg_parts, offsets = [], [0]
    depth = np.zeros((len(cams), T, RES, RES), np.float16)
    seg = np.zeros((len(cams), T, RES, RES), np.uint8)
    c2w = np.zeros((len(cams), T, 4, 4))
    for ci, c in enumerate(cams):
        cj = json.load(open(src / f"blender_CineCamera_{c}.json"))
        frames = cj["frames"]
        for t in range(T):
            c2w[ci, t] = np.array(frames[min(t, len(frames) - 1)]["transform_matrix"], float)
            img = Image.open(src / f"CineCamera_{c}/rgb/{t:04d}.jpg").convert("RGB").resize((RES, RES), Image.BOX)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=92)
            jpeg_parts.append(np.frombuffer(buf.getvalue(), np.uint8))
            offsets.append(offsets[-1] + len(jpeg_parts[-1]))
            depth[ci, t] = np.load(src / f"CineCamera_{c}/depth/{t:04d}.npz")["depth"][step // 2::step, step // 2::step]
            seg[ci, t] = np.load(src / f"CineCamera_{c}/seg/{t:04d}.npz")["seg"][step // 2::step, step // 2::step]
    rec = src / "recorder_stats.json"  # PhysInOne's per-scene object list (stencil labels, materials, physics settings)
    meta = {**meta, "fps": c0.get("fps"), "frames": T, "cameras": cams, "source_res": W0,
            "recorder_stats": json.load(open(rec)) if rec.exists() else None}
    tmp = out_file.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, rgb_jpeg=np.concatenate(jpeg_parts), rgb_offsets=np.array(offsets, np.int64), depth=depth, seg=seg,
                        c2w=c2w, f=np.float64(f), W=np.int64(RES), meta=np.array(json.dumps(meta)))
    tmp.rename(out_file)
    return meta


def fetch_one(item: dict, out_dir: Path, stage: Path, min_free: float) -> str:
    name = Path(item["path"]).stem
    out_file = out_dir / f"{name}.npz"
    if out_file.exists():
        return f"skip {name}"
    if free_gb(out_dir) < min_free + 3 * item["size"] / 1e9:
        return f"STOP low disk ({free_gb(out_dir):.0f} GB free) before {name}"
    url = f"https://huggingface.co/datasets/{item['repo']}/resolve/main/{item['path']}"
    zpath = stage / f"{name}.zip"
    for attempt in range(5):
        r = subprocess.run(["curl", "-sSL", "--fail", "--retry", "3", "-o", str(zpath), url], capture_output=True, text=True)
        if r.returncode == 0 and zpath.exists() and zpath.stat().st_size == item["size"]:
            break
        time.sleep(10 * (attempt + 1))
    else:
        zpath.unlink(missing_ok=True)
        return f"FAIL download {name}: {r.stderr.strip()[:200]}"
    xdir = stage / name
    try:
        with zipfile.ZipFile(zpath) as z:
            z.extractall(stage)
        zpath.unlink()
        src = xdir if xdir.exists() else next(p for p in stage.iterdir() if p.is_dir() and p.name.startswith(name))
        convert_scene(src, out_file, {"scene": name, "repo": item["repo"], "path": item["path"], "phenomena": item["phenomena"],
                                      "tier": item.get("tier")})
    except Exception as e:  # noqa: BLE001
        return f"FAIL convert {name}: {e}"
    finally:
        zpath.unlink(missing_ok=True)
        shutil.rmtree(xdir, ignore_errors=True)
    return f"ok {name} ({out_file.stat().st_size / 1e6:.0f} MB, {free_gb(out_dir):.0f} GB free)"


def main() -> None:
    if sys.argv[1] == "--local":
        src, out_dir = Path(sys.argv[2]), Path(sys.argv[3])
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = convert_scene(src, out_dir / f"{src.name}.npz", {"scene": src.name, "phenomena": src.name.split("__")[0].split("_")})
        print(meta["scene"], f"{(out_dir / f'{src.name}.npz').stat().st_size / 1e6:.0f} MB")
        return
    sources = json.load(open(sys.argv[1]))["scenes"]
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = out_dir.parent / (out_dir.name + "_staging")
    stage.mkdir(exist_ok=True)
    arg = lambda k, d, c=int: c(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d  # noqa: E731
    workers, min_free = arg("--workers", 4), arg("--min-free-gb", 50.0, float)
    names = None
    if "--scenes" in sys.argv:
        names = set(arg("--scenes", "", str).split(","))
    items = [{**it, "name": n} for n, it in sources.items() if names is None or n in names]
    print(f"{len(items)} scenes, {sum(it['size'] for it in items) / 1e9:.0f} GB to download; {workers} workers; "
          f"{free_gb(out_dir):.0f} GB free", flush=True)
    stop = False
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch_one, {**it, "phenomena": it.get("phenomena", it["name"].split("__")[0].split("_"))},
                          out_dir, stage, min_free): it for it in items}
        for k, fu in enumerate(as_completed(futs), 1):
            msg = fu.result()
            print(f"[{k}/{len(items)}] {time.strftime('%H:%M:%S')} {msg}", flush=True)
            if msg.startswith("STOP") and not stop:
                stop = True
                for other in futs:
                    other.cancel()
    print("done", flush=True)


if __name__ == "__main__":
    main()
