"""The training views of a PhysInOne scene cache, as the estimators read them: every camera except the novel views 0
and 1, poses and intrinsics unchanged, and every frame's RGB decoded and re-encoded as JPEG (quality 92, the images DA3
and SAM 2.1 saw in the paper's runs). No depth or masks are copied (depth is stored as zeros, invalid everywhere).

Usage: python dataset/estimated/training_views.py <scene cache.npz> <out dir>
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

NOVEL = (0, 1)


def training_views(cache: Path, out_dir: Path) -> Path:
    d = np.load(cache)
    gt_meta = json.loads(str(d["meta"]))
    C, T, W = d["depth"].shape[0], d["depth"].shape[1], int(d["W"])
    views = [c for c in range(C) if c not in NOVEL]
    jpeg, offs = d["rgb_jpeg"], d["rgb_offsets"]
    bufs, out_offs = [], [0]
    for c in views:
        for t in range(T):
            im = Image.open(io.BytesIO(jpeg[offs[c * T + t]:offs[c * T + t + 1]].tobytes())).convert("RGB")
            b = io.BytesIO()
            im.save(b, "JPEG", quality=92)
            bufs.append(np.frombuffer(b.getvalue(), np.uint8))
            out_offs.append(out_offs[-1] + len(bufs[-1]))
    name = cache.name[:-4]
    meta = {"scene": name, "source": "training-views", "frames": T, "views": views, "novel_views": list(NOVEL),
            "fps": 30, "moving_labels_override": [128], "tier": gt_meta.get("tier"), "phenomena": gt_meta.get("phenomena")}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.npz"
    np.savez_compressed(path, rgb_jpeg=np.concatenate(bufs), rgb_offsets=np.array(out_offs, np.int64),
                        depth=np.zeros((len(views), T, W, W), np.float16), c2w=d["c2w"][views], f=np.float64(d["f"]),
                        W=np.int64(W), meta=json.dumps(meta))
    print(f"ok {name}: {len(views)} training views x {T} frames -> {path}", flush=True)
    return path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    training_views(Path(sys.argv[1]), Path(sys.argv[2]))
