"""Fuse scene caches into sparse voxel grids (adapixflow/fusion.py).

    python dataset/fuse.py data/physinone/cache data/physinone/fused [--split splits/physinone.json] [--workers 4]
    python dataset/fuse.py data/physinone/cache data/physinone/fused_holdout --holdout 0,1 \\
        --bounds-like data/physinone/fused --scenes <a,b,...>        # inputs of the comparison scenes

Options: --voxel 0.0125 (m), --stride 1 (pixel stride), --max-extent 4.0 (m), --occ-min 3.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapixflow.fusion import fuse_scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--split", default=None, help="fuse only the scenes of this split file")
    ap.add_argument("--scenes", default="", help="comma-separated scene names")
    ap.add_argument("--holdout", default="", help="camera indices to leave out, e.g. 0,1")
    ap.add_argument("--bounds-like", default=None, help="take the grid of another fusion of the same scenes")
    ap.add_argument("--voxel", type=float, default=0.0125)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-extent", type=float, default=4.0)
    ap.add_argument("--occ-min", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    files = sorted(p for p in Path(a.cache_dir).glob("*.npz") if not p.name.endswith(".tmp.npz"))
    keep = set(filter(None, a.scenes.split(",")))
    if a.split:
        sp = json.load(open(a.split))
        keep |= {n for k in ("train", "test") for v in sp[k].values() for n in v} | set(sp["val"])
    if keep:
        files = [p for p in files if p.name[:-4] in keep]
    ho = tuple(int(v) for v in a.holdout.split(",")) if a.holdout else ()
    bl = Path(a.bounds_like) if a.bounds_like else None
    out = Path(a.out_dir)
    print(f"fusing {len(files)} scenes into {out}", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        n = len(files)
        for msg in ex.map(fuse_scene, files, [out] * n, [a.voxel] * n, [a.stride] * n, [a.max_extent] * n, [a.occ_min] * n,
                          [ho] * n, [bl] * n):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
