"""PhysGaia GT particles (particles/<object>/particles_frame_XXXX.json, a list of {"id", "position"}) -> one float32 array
per object, (T, N, 3) in the scene's world frame (Houdini units, Y up), frame t = particles_frame_{t+1:04d}.json.

Checks while converting (printed and written to <out>/<object>.json): the ids of every frame are 0..N-1 in order (so
row n is the same particle in every frame), N is constant, and per-frame bounding boxes. The JSON files are untrusted
data: they are only parsed (json.load), never executed. Run with python -I.

Also writes <object>_component.npy: the connected bodies of frame 1 (occupancy at --cell units, 26-connected), int16 per
particle, ordered by their first particle id (the PhysGaia converter tracks each ball by its body).

Usage: python -I dataset/benchmarks/physgaia_particles.py <scene dir> <out dir> [--cell 0.05]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np


def components(P: np.ndarray, cell: float) -> np.ndarray:
    from scipy import ndimage
    lo = P.min(0) - cell
    q = np.floor((P - lo) / cell).astype(np.int64)
    occ = np.zeros(q.max(0) + 2, bool)
    occ[tuple(q.T)] = True
    lab, n = ndimage.label(occ, structure=np.ones((3, 3, 3), bool))
    c = lab[tuple(q.T)] - 1
    first = np.array([np.flatnonzero(c == i)[0] for i in range(n)])  # order bodies by their first particle id
    remap = np.empty(n, np.int64)
    remap[np.argsort(first)] = np.arange(n)
    return remap[c].astype(np.int16)


def convert_object(src: Path, out_dir: Path, cell: float = 0.05) -> dict:
    files = sorted(src.glob("particles_frame_*.json"))
    idx = [int(f.stem.split("_")[-1]) for f in files]
    assert idx == list(range(1, len(files) + 1)), f"{src}: frame files not 1..T"
    arr, ids_ok, n0, t0 = None, True, None, time.time()
    for t, f in enumerate(files):
        q = json.load(open(f))
        ids = np.fromiter((p["id"] for p in q), np.int64, len(q))
        P = np.array([p["position"] for p in q], np.float32)
        if arr is None:
            n0 = len(q)
            arr = np.zeros((len(files), n0, 3), np.float32)
        assert len(q) == n0, f"{f}: {len(q)} particles, frame 1 had {n0}"
        ids_ok &= bool(np.array_equal(ids, np.arange(n0)))
        arr[t] = P
        if t % 40 == 0:
            print(f"  {src.name} frame {t + 1}/{len(files)} ({time.time() - t0:.0f} s)", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{src.name}.npy", arr)
    np.save(out_dir / f"{src.name}_component.npy", components(arr[0], cell))
    disp = np.linalg.norm(arr - arr[:1], axis=-1)
    info = {"object": src.name, "T": int(arr.shape[0]), "N": int(n0), "ids_are_0_to_N-1_every_frame": ids_ok,
            "bbox_lo_all": arr.reshape(-1, 3).min(0).tolist(), "bbox_hi_all": arr.reshape(-1, 3).max(0).tolist(),
            "bbox_lo_frame1": arr[0].min(0).tolist(), "bbox_hi_frame1": arr[0].max(0).tolist(),
            "max_displacement_from_frame1": float(disp.max()),
            "per_frame_max_step": np.linalg.norm(np.diff(arr, axis=0), axis=-1).max(1).tolist(),
            "source": str(src)}
    json.dump(info, open(out_dir / f"{src.name}.json", "w"), indent=1)
    print(f"{src.name}: T {info['T']} N {n0} ids ok {ids_ok} bbox {np.round(info['bbox_lo_all'], 3)} .. "
          f"{np.round(info['bbox_hi_all'], 3)} max disp {info['max_displacement_from_frame1']:.3f} ({time.time() - t0:.0f} s)",
          flush=True)
    return info


def main() -> None:
    scene, out = Path(sys.argv[1]), Path(sys.argv[2])
    cell = float(sys.argv[sys.argv.index("--cell") + 1]) if "--cell" in sys.argv else 0.05
    for src in sorted(p for p in (scene / "particles").iterdir() if p.is_dir()):
        convert_object(src, out, cell)


if __name__ == "__main__":
    main()
