"""Score another method's predicted object points with the protocol of the comparison scenes and the benchmarks
(Tab. 1 and 2): the same ground truth, window, scored targets and metrics as evaluate.py.

    python evaluate_points.py config/physinone/comparison.yaml --pred <dir>

<dir>/<task>/<scene>/pred.npz holds one method's predictions, with task inbetween_s4 or future10:
  frames  (n,) int64     the predicted target steps
  vox     (m, 3) int32   predicted object voxels, as indices of the ground-truth fusion's grid (data.fused_dirs[0])
  vox_off (n + 1,) int64 vox[vox_off[i]:vox_off[i + 1]] are the voxels of step frames[i]
A scored target missing from the file counts as an empty prediction (F1 0). Options: --scenes a,b, --out results.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adapixflow.config import load_config
from adapixflow.data import load_split, motion_window
from adapixflow.evaluation import score
from adapixflow.fusion import FusedScene, object_points

TASKS = {"inbetween_s4": ("inbetween", 4), "future10": ("future", 2)}


def targets(task, s, T, horizon):
    """{target step: the most recent observation} of a task, as evaluation.predictions defines them (a target is scored
    only if the object moved by more than one voxel since that observation)."""
    if task == "future":
        kobs = (T // 2) // 2 * 2
        return {kobs + j: kobs for j in range(1, min(horizon, T - 1 - kobs) + 1)}
    obs = list(range(0, T, s))
    return {k0 + j: k0 for k0 in obs[:-1] for j in range(1, s)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--scenes", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    names = load_split(cfg["data"]["split"])["comparison"]
    only = set(filter(None, args.scenes.split(",")))
    names = [n for n in names if not only or n in only]
    gt_dir, pred_dir = Path(cfg["data"]["fused_dirs"][0]), Path(args.pred)
    res = {}
    for tname, (task, s) in TASKS.items():
        res[tname] = {"scenes": {}}
        for name in names:
            path = pred_dir / tname / name / "pred.npz"
            if not path.exists():
                continue
            Sg = FusedScene(gt_dir, name)
            cg, size = motion_window(Sg, cfg["eval"]["max_side"])
            corner = cg - size // 2
            with np.load(path) as z:
                fr, vox, off = z["frames"], z["vox"], z["vox_off"]
            pred = {int(k): vox[off[i]:off[i + 1]].astype(np.float32) - corner for i, k in enumerate(fr)}
            preds = {}
            for k, kref in targets(task, s, Sg.T, cfg["eval"]["horizon"]).items():
                P = pred.get(k, np.zeros((0, 3), np.float32))
                preds[k] = (P[np.all((P >= 0) & (P < size), 1)], kref)
            gt = {}

            def gt_points(k):
                if k not in gt:
                    gt[k] = object_points(Sg.crop([k], cg, size)[0][0])
                return gt[k]

            m = score(preds, gt_points, Sg.voxel, "median")
            res[tname]["scenes"][name] = m
            print(f"  {tname:13s} {name[:48]:48s} F1@2 {m.get('f1_2', np.nan) * 100:5.1f} (n={m['steps']})", flush=True)
        rows = [m for m in res[tname]["scenes"].values() if m["steps"]]
        res[tname]["mean"] = {k: float(np.nanmean([m[k] for m in rows])) for k in ("f1_1", "f1_2", "chamfer_mm")} if rows else {}
    print(f"\n{'task':16s} {'F1@1':>6s} {'F1@2':>6s} {'CD (mm)':>8s}")
    for tname, v in res.items():
        m = v["mean"]
        if m:
            print(f"{tname:16s} {m['f1_1'] * 100:6.1f} {m['f1_2'] * 100:6.1f} {m['chamfer_mm']:8.1f}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
