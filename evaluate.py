"""Evaluate a trained AdaPixFlow model.

    python evaluate.py config/physinone/heldout.yaml      # Tab. 3: Test and Val scenes, spacings 2/4/8 and future
    python evaluate.py config/physinone/comparison.yaml   # Tab. 1: the six comparison scenes, views 0 and 1 held out

Options: --checkpoint (default: the config's), --scenes a,b (a subset), --out results.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adapixflow.config import load_config, pick_device
from adapixflow.data import SceneSet, load_split
from adapixflow.evaluation import evaluate_comparison, evaluate_heldout
from adapixflow.model import load_checkpoint


def print_table(res, title):
    print(f"\n{title}")
    print(f"{'task':16s} {'F1@1':>6s} {'F1@2':>6s} {'CD (mm)':>8s}")
    for task, v in res.items():
        m = v.get("mean", {})
        print(f"{task:16s} {m.get('f1_1', np.nan) * 100:6.1f} {m.get('f1_2', np.nan) * 100:6.1f} {m.get('chamfer_mm', np.nan):8.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--scenes", default="", help="comma-separated scene names (default: all of the config's)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    dev = pick_device(cfg)
    ckpt = args.checkpoint or cfg["checkpoint"]
    model, scale, _ = load_checkpoint(ckpt, dev, cfg["model"]["amp"])
    split = load_split(cfg["data"]["split"])
    only = set(filter(None, args.scenes.split(",")))
    keep = lambda names: [n for n in names if not only or n in only]  # noqa: E731
    out = {}
    if cfg["task"] == "heldout":
        dirs = [Path(d) for d in cfg["data"]["fused_dirs"]]
        for part in cfg["eval"]["sets"]:
            names = split[part] if isinstance(split[part], dict) else {part: split[part]}
            scenes = SceneSet({t: keep(v) for t, v in names.items()}, dirs)
            print(f"{part}: {len(scenes)} scenes")
            out[part] = evaluate_heldout(model, scale, scenes, cfg["eval"])
            print_table(out[part], f"{part} scenes ({len(scenes)}), F1 (%) and Chamfer distance on scored steps")
    elif cfg["task"] == "comparison":
        names = keep(split["comparison"])
        out["comparison"] = evaluate_comparison(model, scale, names, cfg["data"]["fused_dirs"][0], cfg["data"]["input_dir"],
                                                cfg["eval"])
        print_table(out["comparison"], f"comparison scenes ({len(names)}), F1 (%) and median Chamfer distance")
    else:
        raise ValueError(f"unknown task {cfg['task']}")
    path = Path(args.out or Path(cfg["output_root"]) / "eval" / f"{Path(cfg['_path']).stem}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
