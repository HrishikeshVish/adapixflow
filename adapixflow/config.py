"""YAML configs: every key and its default. A config file only needs the keys it changes."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "name": "adapixflow",
    "output_root": "./experiments/",
    "device": "auto",            # "auto" (CUDA if available), "cuda" or "cpu"
    "memory_fraction": None,     # cap on this process's share of GPU memory (shared GPUs), or null
    "data": {
        "split": "splits/physinone.json",
        "fused_dirs": ["data/physinone/fused"],      # full-view fused scenes (training, held-out ground truth)
        "input_dir": "data/physinone/fused_holdout",  # comparison scenes fused without views 0 and 1
    },
    "model": {
        "widths": [32, 64, 96, 128],
        "horizon": 12,           # H: node fields V_0..V_H per pass
        "amp": True,             # bfloat16 forward pass on CUDA
    },
    "train": {
        "steps": 24000,
        "batch": 4,
        "crop": 48,              # training windows of crop^3 voxels
        "lr": 1.0e-3,
        "warmup": 500,
        "grad_clip": 1.0,
        "seed": 0,
        "spacings": [1, 2, 3, 4, 6, 8, 11],
        "horizon_max": 12,
        "curriculum_frac": 0.3,  # the horizon cap grows from 1 to horizon_max over this fraction of the steps
        "p_end": 0.75,           # probability that the next observation is given (in-between sample)
        "p_eval_settings": 0.5,  # probability that a batch uses one of the evaluated settings below
        "eval_settings": [[2, 2, 1], [4, 4, 1], [8, 8, 1], [2, 10, 0]],  # (s, h, e)
        "p_adaptive": 0.5,       # share of samples drawn by the squared moving average of their relative loss
        "min_moved": 50,         # redraw a window until at least this many object voxels move
        "substeps": 4,
        "points": 1024,          # source and target points per step
        "loss_floor": 0.5,       # floor (voxels) of the static-prediction normalizer
        "rigid_weight": 0.2,
        "rigid_start_frac": 0.4,
        "rigid_neighbors": 6,
        "eval_every": 3000,
        "eval_scenes_per_tier": 2,  # periodic evaluation on a subset; the final evaluation uses every scene
        "log_every": 200,
    },
    "eval": {
        "sets": ["test", "val"],
        "spacings": [2, 4, 8],
        "horizon": 10,
        "substeps": 8,
        "max_side": 128,
        "tiling": ["margin", 64, 12],
    },
    "logging": {"wandb": False, "project": "adapixflow"},
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path):
    """The config at `path` merged over the defaults; `_path` records where it came from."""
    cfg = _merge(DEFAULTS, yaml.safe_load(open(path)) or {})
    cfg["_path"] = str(Path(path))
    return cfg


def pick_device(cfg):
    import torch

    dev = cfg.get("device", "auto")
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda" and cfg.get("memory_fraction"):
        torch.cuda.set_per_process_memory_fraction(float(cfg["memory_fraction"]))
    return dev
