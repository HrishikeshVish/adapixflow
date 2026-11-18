"""Train the AdaPixFlow dynamics operator across scenes.

    python train.py config/physinone/train.yaml [--resume]

The run directory is <output_root>/<name>/. It holds last.pt (weights, optimizer, feature scale and config), the
evaluations eval_step<k>.json on the held-out scenes, and train.log.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from adapixflow.config import load_config, pick_device
from adapixflow.data import SceneSet, TrainSampler, feature_scale, load_split
from adapixflow.evaluation import evaluate_heldout
from adapixflow.model import VelocityUNet
from adapixflow.training import batch_loss


def subset(scenes, per_tier):
    """The first `per_tier` scenes of each tier (periodic evaluation), or every scene (per_tier <= 0)."""
    if per_tier <= 0:
        return scenes
    out = SceneSet({}, [])
    for t in sorted({it["tier"] for it in scenes.items}):
        out.items += [it for it in scenes.items if it["tier"] == t][:per_tier]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--resume", action="store_true", help="continue from <run>/last.pt")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tc, mc = cfg["train"], cfg["model"]
    dev = pick_device(cfg)
    rng = np.random.default_rng(tc["seed"])
    torch.manual_seed(tc["seed"])
    run_dir = Path(cfg["output_root"]) / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    logf = open(run_dir / "train.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    split = load_split(cfg["data"]["split"])
    dirs = [Path(d) for d in cfg["data"]["fused_dirs"]]
    train = SceneSet(split["train"], dirs)
    evals = {"test": SceneSet(split["test"], dirs), "val": SceneSet({"val": split["val"]}, dirs)}
    log(f"train {len(train)} scenes {[(t, len(v)) for t, v in train.by_tier.items()]} (missing {train.missing}), "
        + ", ".join(f"{k} {len(v)}" for k, v in evals.items()))
    assert len(train), "no training scenes found: fuse them first (scripts/prepare_physinone.sh)"

    scale = feature_scale(train, rng)
    sc_t = torch.from_numpy(scale).to(dev).view(1, 6, 1, 1, 1)
    model = VelocityUNet(tuple(mc["widths"]), mc["horizon"], mc["amp"]).to(dev)
    steps, warm = tc["steps"], tc["warmup"]
    opt = torch.optim.Adam(model.parameters(), tc["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(k, steps) / steps)))
    ck_cfg = {**cfg, "widths": ",".join(str(w) for w in mc["widths"]), "horizon": mc["horizon"]}
    start = 0
    if args.resume and (run_dir / "last.pt").exists():
        ck = torch.load(run_dir / "last.pt", weights_only=False, map_location=dev)
        model.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scale = ck["scale"]; sc_t = torch.from_numpy(scale).to(dev).view(1, 6, 1, 1, 1)
        start = ck["step"] + 1
        rng = np.random.default_rng(tc["seed"] + start)
        log(f"resumed from step {ck['step']}")
    sampler = TrainSampler(train, {**tc, "steps": steps}, rng)
    wb = None
    if cfg["logging"]["wandb"]:
        import wandb
        wb = wandb.init(project=cfg["logging"]["project"], name=cfg["name"], config=cfg, resume="allow")

    def evaluate(step, per_tier):
        model.eval()
        res = {}
        for k in cfg["eval"]["sets"]:
            r = evaluate_heldout(model, sc_t, subset(evals[k], per_tier), cfg["eval"], log=lambda m: None)
            for task, v in r.items():
                for metric, val in v["mean"].items():
                    res[f"{k}/{task}/{metric}"] = val
        model.train()
        return res

    t0, skipped = time.time(), 0
    for step in range(start, steps + 1):
        if step % tc["eval_every"] == 0 and not (step == start and start > 0):
            res = evaluate(step, 0 if step == steps else tc["eval_scenes_per_tier"])
            torch.save({"state_dict": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scale": scale, "step": step, "cfg": ck_cfg}, run_dir / "last.pt")
            (run_dir / f"eval_step{step}.json").write_text(json.dumps(res, indent=1))
            log(f"eval {step}: " + "  ".join(f"{k} {v * (100 if 'f1' in k else 1):.1f}" for k, v in res.items()
                                             if k.endswith("f1_2") or k.endswith("chamfer_mm")))
            if wb:
                wb.log({f"eval/{k}": v for k, v in res.items()}, step=step)
            if step == steps:
                break

        raw, _, s, h, use_end, idx = sampler.sample(step)
        loss, rel, cnt = batch_loss(model, sc_t, raw, s, h, use_end, step, tc, dev)
        for b in range(len(idx)):
            if cnt[b]:
                sampler.ema_rel[idx[b]] = 0.98 * sampler.ema_rel[idx[b]] + 0.02 * rel[b] / cnt[b]

        opt.zero_grad()
        if not torch.is_tensor(loss) or not loss.requires_grad or not torch.isfinite(loss):
            skipped += 1; sched.step(); continue  # no term in this batch, or a non-finite loss
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        if not torch.isfinite(gn):
            opt.zero_grad(); skipped += 1; sched.step(); continue
        opt.step(); sched.step()
        if wb and step % 20 == 0:
            wb.log({"train/loss": loss.item(), "train/grad_norm": float(gn), "train/lr": sched.get_last_lr()[0],
                    "train/spacing": s, "train/horizon": h}, step=step)
        if step % tc["log_every"] == 0:
            log(f"step {step:6d} loss {loss.item():.4f} (s {s}, h {h}, skipped {skipped})  {time.time() - t0:.0f} s")
    if wb:
        wb.finish()


if __name__ == "__main__":
    main()
