"""Estimated moving-object masks for the estimated-input experiment: fully automatic SAM 2.1 (no ground truth, no human
prompts), seg = 128 at the moving-object pixels of every view at the observed frames, from those frames' RGB and estimated
depth only. SAM 2.1: https://github.com/facebookresearch/sam2, weights https://huggingface.co/facebook/sam2.1-hiera-large,
run through the transformers implementation (Sam2Model, Sam2VideoModel; transformers 5.17), inference only.
Pipeline:
  (a) evidence, per view and observed frame t: non-shadow RGB change vs the other observed frames (some pair s != t differs
      by more than RGB_K noise sigmas (median-absolute-difference estimate per pair and channel, floor RGB_FLOOR); shadow =
      the darker side of a same-chromaticity change (luminance ratio in SHADOW_LUM, chromaticity within SHADOW_CHROMA;
      directional, so a bright grey object on a darker grey floor still counts where it is)), attributed to t either by
      the RGB (t also differs from the per-pixel median of the observed frames, so the places an object only visits at
      other frames do not count at t) or by the depth (t is foreground: closer than the per-pixel BG_PCT percentile of the
      observed depth by > max(FG_ABS, FG_REL * depth); this keeps objects at rest at the median; foreground alone is not
      evidence); 3x3 opening; connected
      components of >= MIN_CC px are the seeds (at most MAX_SEEDS per frame, largest first). Each frame's depth is first
      divided by its global scale, shared by its views (DA3: one Sim(3) fit per frame): the median, pooled over the views'
      RGB-static pixels, of the ratio to the per-pixel median over the observed frames, within +-SCALE_WIN (ratios beyond
      are moving or occluding objects). On 3 dev scenes (DA3-large, 4-6 s4 frames) it spans 1.0-1.5 % over the frames,
      which as foreground and centroid motion would cover whole frames. FG_REL and MOVE_REL come from the residual noise
      on the RGB-static pixels of the dev scenes CrankPush__bg062 and MovingHitsMoving__bg139 (UniformPanelSpin_BalloonLift's
      are not static: dark balloons pass over dark shadows): FG_REL = 4 %, the smallest of 1.5 / 2 / 3 / 4 % that marks
      <= 1 % of them (median over views; 3 %: 0.8 % and 2.3 %); MOVE_REL = 3 %, above the 99th percentile of the
      40-px-block residual (1.6 % and 2.8 %, median over views).
  (b) each seed prompts the SAM 2.1 image model: its box, up to 3 positive points at distance-transform peaks, up to 2
      negatives at never-changed pixels in a ring around the box (SAM's convention: integer pixel indices, inclusive box);
      of the 3 candidate masks the one with the best F1 vs the dilated evidence (precision vs all of the frame's evidence,
      recall vs this seed) is kept; masks with IoU > 0.5 merged, masks > 60 % of the image dropped; seeds already > 50 %
      inside a kept mask are not prompted again
  (c) per view, SAM 2.1 video propagation over the observed frames only: a forward sweep assigns each frame's seed masks to
      the masklet whose propagated mask they overlap (IoU > 0.5; added as corrections), skips them if > 50 % inside a
      tracked masklet (a part of it), or starts a new masklet (at most MAX_MASKLETS per view); a backward sweep (its own
      session, each masklet entering at its first seed) covers the frames before each masklet's first seed; seed masks
      override propagated ones at their own frames, so objects at rest in some observed frames stay labelled
  (d) motion test per masklet: moved if in >= CHANGE_FRAMES frames its 3D centroid (median of its unprojected pixels, known
      K and c2w, scale-corrected depth) is farther than max(MOVE_M, MOVE_REL * its median depth) from its median centroid
      over the observed frames (depth-aware: mask jitter and residual depth noise grow with depth; robust to a single bad
      frame); or changed if >= CHANGE_FRAC of its pixels show attributed change in >= CHANGE_FRAMES frames; shadow veto:
      more attributed shadow-type change than attributed other change (like with like) and < CHANGE_FRAC depth foreground
  (e) cross-view vote with the known poses: masklets lifted to a VOX voxel grid per frame (LINK_FRAMES frames, 6-neighbour
      dilation against depth noise: it bridges surfaces within ~2 VOX, so masklets link only where their views see
      overlapping surfaces; masklets of an object thicker than that seen from opposite sides do not link and keep their
      own decisions), linked across views by voxel IoU > LINK_IOU; an object (connected group) is moving if accepted in
      >= VOTE_MIN views or in >= VOTE_FRAC of the views that see it (views with one of its masklets, or where >= SEE_FRAC
      of its voxels project onto the depth surface within VIS_TOL); then all its masklets are moving
  --no-depth: RGB evidence only, no depth attribution, no centroid test, no vote (smoke tests on GT caches without an estimate).
All thresholds are the module constants below, fixed a priori or (FG_REL, MOVE_REL) from DA3 noise measured
on dev scenes; none was tuned on the headline scenes.
--v2 (the paper's setting; chosen on dev scenes only, by a CPU replay of the vote on per-masklet dumps of the v1 run): two
  changes after the motion test, nothing before it. (1) The vote's voxel keys come only from masklet pixels whose depth is
  not at a depth discontinuity (5x5 max - min of the scale-corrected depth <= KEY_EDGE * depth) and, for a da3_depth.py
  estimate with a confidence map, whose confidence is >= KEY_CONF: DA3's edge pixels (median error 17-20 mm, 30 % beyond
  5 cm) smear each view's voxels along its rays, so masklets of the same object stopped linking (median voxel IoU between
  views 0.17 < LINK_IOU) and were outvoted. (2) A masklet accepted by the motion test is rejected if its depth-foreground
  fraction (valid-depth pixels) is < FG_VETO in every frame where it exists: near-black floor shadows (luminance ratio
  below SHADOW_LUM, so the colour veto misses them) lie on the background surface. Dev scenes (pooled IoU v1 -> v2):
  CrankPush__bg062 0.563 -> 0.847, MovingHitsMoving__bg139 0.823 -> 0.841, UniformPanelSpin_BalloonLift__bg001
  0.552 -> 0.607, LiquidMultiTransfers__bg538 0.715 -> 0.709 (a stronger veto, 0.1, cost it 0.1: its DA3 depth is poor);
  the two pendulum dev scenes are unchanged within 0.015. The rule: best mean over the 4 dev scenes with no scene down by
  more than 0.01.

Views: the training views of a cache written by training_views.py (meta['views']; a cache with a novel view is refused),
the same RGB that DA3 sees. The output's views are dataset view ids.
Depth (GT depth is never an input): a da3_depth.py npz (no 'seg' / 'rgb_jpeg'; meta scene = the cache's, model, views
without novel ones, frames; 'depth' (C, T, 560, 560) or (C, len(meta['frames']), 560, 560), planar z in metres along -z,
0 = invalid). Only the requested frames' RGB and depth are read: the arrays are streamed out of the .npz
and every other frame is skipped.
--frames: s4 = stored frames with t % 8 == 0 (in-between s4 observations); future = video frames 0, 2, ..., 2 kobs (future10
observations as the baselines get them, kobs = (T // 2 // 2) // 2 * 2; the odd stored multiples of 5 are not used); or a list
of stored frames.
Memory: one SAM model on the device at a time (all seed masks first, then propagation); the video session keeps one frame's
vision features, drops the 1024^2 high-res masks after each frame, keeps the memories of tracked frames only while they are in
the attention window (num_maskmem - 1 frames; object pointers kept) and moves conditioning memories to the host. The GPU
allocator is capped at min(--mem, (free - 1.5 GiB) / total) of the chosen card.

Usage: python dataset/estimated/sam_masks.py <training-view cache.npz> <depth.npz> <out.npz> --frames s4|future|t1,t2,...
       --v2 [--device cuda|cuda:N|cpu] [--mem 0.3] [--views i,j  (positions in the run's view list, smoke tests)]
"""
import gc
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.sparse import csgraph

MODEL = "facebook/sam2.1-hiera-large"
NOVEL = (0, 1)  # GT-cache novel views: never read
DEPTH_CACHE = ("mvs", "mapanything")  # RGB-only cache depths that are estimates (build_cache.py --depth), usable as 'cache'
GPU_SPARE = 1.5 * 2 ** 30  # bytes of the card's free memory left to other processes
# (a) evidence
RGB_K, RGB_FLOOR = 4.0, 10 / 255  # change: |dI| > max(RGB_FLOOR, RGB_K * 1.4826 * median|dI|) in some channel
SHADOW_LUM, SHADOW_CHROMA = (0.3, 0.9), 0.03  # shadow: luminance ratio in this range, rg-chromaticity within this
BG_PCT, FG_ABS, FG_REL = 90, 0.025, 0.04  # depth foreground: D_bg - D_t > max(FG_ABS, FG_REL * D_bg)
SCALE_WIN, SCALE_BIN = 0.1, 1e-4  # per-frame depth scale: median log ratio within +-log(1 + SCALE_WIN), bins of SCALE_BIN
MIN_CC, MAX_SEEDS = 20, 16  # seed components (px), seeds per frame
# (b) seed masks
POS_PTS, NEG_PTS, EV_DILATE, NEVER_MARGIN = 3, 2, 3, 5  # prompt points; evidence dilation (px); never-changed margin (px)
MERGE_IOU, COVER, MAX_AREA = 0.5, 0.5, 0.6  # merge masks / skip covered seeds / drop large masks
# (c) propagation
MAX_MASKLETS = 32  # per view
# (d) motion test
MOVE_M, MOVE_REL, CHANGE_FRAC, CHANGE_FRAMES, MIN_PTS = 0.025, 0.03, 0.3, 2, 20
# (e) cross-view vote
VOX, LINK_IOU, LINK_FRAMES, VOTE_MIN, VOTE_FRAC, SEE_FRAC, VIS_TOL = 0.025, 0.2, 8, 2, 0.3, 0.2, 0.05
# --v2 (dev scenes only): voxel keys off depth edges / low DA3 confidence; depth-foreground veto of accepted masklets
KEY_EDGE, KEY_CONF, FG_VETO = 0.05, 3.0, 0.02
CONSTS = {k: v for k, v in dict(globals()).items() if k.isupper() and k not in ("ROOT", "NOVEL", "DEPTH_CACHE", "GPU_SPARE")}


def arg(k, d, c=str):
    return c(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d


def read_header(f):
    ver = np.lib.format.read_magic(f)
    shape, fortran, dtype = (np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0)(f)
    assert not fortran
    return shape, dtype


def npz_header(path, key):
    with zipfile.ZipFile(path) as z, z.open(f"{key}.npy") as f:
        return read_header(f)


def npz_spans(path, key, spans):
    """Elements [a, b) of the flattened array `key` of an .npz for each (a, b) in spans, streamed (forward seeks): every other
    element is decompressed and skipped, never kept."""
    out = {}
    with zipfile.ZipFile(path) as z, z.open(f"{key}.npy") as f:
        _, dtype = read_header(f)
        base = f.tell()
        for a, b in sorted(set(spans)):
            f.seek(base + int(a) * dtype.itemsize)
            out[a, b] = np.frombuffer(f.read((int(b) - int(a)) * dtype.itemsize), dtype)
    return [out[s] for s in spans]


def npz_frames(path, key, cells):
    """{(c, t): (H, W) array} for the (view index, frame index) cells of a (C, T, H, W) array in an .npz, streamed."""
    shape, _ = npz_header(path, key)
    n = int(np.prod(shape[2:]))
    spans = [((c * shape[1] + t) * n, (c * shape[1] + t + 1) * n) for c, t in cells]
    return {ct: a.reshape(shape[2:]) for ct, a in zip(cells, npz_spans(path, key, spans))}


def parse_frames(spec, T):
    ts = sorted(set(range(0, T, 2)) | set(range(0, T, 5)))  # the frames the caches store
    if spec == "s4":
        return [t for t in ts if t % 8 == 0]
    if spec == "future":  # future10's observation: video frames 0, 2, ..., 2 kobs (latents 0..kobs)
        return [t for t in ts if t <= 2 * ((T // 2 // 2) // 2 * 2) and t % 2 == 0]
    fr = sorted(int(x) for x in spec.split(","))
    assert set(fr) <= set(ts), f"frames must be stored frames: {sorted(set(fr) - set(ts))}"
    return fr


def scene_views(cache):
    """(cache view indices, GT-cache view ids, cache meta, RGB-only?) of the views a run may use."""
    meta = json.loads(str(np.load(cache)["meta"]))
    meta.setdefault("scene", Path(cache).stem)
    C = npz_header(cache, "depth")[0][0]
    if "views" in meta:  # training-view cache: array index i is GT-cache view meta["views"][i]
        bad = set(meta["views"]) & (set(NOVEL) | set(meta.get("novel_views") or []))
        if bad:
            sys.exit(f"refusing a cache with the novel views {sorted(bad)}")
        return list(range(C)), list(meta["views"]), meta, True
    sys.exit(f"{cache} is not a training-view cache: write one with dataset/estimated/training_views.py")


def load_depth(src, cache, meta, cidx, vids, frames, T):
    """({(view position, frame): (W, W) float32 depth} at the requested frames only, the depth's producer), from a
    da3_depth.py estimate or an RGB-only cache's own estimated depth ('cache'). GT depth is refused in every form."""
    if src == "cache":
        if "views" not in meta:
            sys.exit("refusing depth 'cache' for a GT cache: its depth is ground truth")
        if meta.get("depth") not in DEPTH_CACHE:
            sys.exit(f"refusing depth 'cache': the cache's depth is {meta.get('depth')!r}, not an estimate {DEPTH_CACHE}")
        D = npz_frames(cache, "depth", [(c, t) for c in cidx for t in frames])
        out, model = {(i, t): D[c, t].astype(np.float32) for i, c in enumerate(cidx) for t in frames}, meta["depth"]
    else:
        d = np.load(src)
        if {"seg", "rgb_jpeg"} & set(d.files) or "meta" not in d.files:
            sys.exit(f"refusing depth {src}: a cache, not a da3_depth.py estimate")
        m = json.loads(str(d["meta"]))
        lacks = [k for k in ("scene", "model", "views", "frames") if k not in m]
        if lacks:
            sys.exit(f"refusing depth {src}: its meta lacks {lacks}, not a da3_depth.py estimate")
        if m["scene"] != meta["scene"]:
            sys.exit(f"refusing depth {src}: an estimate of {m['scene']}, not {meta['scene']}")
        if set(m["views"]) & set(NOVEL):
            sys.exit(f"refusing depth {src}: made with the novel views {sorted(set(m['views']) & set(NOVEL))}")
        if not set(vids) <= set(m["views"]) or not set(frames) <= set(m["frames"]):
            sys.exit(f"refusing depth {src}: no estimate for views {sorted(set(vids) - set(m['views']))} or frames "
                     f"{sorted(set(frames) - set(m['frames']))}")
        shape = npz_header(src, "depth")[0]
        pos = [m["views"].index(v) for v in vids]
        fpos = {t: t for t in frames} if shape[1] == T else {t: m["frames"].index(t) for t in frames}
        D = npz_frames(src, "depth", [(p, fpos[t]) for p in pos for t in frames])
        out, model = {(i, t): D[p, fpos[t]].astype(np.float32) for i, p in enumerate(pos) for t in frames}, m["model"]
    assert all((z > 0).any() for z in out.values()), "depth source has empty frames among the requested ones"
    return out, model


def load_rgb(cache, cells, T):
    """{(view index, frame): (W, W, 3) uint8} for the requested cells only (JPEG bytes streamed, others skipped)."""
    offs = np.load(cache)["rgb_offsets"]
    spans = [(offs[c * T + t], offs[c * T + t + 1]) for c, t in cells]
    jpegs = npz_spans(cache, "rgb_jpeg", spans)
    return {ct: np.asarray(Image.open(io.BytesIO(b.tobytes())).convert("RGB")) for ct, b in zip(cells, jpegs)}


def unproject(z, M, f, mask):
    v, u = np.nonzero(mask & (z > 0) & (z < 1e3))
    W, zz = z.shape[1], z[v, u].astype(np.float64)
    pc = np.stack([(u + 0.5 - W / 2) / f * zz, -(v + 0.5 - W / 2) / f * zz, -zz], -1)
    return pc @ M[:3, :3].T + M[:3, 3]


def pack(a):
    return np.packbits(a, axis=-1)


def unpack(a, W):
    return np.unpackbits(a, axis=-1, count=W).astype(bool)


# (a) evidence ------------------------------------------------------------------------------------------------------------
def lum_chroma(a):
    la = a.sum(-1) + 1e-3
    return la, a[..., :2] / la[..., None]


def changed(a, b, lca, lcb):
    """Change between images a and b ((H, W, 3) float in [0, 1]; lc: lum_chroma), noise-adaptive per channel, split per side:
    (non-shadow, shadow) for a, then for b. Shadow is directional: only the darker side of a same-chromaticity change is
    excused (a bright grey object over a darker grey floor is a change where the object is)."""
    d = np.abs(a - b)
    sig = 1.4826 * np.median(d[::4, ::4].reshape(-1, 3), 0)
    ch = (d > np.maximum(RGB_FLOOR, RGB_K * sig)).any(-1)
    same = ch & (np.abs(lca[1] - lcb[1]).max(-1) < SHADOW_CHROMA)
    r = lca[0] / lcb[0]
    sha, shb = same & (r > SHADOW_LUM[0]) & (r < SHADOW_LUM[1]), same & (1 / r > SHADOW_LUM[0]) & (1 / r < SHADOW_LUM[1])
    return ch & ~sha, sha, ch & ~shb, shb


def rgb_evidence(imgs):
    """Per observed frame of one view, (n, H, W) bool each: non-shadow change vs some other frame, shadow-type change, and
    both attributed to the frame by the RGB (it also differs from the per-pixel median of the observed frames)."""
    I = imgs.astype(np.float32) / 255
    n = len(I)
    med = np.median(I, 0)
    lc, lcm = [lum_chroma(x) for x in I], lum_chroma(med)
    sup, shd = np.zeros(I.shape[:3], bool), np.zeros(I.shape[:3], bool)
    for t in range(n):
        for s in range(t + 1, n):
            ct, ht, cs, hs = changed(I[t], I[s], lc[t], lc[s])
            sup[t] |= ct
            shd[t] |= ht
            sup[s] |= cs
            shd[s] |= hs
    vs_med = [changed(I[t], med, lc[t], lcm)[:2] for t in range(n)]  # differs from the typical appearance: object here
    return sup, shd, sup & np.stack([c for c, _ in vs_med]), shd & np.stack([h for _, h in vs_med])


def log_ratio_hist(D, quiet):
    """Per frame of one view, counts of log(depth / per-pixel median over the frames) in SCALE_BIN bins on
    [-log(1 + SCALE_WIN), log(1 + SCALE_WIN)), on the RGB-static pixels valid in every frame."""
    lim = np.log1p(SCALE_WIN)
    nb = int(round(2 * lim / SCALE_BIN))
    h = np.zeros((len(D), nb), np.int64)
    ok = (D > 0).all(0) & quiet
    if ok.any():
        idx = np.floor((np.log(D[:, ok] / np.median(D[:, ok], 0)) + lim) / SCALE_BIN).astype(np.int64)
        for k in range(len(D)):
            h[k] = np.bincount(idx[k][(idx[k] >= 0) & (idx[k] < nb)], minlength=nb)
    return h


def frame_scales(h, min_px):
    """Per-frame depth scale: exp of the median of the log ratios pooled over the views (bin centre), 1 with < min_px."""
    c = np.cumsum(h, 1)
    i = np.array([np.searchsorted(ck, ck[-1] / 2) for ck in c])
    return np.where(c[:, -1] >= min_px, np.exp(-np.log1p(SCALE_WIN) + SCALE_BIN * (i + 0.5)), 1.0)


def depth_foreground(D):
    """(n, H, W) bool: closer than the per-pixel BG_PCT percentile of the valid observed depth (linear interpolation) by
    more than max(FG_ABS, FG_REL * that)."""
    Dn = np.where(D > 0, D, np.nan)
    cnt = (D > 0).sum(0)
    pos = BG_PCT / 100 * np.maximum(cnt - 1, 0)
    lo, w = np.floor(pos).astype(int), pos - np.floor(pos)
    Ds = np.sort(Dn, 0)  # invalid (nan) last
    at = lambda i: np.take_along_axis(Ds, np.minimum(i, len(D) - 1)[None], 0)[0]  # noqa: E731
    bg = np.where(cnt > 0, at(lo) * (1 - w) + at(lo + 1) * w * (w > 0), np.nan)
    with np.errstate(invalid="ignore"):
        return (bg - Dn) > np.maximum(FG_ABS, FG_REL * bg)


def seed_evidence(sup, shd, chg, fg):
    """Seed evidence (n, H, W): change attributed by the RGB, or by the depth (foreground and changed vs some other frame),
    3x3 opening; and the never-changed pixels (H, W)."""
    ev = np.stack([ndimage.binary_opening(chg[t] | (fg[t] & sup[t]), np.ones((3, 3))) for t in range(len(sup))])
    return ev, ~ndimage.binary_dilation((sup | shd | fg).any(0), iterations=NEVER_MARGIN)


def components(ev):
    lab, n = ndimage.label(ev)
    if not n:
        return []
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    keep = [i + 1 for i in np.argsort(-sizes) if sizes[i] >= MIN_CC][:MAX_SEEDS]
    return [lab == i for i in keep]


# (b) seed masks ----------------------------------------------------------------------------------------------------------
def peaks(region, k, min_sep=5):
    """Up to k (x, y) pixel indices at distance-transform peaks of a region, each suppressing a disk around it."""
    dt = ndimage.distance_transform_edt(region)
    yy, xx = np.mgrid[0:region.shape[0], 0:region.shape[1]]
    out = []
    for _ in range(k):
        i = np.unravel_index(np.argmax(dt), dt.shape)
        if dt[i] < 1:
            break
        out.append([float(i[1]), float(i[0])])  # SAM adds the half pixel itself
        dt[(yy - i[0]) ** 2 + (xx - i[1]) ** 2 <= max(dt[i], min_sep) ** 2] = 0
    return out


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0


def seed_masks(im, ip, img, comps, ev, never, dev):
    """SAM 2.1 masks of one frame's seeds (box + positive peaks + negatives at never-changed pixels around the box), best of
    the 3 candidates by F1 vs the dilated evidence; merged at IoU > MERGE_IOU, > MAX_AREA of the image dropped. Seeds (largest
    first) already covered by a kept mask (> COVER of their pixels) are not prompted again."""
    import torch
    if not comps:
        return []
    W = img.shape[0]
    emb = im.get_image_embeddings(ip(images=img, return_tensors="pt")["pixel_values"].to(dev))
    ev_d = ndimage.binary_dilation(ev, iterations=EV_DILATE)
    masks = []
    for c in comps:
        if any((c & m).sum() > COVER * c.sum() for m in masks):
            continue
        ys, xs = np.nonzero(c)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
        ring = np.zeros_like(c)
        for sc, val in ((2.0, True), (1.25, False)):  # negatives: between the box scaled by 1.25 and by 2
            ring[max(0, int(cy - sc * hh)):int(np.ceil(cy + sc * hh)), max(0, int(cx - sc * hw)):int(np.ceil(cx + sc * hw))] = val
        pos, neg = peaks(c, POS_PTS), peaks(ring & never, NEG_PTS)
        enc = ip(original_sizes=[[W, W]], input_points=[[pos + neg]], input_labels=[[[1] * len(pos) + [0] * len(neg)]],
                 input_boxes=[[[float(x0), float(y0), float(x1 - 1), float(y1 - 1)]]], return_tensors="pt")  # inclusive box
        with torch.inference_mode():
            o = im(image_embeddings=emb, input_points=enc["input_points"].to(dev), input_labels=enc["input_labels"].to(dev),
                   input_boxes=enc["input_boxes"].to(dev), multimask_output=True)
        cand = ip.post_process_masks(o.pred_masks.cpu(), [[W, W]])[0][0].numpy()
        c_d = ndimage.binary_dilation(c, iterations=EV_DILATE)
        best, f1 = None, -1.0
        for m in cand:
            a = m.sum()
            if a == 0 or a > MAX_AREA * m.size:
                continue
            p, r = (m & ev_d).sum() / a, (m & c_d).sum() / c_d.sum()
            if 2 * p * r / max(p + r, 1e-9) > f1:
                best, f1 = m, 2 * p * r / max(p + r, 1e-9)
        if best is not None:
            masks.append(best)
    merged = []
    for m in sorted(masks, key=lambda m: -m.sum()):
        j = next((j for j, k in enumerate(merged) if iou(k, m) > MERGE_IOU), None)
        if j is None:
            merged.append(m)
        else:
            merged[j] = merged[j] | m
    return [m for m in merged if m.sum() <= MAX_AREA * m.size]


# (c) propagation ---------------------------------------------------------------------------------------------------------
def tidy(sess, k, step, window):
    """Session state after frame k of a sweep (step +1 forward, -1 backward; transformers 5.17 layout): drop the 1024^2
    high-res masks (only the memory encoder of the same call reads them), move conditioning memories (attention reads all
    of them) to the state device, drop the memories of the tracked frame leaving the attention window (object pointers
    kept: up to 16 frames read them)."""
    for od in sess.output_dict_per_obj.values():
        cond, track_ = od["cond_frame_outputs"], od["non_cond_frame_outputs"]
        for o in (cond.get(k), track_.get(k)):
            if o:
                o.pop("high_res_masks", None)
        for key in ("maskmem_features", "maskmem_pos_enc"):
            if key in cond.get(k, {}):
                cond[k][key] = cond[k][key].to(sess.inference_state_device)
            track_.get(k - step * window, {}).pop(key, None)


def frame_masks(vm, sess, k, new_ids=(), reverse=False):
    """SAM 2.1 video output at frame k. With new inputs at k (new_ids), the other objects' outputs of this frame's first
    call are reused: lent to the conditioning outputs for the call (HF re-tracks every object without new inputs
    otherwise), then put back."""
    lent = []
    for i in range(sess.get_obj_num()):
        od = sess.output_dict_per_obj[i]
        if new_ids and sess.obj_idx_to_id(i) not in new_ids and k in od["non_cond_frame_outputs"] \
                and k not in od["cond_frame_outputs"]:
            od["cond_frame_outputs"][k] = od["non_cond_frame_outputs"].pop(k)
            lent.append(od)
    try:
        return vm(inference_session=sess, frame_idx=k, reverse=reverse)
    finally:
        for od in lent:
            od["non_cond_frame_outputs"][k] = od["cond_frame_outputs"].pop(k)


def track(vm, vp, imgs, seeds, dev):
    """Masklets of one view over its observed frames, {id: (n, H, W) bool}: forward sweep with seed assignment (corrections
    or new masklets), then a backward sweep in a new session where each masklet enters at its first seed, for the frames
    before it; seed masks override at their frames. Also returns the number of seeds dropped at the MAX_MASKLETS cap."""
    n, W = len(imgs), imgs.shape[1]
    window = vm.num_maskmem - 1  # tracked frames the memory attention reads

    def session():  # frames and state on the host, one frame's vision features cached
        return vp.init_video_session(video=list(imgs), inference_device=dev, inference_state_device="cpu",
                                     video_storage_device="cpu", max_vision_features_cache_size=1)

    def post(o):  # {object id: (W, W) bool}
        return dict(zip(o.object_ids, vp.post_process_masks([o.pred_masks.cpu()], original_sizes=[[W, W]])[0][:, 0].numpy()))

    sess, out, first, first_mask, dropped = session(), {}, {}, {}, 0
    for k in range(n):
        cur = post(frame_masks(vm, sess, k)) if out else {}
        add = {}
        for m in seeds[k]:
            j = max(cur, key=lambda i: iou(cur[i], m), default=None)
            if j is not None and iou(cur[j], m) > MERGE_IOU:
                oid = j
            elif any((m & c).sum() > COVER * m.sum() for c in cur.values()):
                continue  # part of a tracked masklet: neither a duplicate nor a correction
            elif len(first) < MAX_MASKLETS:
                oid = len(first) + 1
                first[oid], first_mask[oid], out[oid] = k, m, np.zeros((n, W, W), bool)
            else:
                dropped += 1
                continue
            add[oid] = add[oid] | m if oid in add else m
        if add:
            vp.add_inputs_to_inference_session(inference_session=sess, frame_idx=k, obj_ids=list(add), input_masks=list(add.values()))
            cur = post(frame_masks(vm, sess, k, set(add)))
        for oid, m in cur.items():
            out[oid][k] = add.get(oid, m)
        tidy(sess, k, 1, window)
    del sess
    late = {oid: k0 for oid, k0 in first.items() if k0 > 0}
    if late:
        sess = session()
        for k in range(max(late.values()), -1, -1):
            new = [oid for oid, k0 in late.items() if k0 == k]  # appended after the tracked objects (HF's reverse flag)
            if new:
                vp.add_inputs_to_inference_session(inference_session=sess, frame_idx=k, obj_ids=new,
                                                   input_masks=[first_mask[oid] for oid in new])
            for oid, m in post(frame_masks(vm, sess, k, reverse=True)).items():
                if k < late[oid]:
                    out[oid][k] = m
            tidy(sess, k, -1, window)
        del sess
    return out, dropped


# (d) motion test ---------------------------------------------------------------------------------------------------------
def motion_test(mk, chg, shd, fg, D, M, f):
    """(moved, changed, shadow veto) of one masklet (n, H, W) given its view's attributed change / shadow, foreground and
    scale-corrected depth."""
    ks = [k for k in range(len(mk)) if mk[k].any()]
    if not ks:
        return False, False, False
    changed_ = sum(chg[k][mk[k]].mean() >= CHANGE_FRAC for k in ks) >= CHANGE_FRAMES
    moved = False
    if D is not None:
        cz = []
        for k in ks:
            P = unproject(D[k], M, f, mk[k])
            if len(P) >= MIN_PTS:
                z = D[k][mk[k]]
                cz.append((np.median(P, 0), np.median(z[(z > 0) & (z < 1e3)])))
        if len(cz) >= 2:
            C = np.array([c for c, _ in cz])
            thr = max(MOVE_M, MOVE_REL * float(np.median([z for _, z in cz])))
            moved = bool((np.linalg.norm(C - np.median(C, 0), axis=1) > thr).sum() >= CHANGE_FRAMES)
    sh, ch = sum(shd[k][mk[k]].sum() for k in ks), sum(chg[k][mk[k]].sum() for k in ks)
    fgf = sum(fg[k][mk[k]].sum() for k in ks) / sum(mk[k].sum() for k in ks)
    veto = bool(sh > ch and fgf < CHANGE_FRAC)
    return moved, bool(changed_), veto


# (e) cross-view vote -----------------------------------------------------------------------------------------------------
def voxel_keys(mk, D, M, f, link_ks):
    """int64 keys (frame position, VOX voxel) of a masklet's unprojected pixels at the link frames."""
    keys = []
    for j, k in enumerate(link_ks):
        P = unproject(D[k], M, f, mk[k])
        if len(P):
            q = np.clip(np.floor(P / VOX).astype(np.int64) + 2 ** 15, 0, 2 ** 16 - 1)
            keys.append(np.unique(((j * 2 ** 16 + q[:, 0]) * 2 ** 16 + q[:, 1]) * 2 ** 16 + q[:, 2]))
    return np.concatenate(keys) if keys else np.zeros(0, np.int64)


def dilate_keys(k):
    nb = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    return np.unique(np.concatenate([k + (dx * 2 ** 16 + dy) * 2 ** 16 + dz for dx, dy, dz in nb]))


def decode_keys(k):
    q = np.stack([(k >> 32) & 0xFFFF, (k >> 16) & 0xFFFF, k & 0xFFFF], -1) - 2 ** 15
    return k >> 48, (q + 0.5) * VOX


def vote(items, Ds, c2w, f, W, link_ks):
    """Groups of masklets linked across views by voxel IoU > LINK_IOU; a group is moving if accepted in >= VOTE_MIN views or
    in >= VOTE_FRAC of the views that see it. items: dicts with view, accepted, keys. Returns per-item moving flags, stats."""
    n = len(items)
    dk = [dilate_keys(it["keys"]) for it in items]
    allk = np.concatenate(dk) if n else np.zeros(0, np.int64)
    uk, inv = np.unique(allk, return_inverse=True)
    rows = np.repeat(np.arange(n), [len(k) for k in dk])
    A = sparse.csr_matrix((np.ones(len(allk), np.float32), (rows, inv)), shape=(n, len(uk)))
    inter = (A @ A.T).toarray()
    size = np.diag(inter)
    link = inter / np.maximum(size[:, None] + size[None] - inter, 1) > LINK_IOU
    ng, gid = csgraph.connected_components(sparse.csr_matrix(link), directed=False)
    moving, groups = np.zeros(n, bool), []
    for g in range(ng):
        mem = np.flatnonzero(gid == g)
        acc = {items[i]["view"] for i in mem if items[i]["accepted"]}
        see = {items[i]["view"] for i in mem}
        fr, P = decode_keys(np.unique(np.concatenate([items[i]["keys"] for i in mem])))
        for v in range(len(Ds)):
            if v in see or not len(P):
                continue
            M = c2w[v]
            pc = (P - M[:3, 3]) @ M[:3, :3]
            z = -pc[:, 2]
            u = np.floor(f * pc[:, 0] / np.maximum(z, 1e-6) + W / 2).astype(int)
            w = np.floor(-f * pc[:, 1] / np.maximum(z, 1e-6) + W / 2).astype(int)
            ok = (z > 0) & (u >= 0) & (u < W) & (w >= 0) & (w < W)
            vis = np.zeros(len(P), bool)
            for j, k in enumerate(link_ks):
                s = ok & (fr == j)
                vis[s] = np.abs(Ds[v][k][w[s], u[s]] - z[s]) < VIS_TOL
            if vis.mean() >= SEE_FRAC:
                see.add(v)
        mv = len(acc) >= VOTE_MIN or (len(see) and len(acc) >= VOTE_FRAC * len(see))
        moving[mem] = bool(mv)
        groups.append({"masklets": len(mem), "accepted_views": len(acc), "seeing_views": len(see), "moving": bool(mv)})
    return moving, groups


# run ---------------------------------------------------------------------------------------------------------------------
def run(cache, depth_src, out):
    import torch
    from transformers import Sam2Model, Sam2Processor, Sam2VideoModel, Sam2VideoProcessor
    t_start = time.time()
    no_depth = "--no-depth" in sys.argv
    if no_depth != (depth_src == "none"):
        sys.exit("depth 'none' goes with --no-depth, and only with it")
    if "--frames" not in sys.argv:
        sys.exit("--frames s4|future|t1,t2,... is required (the observed frames)")
    dev = arg("--device", "cuda")
    cidx, vids, cmeta, _ = scene_views(cache)
    sel = [int(i) for i in arg("--views", "").split(",") if i] or list(range(len(vids)))
    cidx, vids = [cidx[i] for i in sel], [vids[i] for i in sel]
    assert not set(vids) & set(NOVEL), f"novel views {sorted(set(vids) & set(NOVEL))} among the run's views"
    T = npz_header(cache, "depth")[0][1]
    frames = parse_frames(arg("--frames", None), T)
    mem = None
    if dev.startswith("cuda"):  # cap the allocator below what is free now, on the chosen card
        gi = torch.device(dev).index
        gi = torch.cuda.current_device() if gi is None else gi
        free, total = torch.cuda.mem_get_info(gi)
        mem = min(arg("--mem", 0.3, float), (free - GPU_SPARE) / total)
        if mem <= 0:
            sys.exit(f"only {free / 2 ** 30:.1f} GiB free on {dev}")
        torch.cuda.set_per_process_memory_fraction(float(mem), gi)
        print(f"{dev}: allocator capped at {mem * total / 2 ** 30:.1f} GiB ({free / 2 ** 30:.1f} of {total / 2 ** 30:.1f} GiB "
              f"free)", flush=True)
    d = np.load(cache)
    c2w, f, W = d["c2w"][cidx, 0], float(d["f"]), int(d["W"])
    rgb = load_rgb(cache, [(c, t) for c in cidx for t in frames], T)
    dep, depth_model = (None, None) if no_depth else load_depth(depth_src, cache, cmeta, cidx, vids, frames, T)
    v2 = "--v2" in sys.argv and not no_depth
    confs = [None] * len(vids)  # --v2: the estimate's confidence at the run's frames (none for a cache's depth)
    if v2 and depth_src != "cache" and "conf" in np.load(depth_src).files:
        dm = json.loads(str(np.load(depth_src)["meta"]))
        cshape = npz_header(depth_src, "conf")[0]
        fpos = {t: t for t in frames} if cshape[1] == T else {t: dm["frames"].index(t) for t in frames}
        vp = [dm["views"].index(v) for v in vids]
        Cf = npz_frames(depth_src, "conf", [(q, fpos[t]) for q in vp for t in frames])
        confs = [np.stack([Cf[q, fpos[t]].astype(np.float32) for t in frames]) for q in vp]
        del Cf
    timing = {"load": time.time() - t_start}
    print(f"{cmeta['scene']}: {len(vids)} views {vids}, {len(frames)} frames {frames[:4]}..{frames[-1]}, "
          f"depth {'none' if no_depth else f'{depth_src} ({depth_model})'}, load {timing['load']:.0f} s", flush=True)
    n = len(frames)
    Ds = [None if no_depth else np.stack([dep.pop((i, t)) for t in frames]) for i in range(len(vids))]
    timing.update(dict.fromkeys(("seeds", "track", "test"), 0.0))
    # stage 0: RGB evidence of every view, then each frame's depth scale pooled over the views (one Sim(3) fit per frame)
    t0 = time.time()
    rgb_ev, hist = [], 0
    for i, c in enumerate(cidx):
        sup, shd, chg, shd_t = rgb_evidence(np.stack([rgb[c, t] for t in frames]))
        rgb_ev.append([pack(a) for a in (sup, shd, chg, shd_t)])
        if not no_depth:
            hist = hist + log_ratio_hist(Ds[i], ~(sup | shd).any(0))
    scales = None if no_depth else frame_scales(hist, 0.01 * W * W)
    if not no_depth:
        Ds = [D / scales[:, None, None].astype(np.float32) for D in Ds]
        print(f"  depth scale per frame (removed): {np.round(scales, 4).tolist()}", flush=True)
    timing["evidence"] = time.time() - t0
    # stage 1, the image model only: seed masks of every view
    t0 = time.time()
    im, ip = Sam2Model.from_pretrained(MODEL).to(dev).eval(), Sam2Processor.from_pretrained(MODEL)
    timing["model"] = time.time() - t0
    counts, ev_packed, seeds_packed = [], [], []
    for i, (c, v) in enumerate(zip(cidx, vids)):
        imgs = np.stack([rgb[c, t] for t in frames])
        t0 = time.time()
        sup, shd, chg, shd_t = (unpack(a, W) for a in rgb_ev[i])
        fg = np.zeros_like(sup) if no_depth else depth_foreground(Ds[i])
        ev, never = seed_evidence(sup, shd, chg, fg)
        timing["evidence"] += time.time() - t0
        t0 = time.time()
        comps = [components(ev[k]) for k in range(n)]
        seeds = [seed_masks(im, ip, imgs[k], comps[k], ev[k], never, dev) for k in range(n)]
        timing["seeds"] += time.time() - t0
        ev_packed.append([pack(a) for a in (chg, shd_t, fg)])
        seeds_packed.append([[pack(m) for m in s] for s in seeds])
        rgb_ev[i] = None
        counts.append({"view": v, "seeds": sum(map(len, comps)), "seed_masks": sum(map(len, seeds)),
                       "fg_frac": round(float(fg.mean()), 4), "evidence_frac": round(float(ev.mean()), 4)})
        print(f"  view {v:2d}: {counts[-1]['seeds']} seeds -> {counts[-1]['seed_masks']} SAM masks; evidence "
              f"{counts[-1]['evidence_frac']:.3f}, depth foreground {counts[-1]['fg_frac']:.3f} of pixels "
              f"({time.time() - t_start:.0f} s)", flush=True)
    del im
    gc.collect()
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    # stage 2, the video model only: propagation, motion test, voxel keys
    t0 = time.time()
    vm, vp = Sam2VideoModel.from_pretrained(MODEL).to(dev).eval(), Sam2VideoProcessor.from_pretrained(MODEL)
    timing["model"] += time.time() - t0
    link_ks = sorted({int(round(x)) for x in np.linspace(0, n - 1, min(n, LINK_FRAMES))})
    items, packed = [], []
    for i, (c, v) in enumerate(zip(cidx, vids)):
        imgs = np.stack([rgb.pop((c, t)) for t in frames])
        seeds = [[unpack(m, W) for m in s] for s in seeds_packed[i]]
        t0 = time.time()
        with torch.inference_mode():
            mks, dropped = track(vm, vp, imgs, seeds, dev)
        cnt = dict(counts[i], masklets=len(mks), dropped_at_cap=dropped, track_s=round(time.time() - t0, 1), moved=0,
                   changed=0, vetoed=0, accepted=0, **({"fg_vetoed": 0} if v2 else {}))
        timing["track"] += time.time() - t0
        t0 = time.time()
        chg, shd, fg = (unpack(a, W) for a in ev_packed[i])
        if v2:  # pixels the vote's voxel keys may use: off depth discontinuities, confident (DA3)
            D = Ds[i]
            key_ok = ~np.stack([(ndimage.maximum_filter(z, 5) - ndimage.minimum_filter(z, 5)) > KEY_EDGE * z for z in D])
            if confs[i] is not None:
                key_ok &= confs[i] >= KEY_CONF
        for oid, mk in mks.items():
            moved, changed_, veto = motion_test(mk, chg, shd, fg, Ds[i], c2w[i], f)
            acc = (moved or changed_) and not veto
            if v2 and acc:  # depth-foreground veto: on the background surface in every frame (e.g. near-black shadows)
                ks = [k for k in range(n) if mk[k].any()]
                fr = [fg[k][mk[k] & (Ds[i][k] > 0)].mean() if (mk[k] & (Ds[i][k] > 0)).any() else 0.0 for k in ks]
                if max(fr, default=0.0) < FG_VETO:
                    acc = False
                    cnt["fg_vetoed"] += 1
            cnt["moved"] += moved
            cnt["changed"] += changed_
            cnt["vetoed"] += veto
            cnt["accepted"] += acc
            items.append({"view": i, "accepted": acc,
                          "keys": None if no_depth else voxel_keys(mk & key_ok if v2 else mk, Ds[i], c2w[i], f, link_ks)})
            packed.append(pack(mk))
        timing["test"] += time.time() - t0
        counts[i] = cnt
        print(f"  view {v:2d}: {cnt['masklets']} masklets ({dropped} seeds dropped at the cap, tracking "
              f"{cnt['track_s']:.0f} s); moved {cnt['moved']} changed {cnt['changed']} vetoed {cnt['vetoed']} accepted {cnt['accepted']} "
              f"({time.time() - t_start:.0f} s)", flush=True)
        del mks, seeds, chg, shd, fg
    t0 = time.time()
    if no_depth or not items:
        moving, groups = np.array([it["accepted"] for it in items], bool), []
    else:
        moving, groups = vote(items, Ds, c2w, f, W, link_ks)
    timing["vote"] = time.time() - t0
    S = np.zeros((len(vids), T, W, W), np.uint8)
    for it, mv, pk in zip(items, moving, packed):
        if mv:
            mk = unpack(pk, W)
            for k, t in enumerate(frames):
                S[it["view"], t][mk[k]] = 128
    timing["total"] = time.time() - t_start
    meta = {"scene": cmeta["scene"], "source": str(Path(cache).resolve()),
            "depth": "none" if no_depth else "cache" if depth_src == "cache" else str(Path(depth_src).resolve()),
            "depth_model": depth_model, "model": MODEL, "frames": frames, "frames_spec": arg("--frames", None), "views": vids,
            "cache_view_indices": cidx, "no_depth": no_depth,
            "depth_scales": None if scales is None else [round(float(x), 5) for x in scales], "device": dev,
            "version": "v2" if v2 else "v1", "conf_used": any(c is not None for c in confs),
            "gpu_mem_fraction": mem, "consts": CONSTS, "per_view": counts, "groups": groups,
            "moving_masklets": int(moving.sum()), "masklets": len(items), "timing_s": {k: round(x, 1) for k, x in timing.items()}}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, seg=S, meta=json.dumps(meta))
    print(f"ok {meta['scene']}: {meta['moving_masklets']}/{len(items)} masklets moving, {len(groups)} groups, "
          f"{(S == 128).sum() / (len(vids) * n * W * W):.4f} of pixels labelled; "
          + " ".join(f"{k} {x:.0f}s" for k, x in timing.items()), flush=True)


# score -------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    run(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
