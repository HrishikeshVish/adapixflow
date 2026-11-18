"""Velocity fields and advection (Sec. 3.3): node fields, tiled inference on large windows, and the midpoint integrator.

Positions are in grid (voxel) coordinates. The velocity at time t in [m, m + 1] is linear between the node fields V_m and
V_{m+1}, and is sampled trilinearly in space; positions outside the grid take the velocity at its boundary.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn.functional as F

from .model import TIME_UNIT


def sample_at(field, pos):
    """Trilinear samples of field (B, c, X, Y, Z) at voxel coordinates pos (B, ..., 3), clamped to the grid border."""
    size = torch.tensor(field.shape[2:], device=pos.device, dtype=pos.dtype)
    grid = (2 * pos / (size - 1) - 1).flip(-1)  # grid_sample expects (z, y, x) order in [-1, 1]
    return F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)


@torch.no_grad()
def node_fields(model, ctx, s, n, tiling=None):
    """The n + 1 node fields V_0..V_n (each (1, 3, X, Y, Z)) of one context (1, c, X, Y, Z) at spacing s.

    tiling: None (one pass), ("centered", tile, keep) or ("margin", tile, margin) for windows larger than `tile` voxels.
    "centered" runs overlapping tile^3 blocks at stride `keep` and takes each voxel from the block whose centre is
    nearest (comparison scenes); "margin" runs blocks at stride tile - 2 * margin and keeps each block's interior
    (held-out evaluation). Both give every voxel enough context for the U-Net's receptive field of about 60 voxels."""
    dev = next(model.parameters()).device
    tau = torch.zeros(1, device=dev)
    s_t = torch.full((1,), s / TIME_UNIT, device=dev)
    n_out = 3 * (n + 1)
    shape = ctx.shape[2:]
    if tiling is None or max(shape) <= tiling[1]:
        V = model(ctx.to(dev), tau, s_t)[:, :n_out]
    elif tiling[0] == "centered":
        V = _tiled_centered(model, ctx, tau, s_t, n_out, tiling[1], tiling[2])
    elif tiling[0] == "margin":
        V = _tiled_margin(model, ctx.to(dev), tau, s_t, n_out, tiling[1], tiling[2])
    else:
        raise ValueError(f"unknown tiling {tiling}")
    return [V[:, 3 * m:3 * m + 3] for m in range(n + 1)]


def _tiled_centered(model, ctx, tau, s_t, n_out, tile, keep):
    dev = tau.device
    shape, axes = ctx.shape[2:], []
    for L in shape:
        if L <= tile:
            axes.append([(0, L, 0, L)])
            continue
        starts = list(range(0, L - tile, keep)) + [L - tile]
        own = np.argmin(np.abs(np.arange(L)[:, None] + 0.5 - (np.array(starts) + tile / 2)[None]), 1)
        axes.append([(a, a + tile, np.flatnonzero(own == i)[0], np.flatnonzero(own == i)[-1] + 1)
                     for i, a in enumerate(starts) if (own == i).any()])
    out = torch.zeros((1, n_out, *shape))
    for (ax, bx, x0, x1), (ay, by, y0, y1), (az, bz, z0, z1) in itertools.product(*axes):
        V = model(ctx[:, :, ax:bx, ay:by, az:bz].to(dev), tau, s_t)[:, :n_out].cpu()
        out[:, :, x0:x1, y0:y1, z0:z1] = V[:, :, x0 - ax:x1 - ax, y0 - ay:y1 - ay, z0 - az:z1 - az]
    return out.to(dev)


def _tiled_margin(model, ctx, tau, s_t, n_out, tile, margin):
    X = ctx.shape[2:]
    st = [sorted(set(list(range(0, max(d - tile, 0) + 1, tile - 2 * margin)) + [max(d - tile, 0)])) for d in X]
    out = None
    for a0 in st[0]:
        for a1 in st[1]:
            for a2 in st[2]:
                a_ = (a0, a1, a2)
                sl = tuple(slice(a, a + min(tile, d)) for a, d in zip(a_, X))
                V = model(ctx[(slice(None), slice(None)) + sl], tau, s_t)[:, :n_out]
                if out is None:
                    out = torch.zeros((1, V.shape[1], *X), device=V.device)
                keep = tuple(slice(0 if a == 0 else margin, min(tile, d) if a + tile >= d else tile - margin)
                             for a, d in zip(a_, X))
                dst = tuple(slice(a + k.start, a + k.stop) for a, k in zip(a_, keep))
                out[(slice(None), slice(None)) + dst] = V[(slice(None), slice(None)) + keep]
    return out


def advect(Vn, pts, t0, t1, substeps):
    """Points (N, 3) carried from time t0 to t1 (t1 < t0: backward) along the field of node fields Vn.

    Uses `substeps` midpoint steps per time step, with both stages evaluated at the substep's mid-time (Eq. 5)."""
    n_steps = int(round(abs(t1 - t0) * substeps))
    if n_steps == 0 or len(pts) == 0:
        return pts
    dt, p = (t1 - t0) / n_steps, pts[None, :, None, None, :]

    def v(q, t):
        m = min(int(t), len(Vn) - 2)
        a = t - m
        f = lambda V: sample_at(V, q)[:, :, :, 0, 0].permute(0, 2, 1)[:, :, None, None, :]  # noqa: E731
        return (1 - a) * f(Vn[m]) + a * f(Vn[m + 1])

    for i in range(n_steps):
        tm = t0 + (i + 0.5) * dt
        p = p + dt * v(p + 0.5 * dt * v(p, tm), tm)
    return p[0, :, 0, 0]


def substep_displacements(Vn, n, substeps):
    """Per-substep displacement fields for training: n time steps with `substeps` substeps each, the velocity taken at
    each substep's mid-time and linear in time between nodes. Returns a list of n * substeps fields (B, 3, X, Y, Z)."""
    D = []
    for i in range(n * substeps):
        t = (i + 0.5) / substeps
        m, a = int(t), t - int(t)
        D.append(((1 - a) * Vn[m] + a * Vn[m + 1]) / substeps)
    return D


def advect_trajectory(D, pts, substeps, sign):
    """Points (N, 3) carried through the per-substep displacements D with midpoint steps.

    sign +1: start at t = 0 and return the positions at steps 1..n (list); sign -1: start at t = n and return the positions
    at steps n - 1..0 (dict step -> points)."""
    p = pts[None, :, None, None, :]
    v_at = lambda f, q: sample_at(f, q)[:, :, :, 0, 0].permute(0, 2, 1)[:, :, None, None, :]  # noqa: E731
    if sign > 0:
        out = []
        for i in range(len(D)):
            p = p + v_at(D[i], p + 0.5 * v_at(D[i], p))
            if (i + 1) % substeps == 0:
                out.append(p[0, :, 0, 0])
        return out
    out = {}
    for i in reversed(range(len(D))):
        p = p - v_at(D[i], p - 0.5 * v_at(D[i], p))
        if i % substeps == 0:
            out[i // substeps] = p[0, :, 0, 0]
    return out
