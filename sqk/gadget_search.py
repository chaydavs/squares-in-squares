"""Basin hopping on single gadgets: lower the bloat f of (w, h, d), or add a square.

A gadget is a small problem (tens to a few hundred squares), so many trials are cheap.
Each trial perturbs the incumbent, squeezes the box to a target bloat, relaxes the overlap
penalty there (L-BFGS), and lets the box LP (sqk.lp.solve(box=...)) certify the smallest
bloat for the resulting angles. Only certified, strictly better gadgets are kept.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import lp
from .gadgets import Gadget, _corner_flags
from .geometry import fold_angle, min_gap, neighbor_pairs
from .relax import DT, H, _corners

def box_energy(X, Y, T, Wx, Hy, I, J):
    cx, cy = _corners(X, Y, T)
    e = (torch.relu(-cx) ** 2 + torch.relu(cx - Wx) ** 2 + torch.relu(-cy) ** 2 + torch.relu(cy - Hy) ** 2).sum()
    if len(I):
        ti, tj = T[I], T[J]
        ax = torch.stack([torch.cos(ti), -torch.sin(ti), torch.cos(tj), -torch.sin(tj)], 1)
        ay = torch.stack([torch.sin(ti), torch.cos(ti), torch.sin(tj), torch.cos(tj)], 1)
        pa = cx[I][:, None, :] * ax[:, :, None] + cy[I][:, None, :] * ay[:, :, None]
        pb = cx[J][:, None, :] * ax[:, :, None] + cy[J][:, None, :] * ay[:, :, None]
        g = torch.maximum(pb.min(2).values - pa.max(2).values, pa.min(2).values - pb.max(2).values)
        e = e + (torch.relu(-g.max(1).values) ** 2).sum()
    return e


def relax_box(P, Wx, Hy, iters=200):
    pr = neighbor_pairs(P, 1.9)
    I, J = torch.tensor(pr[:, 0]), torch.tensor(pr[:, 1])
    X = torch.tensor(P[:, 0], dtype=DT, requires_grad=True)
    Y = torch.tensor(P[:, 1], dtype=DT, requires_grad=True)
    T = torch.tensor(P[:, 2], dtype=DT, requires_grad=True)
    opt = torch.optim.LBFGS([X, Y, T], lr=1.0, max_iter=iters, tolerance_grad=1e-14, tolerance_change=1e-16,
                            history_size=40, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        e = box_energy(X, Y, T, Wx, Hy, I, J)
        e.backward()
        return e

    opt.step(closure)
    with torch.no_grad():
        e = float(box_energy(X, Y, T, Wx, Hy, I, J))
    return np.stack([X.detach().numpy(), Y.detach().numpy(), T.detach().numpy()], 1), e


def certify(P, w, h, trust=0.08, rounds=6):
    """LP repair then minimize the bloat; returns (P, F) or None."""
    Q = P
    F = math.inf
    for _ in range(rounds):
        r = lp.solve(Q, trust=trust, box=(w, h))
        if r.status != 0:
            return None
        if min_gap(r.P) > -1e-9:
            if r.L < F - 1e-12:
                Q, F = r.P, r.L
                continue
            break
        Q = r.P
    if not math.isfinite(F):
        return None
    return Q, F


def trial(g: Gadget, rng, squeeze=0.004, ang=0.01, pos=0.01, insert=False):
    P = g.P.copy()
    w, h = g.w, g.h
    tilt = np.abs(fold_angle(P[:, 2])) > 1e-6
    P[tilt, 2] += rng.normal(0, ang, tilt.sum()) + rng.normal(0, ang)  # local plus coherent
    P[:, :2] += rng.normal(0, pos, (len(P), 2))
    F = g.f - squeeze
    d = g.d
    if insert:
        F = min(0.995, g.f + 0.05)
        # drop the new square at the emptiest of a few random spots, at the mean tilt
        best_pt, best_d = None, -1
        for _ in range(64):
            pt = rng.uniform([0.5, 0.5], [w + F - 0.5, h + F - 0.5])
            dd = np.min(np.hypot(P[:, 0] - pt[0], P[:, 1] - pt[1]))
            if dd > best_d:
                best_pt, best_d = pt, dd
        th = float(np.mean(P[tilt, 2])) if tilt.any() else 0.0
        P = np.vstack([P, [best_pt[0], best_pt[1], th]])
        d += 1
    sx, sy = (w + F) / (w + g.f), (h + F) / (h + g.f)
    P[:, 0] *= sx
    P[:, 1] *= sy
    Q, e = relax_box(P, w + F, h + F)
    if e > 5e-2:
        return None
    c = certify(Q, w, h)
    if c is None:
        return None
    R, Fc = c
    if Fc >= 1 - 1e-9:
        return None
    tl, br = _corner_flags(R, w, h, Fc)
    return Gadget(w, h, d, Fc, R, g.source + "+gs", tl, br)


def improve(g: Gadget, trials: int = 50, seed: int = 0, verbose: bool = False) -> Gadget:
    rng = np.random.default_rng(seed)
    best = g
    sq = 0.004
    for t in range(trials):
        r = trial(best, rng, squeeze=sq, ang=rng.choice([0.002, 0.01, 0.03]), pos=rng.choice([0.002, 0.01, 0.03]))
        if r is not None and r.f < best.f - 1e-9:
            if verbose:
                print(f"    trial {t}: f {best.f:.6f} -> {r.f:.6f}", flush=True)
            best = r
            sq *= 1.5
        else:
            sq = max(sq * 0.7, 2e-4)
    return best
