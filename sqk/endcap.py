"""Insert one more square into a gadget end region (the strip end cap) at the same bloat.

The periodic middle of a strip gadget is frozen; only squares near one end are free. A new
square is dropped into the largest voids of that region at assorted angles, the free
squares are relaxed against the frozen ones and the box walls (penalty L-BFGS), and any
zero energy state is certified by the box LP over the whole gadget.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import gadget_search as GS
from .gadgets import Gadget, _corner_flags
from .geometry import fold_angle, neighbor_pairs
from .relax import DT, _corners


def relax_partial(P: np.ndarray, free: np.ndarray, W: float, H: float, iters: int = 300):
    pr = neighbor_pairs(P, 1.9)
    keep = free[pr[:, 0]] | free[pr[:, 1]]
    pr = pr[keep]
    I, J = torch.tensor(pr[:, 0]), torch.tensor(pr[:, 1])
    fidx = torch.tensor(np.nonzero(free)[0])
    base = torch.tensor(P, dtype=DT)
    V = base[fidx].clone().requires_grad_(True)

    def full():
        Q = base.clone()
        Q[fidx] = V
        return Q

    def energy():
        Q = full()
        X, Y, T = Q[:, 0], Q[:, 1], Q[:, 2]
        cx, cy = _corners(X[fidx], Y[fidx], T[fidx])
        e = (torch.relu(-cx) ** 2 + torch.relu(cx - W) ** 2 + torch.relu(-cy) ** 2 + torch.relu(cy - H) ** 2).sum()
        if len(I):
            cxa, cya = _corners(X, Y, T)
            ti, tj = T[I], T[J]
            ax = torch.stack([torch.cos(ti), -torch.sin(ti), torch.cos(tj), -torch.sin(tj)], 1)
            ay = torch.stack([torch.sin(ti), torch.cos(ti), torch.sin(tj), torch.cos(tj)], 1)
            pa = cxa[I][:, None, :] * ax[:, :, None] + cya[I][:, None, :] * ay[:, :, None]
            pb = cxa[J][:, None, :] * ax[:, :, None] + cya[J][:, None, :] * ay[:, :, None]
            g = torch.maximum(pb.min(2).values - pa.max(2).values, pa.min(2).values - pb.max(2).values)
            e = e + (torch.relu(-g.max(1).values) ** 2).sum()
        return e

    opt = torch.optim.LBFGS([V], lr=1.0, max_iter=iters, tolerance_grad=1e-15, tolerance_change=1e-18,
                            history_size=40, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        e = energy()
        e.backward()
        return e

    opt.step(closure)
    with torch.no_grad():
        e = float(energy())
        Q = full().numpy()
    return Q, e


def voids(P: np.ndarray, region: tuple[float, float, float, float], k: int, rng) -> np.ndarray:
    """k candidate points in the region, farthest from existing centres (best of random draws)."""
    x0, y0, x1, y1 = region
    pts = rng.uniform([x0, y0], [x1, y1], (400, 2))
    d = np.min(np.hypot(pts[:, None, 0] - P[None, :, 0], pts[:, None, 1] - P[None, :, 1]), 1)
    return pts[np.argsort(-d)[:k]]


def insert_one(g: Gadget, F: float, end: str, tries: int = 40, depth: float = 6.0, seed: int = 0,
               verbose: bool = False) -> Gadget | None:
    """Try to fit g.d + 1 squares in (w + F) x (h + F) by changing one end region only."""
    rng = np.random.default_rng(seed)
    W, H = g.w + F, g.h + F
    P = g.P.copy()
    P[:, 0] *= W / (g.w + g.f)
    P[:, 1] *= H / (g.h + g.f)
    if end == "bottom":
        region = (0.0, 0.0, W, depth)
        free = P[:, 1] < depth + 0.8
    else:
        region = (0.0, H - depth, W, H)
        free = P[:, 1] > H - depth - 0.8
    tilt = np.abs(fold_angle(P[:, 2]))
    bar_t = float(np.median(P[tilt > 0.12, 2])) if (tilt > 0.12).any() else 0.0
    for pt in voids(P, region, tries, rng):
        for th in (0.0, bar_t, bar_t / 2, rng.uniform(-0.8, 0.8)):
            Q = np.vstack([P, [pt[0], pt[1], th]])
            fr = np.append(free, True)
            Q[fr, :2] += rng.normal(0, 0.003, (fr.sum(), 2))
            R, e = relax_partial(Q, fr, W, H)
            if e > 1e-12:
                continue
            c = GS.certify(R, g.w, g.h, trust=0.03, rounds=8)
            if c is None or c[1] > F + 1e-9:
                continue
            Rc, Fc = c
            tl, br = _corner_flags(Rc, g.w, g.h, Fc)
            if verbose:
                print(f"    inserted at {pt.round(2)} angle {th:.3f}: F={Fc:.6f}", flush=True)
            return Gadget(g.w, g.h, g.d + 1, Fc, Rc, g.source + f"+cap({end})", tl, br)
    return None
