"""Tilted core Monte Carlo with exact upright refill (our FTMC).

State: the tilted core T of a gadget inside a fixed box (w + F) x (h + F). The score of a
state is |T| + (maximum upright refill around T). Moves change the core (rigid shifts and
rotations, global retilt, dropping a boundary core square, extending a bar by one square);
after each move the core alone is relaxed in the box (penalty L-BFGS), then uprights are
refilled exactly (sqk.fill). Moves that do not lower the score are kept (plateau moves are
how the core walks to a better basin). When the score reaches w h + d_target, the full
configuration is certified by the box LP.
"""
from __future__ import annotations

import math

import numpy as np

from . import fill
from . import gadget_search as GS
from .gadgets import Gadget, _corner_flags
from .geometry import fold_angle, min_gap, neighbor_pairs


def _boundary(T: np.ndarray) -> np.ndarray:
    pr = neighbor_pairs(T, 1.45)
    deg = np.bincount(pr.ravel(), minlength=len(T)) if len(pr) else np.zeros(len(T), int)
    return np.nonzero(deg <= np.percentile(deg, 25))[0]


def move(T: np.ndarray, rng, W: float, H: float) -> np.ndarray:
    T = T.copy()
    r = rng.random()
    if r < 0.30:  # rigid shift
        T[:, :2] += rng.normal(0, rng.choice([0.01, 0.04, 0.12]), 2)
    elif r < 0.50:  # rigid rotation about the centroid
        a = rng.normal(0, rng.choice([0.002, 0.008]))
        c = T[:, :2].mean(0)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        T[:, :2] = c + (T[:, :2] - c) @ R.T
        T[:, 2] += a
    elif r < 0.65:  # global retilt only
        T[:, 2] += rng.normal(0, 0.004)
    elif r < 0.82:  # drop a boundary core square
        b = _boundary(T)
        if len(b):
            T = np.delete(T, rng.choice(b), 0)
    else:  # extend: copy a boundary square one unit along one of its own axes
        b = _boundary(T)
        if len(b):
            i = rng.choice(b)
            th = T[i, 2] + rng.integers(4) * math.pi / 2
            new = T[i].copy()
            new[:2] += [math.cos(th), math.sin(th)]
            if 0.5 < new[0] < W - 0.5 and 0.5 < new[1] < H - 0.5:
                T = np.vstack([T, new])
    return T


def score(T: np.ndarray, W: float, H: float, tl: float = 5.0):
    R = fill.refill(T, W, H, time_limit=tl)
    return len(R), R


def search(g: Gadget, F: float, d_target: int, steps: int = 200, seed: int = 0, tau: float = 0.12,
           verbose: bool = False) -> Gadget | None:
    rng = np.random.default_rng(seed)
    W, H = g.w + F, g.h + F
    need = g.w * g.h + d_target
    near = np.abs(fold_angle(g.P[:, 2])) < tau
    T = g.P[~near].copy()
    cur, R = score(T, W, H)
    best = cur
    for st in range(steps):
        T2 = move(T, rng, W, H)
        if len(T2) == 0:
            continue
        T2, e = GS.relax_box(T2, W, H, iters=120)
        if e > 1e-6:
            continue
        c2, R2 = score(T2, W, H)
        if c2 >= cur:
            T, cur, R = T2, c2, R2
            if verbose and c2 > best:
                print(f"    step {st}: count {c2} (need {need})", flush=True)
            best = max(best, c2)
        if cur >= need:
            c = GS.certify(R, g.w, g.h, trust=0.05, rounds=8)
            if c is not None and c[1] <= F + 1e-9:
                Rc, Fc = c
                tl, br = _corner_flags(Rc, g.w, g.h, Fc)
                return Gadget(g.w, g.h, len(Rc) - g.w * g.h, Fc, Rc, g.source + "+core", tl, br)
    return None
