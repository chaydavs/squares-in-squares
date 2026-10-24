"""Lengthen a strip gadget by copying bars from its periodic middle, then re-squeeze.

A strip gadget of width w is a stack of bars of w + 1 squares tilted by theta; consecutive
bars differ by a vertical shift p = 1 / cos(theta). Copying k bars at a middle cut makes a
strip k bars longer, exactly valid at total height h + f + k p. Re-squeezing to a smaller
integer height h' and a lower bloat F lets the bars retilt and the end caps adjust; the
box LP certifies the result. The strip runs along y (transpose first for wide gadgets).
"""
from __future__ import annotations

import math

import numpy as np

from . import gadget_search as GS
from .gadgets import Gadget, _corner_flags
from .geometry import fold_angle


def bar_mask(P: np.ndarray, min_tilt: float = 0.12) -> np.ndarray:
    return np.abs(fold_angle(P[:, 2])) > min_tilt


def bar_pitch(P: np.ndarray, mask: np.ndarray) -> float:
    Q = P[mask]
    dys = []
    for i in range(len(Q)):
        d = Q[:, 1] - Q[i, 1]
        ok = (np.abs(Q[:, 0] - Q[i, 0]) < 0.15) & (d > 0.5) & (d < 1.6)
        if ok.any():
            dys.append(d[ok].min())
    return float(np.median(dys)) if dys else math.nan


def insert_bars(g: Gadget, k: int) -> tuple[np.ndarray, float] | None:
    """Exact k bar insertion at the middle. Returns (P, new total height) or None."""
    P = g.P
    m = bar_mask(P)
    p = bar_pitch(P, m)
    if not math.isfinite(p):
        return None
    H = g.h + g.f
    ys = P[m, 1]
    c = 0.5 * (ys.min() + ys.max())
    slab = m & (P[:, 1] > c - p) & (P[:, 1] <= c)
    other_in_band = (~m) & (P[:, 1] > c - p - 1) & (P[:, 1] <= c + 1)
    if slab.sum() != g.w + 1 or other_in_band.any():
        return None  # the middle is not a clean single bar
    Q = P.copy()
    up = Q[:, 1] > c
    Q[up, 1] += k * p
    copies = []
    for i in range(1, k + 1):
        C = P[slab].copy()
        C[:, 1] += i * p
        copies.append(C)
    return np.vstack([Q] + copies), H + k * p


def stretch(g: Gadget, k: int, h_new: int, F_targets=None, seed: int = 0) -> Gadget | None:
    """Make a (w, h_new) gadget from g with k more bars; best certified bloat over targets."""
    r = insert_bars(g, k)
    if r is None:
        return None
    P0, Htot = r
    d_new = len(P0) - g.w * h_new
    if d_new <= 0:
        return None
    if F_targets is None:
        F_targets = [min(0.995, Htot - h_new)] + [g.f + x for x in (0.03, 0.015, 0.0, -0.01)]
    best = None
    rng = np.random.default_rng(seed)
    for F in sorted(set(F_targets), reverse=True):
        if not (0.4 < F < 0.999):
            continue
        Q = P0.copy()
        Q[:, 0] *= (g.w + F) / (g.w + g.f)
        Q[:, 1] *= (h_new + F) / Htot
        Q[:, :2] += rng.normal(0, 1e-4, (len(Q), 2))
        Q, e = GS.relax_box(Q, g.w + F, h_new + F, iters=400)
        if e > 2e-2:
            continue
        c = GS.certify(Q, g.w, h_new, trust=0.1, rounds=10)
        if c is None:
            continue
        R, Fc = c
        if Fc < 1 - 1e-9 and (best is None or Fc < best.f):
            tl, br = _corner_flags(R, g.w, h_new, Fc)
            best = Gadget(g.w, h_new, d_new, Fc, R, g.source + f"+stretch{k}", tl, br)
    return best
