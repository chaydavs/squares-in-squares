"""Transfer a packing to the next block by repeating one slab in each direction.

L border extension adds two plain strips at the edge, wasting 2 frac(s) per step. Instead,
cut the packing at y = c and x = d, shift everything beyond the cuts by one unit, and copy
the slab just below (left of) each cut into the gap. Inside upright grid regions this is an
exact row or column insertion. Where a slab crosses a band of leaning stacks, the copied
bar sits one unit from its original instead of one bar pitch 1/cos(theta), so it overlaps
slightly; the LP then spreads the band, which works when the band has slack. On average
the band adds 2 sin(theta) of waste per step instead of 2 frac(s).

Cuts are chosen so each slab holds one bar of every band it crosses; we simply try many
cut positions and keep what the LP certifies.
"""
from __future__ import annotations

import math

import numpy as np

from . import lp
from .geometry import min_gap, normalize, side


def insert_slabs(P: np.ndarray, c: float, d: float) -> np.ndarray:
    P = normalize(P)
    x, y = P[:, 0], P[:, 1]
    up = y > c
    right = x > d
    Q = P.copy()
    Q[up, 1] += 1.0
    Q[right, 0] += 1.0
    row = (y > c - 1) & (y <= c)
    col = (x > d - 1) & (x <= d)
    new = []
    R = P[row].copy(); R[:, 1] += 1.0; R[right[row], 0] += 1.0; new.append(R)
    C = P[col].copy(); C[:, 0] += 1.0; C[up[col], 1] += 1.0; new.append(C)
    X = P[row & col].copy(); X[:, :2] += 1.0; new.append(X)
    return np.vstack([Q] + new)


def grow(P: np.ndarray, cuts: int = 7, trust: float = 0.15, rng=None) -> list[tuple[int, float, np.ndarray]]:
    """Try a grid of cut positions; return (count, side, packing) for every LP certified result."""
    P = normalize(P)
    S = side(P)
    out = []
    fr = np.linspace(0.2, 0.8, cuts)
    for fc in fr:
        for fd in fr:
            c, d = fc * S, fd * S
            R = repair(insert_slabs(P, c, d), trust=trust)
            if R is not None:
                R = lp.compact(R, iters=12, trust=trust)
                out.append((len(R), side(R), R))
    return out


def repair(Q: np.ndarray, trust: float = 0.15, iters: int = 6) -> np.ndarray | None:
    """Push an slightly overlapping configuration apart by LP (axes re-chosen each round)."""
    R = normalize(Q)
    for _ in range(iters):
        if min_gap(R) > -1e-9:
            return R
        r = lp.solve(R, trust=trust)
        if r.status != 0:
            return None
        R = r.P
    return R if min_gap(R) > -1e-9 else None
