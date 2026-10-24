"""Exact refill of upright squares around a fixed tilted core (max independent set, MILP).

Given tilted squares T inside the box [0, Wb] x [0, Hb], find the largest set of axis
aligned unit squares that fit in the box, disjoint from T and from each other.

Candidates. An optimal set can be pushed left and down until every upright touches a wall,
another upright, or a tilted square; we approximate that closure with:
  * the four corner lattices (x from the left or right wall, y from the bottom or top),
  * contact positions: for every tilted square and every lattice row (column), the upright
    that touches it from the left/right (below/above), plus a few uprights stacked beyond it.
Conflicts. Two uprights overlap iff |dx| < 1 and |dy| < 1. Every conflicting pair lies in
one of the two windows [x, x+1) x [y, y+1) or [x, x+1) x (y-1, y) anchored at the left one,
and each window is a clique, so 2 clique constraints per candidate cover all conflicts.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .geometry import corners

EPS = 1e-9


def _band_extent(C: np.ndarray, lo: float, hi: float, axis: int):
    """Extent along the other axis of each polygon (m, 4, 2) clipped to lo <= coord[axis] <= hi.
    Returns (min, max) arrays; NaN where the polygon misses the band."""
    other = 1 - axis
    m = len(C)
    pts = [C]  # vertices
    for k in range(4):
        a, b = C[:, k], C[:, (k + 1) % 4]
        for line in (lo, hi):
            da = a[:, axis] - line
            db = b[:, axis] - line
            cross = (da * db) < 0
            t = np.where(cross, da / np.where(cross, da - db, 1.0), np.nan)
            p = a + t[:, None] * (b - a)
            pts.append(p[:, None, :])
    P = np.concatenate(pts, 1)  # (m, k, 2)
    inside = (P[..., axis] >= lo - 1e-12) & (P[..., axis] <= hi + 1e-12) & ~np.isnan(P[..., 0])
    vals = np.where(inside, P[..., other], np.nan)
    with np.errstate(all="ignore"):
        return np.nanmin(vals, 1), np.nanmax(vals, 1)


def candidates(T: np.ndarray, Wb: float, Hb: float, stack: int = 3) -> np.ndarray:
    xs_l = np.arange(0.5, Wb - 0.5 + EPS, 1.0)
    xs_r = Wb - 0.5 - np.arange(0, Wb - 1 + EPS, 1.0)
    ys_b = np.arange(0.5, Hb - 0.5 + EPS, 1.0)
    ys_t = Hb - 0.5 - np.arange(0, Hb - 1 + EPS, 1.0)
    X = np.concatenate([xs_l, xs_r])
    Y = np.concatenate([ys_b, ys_t])
    cand = [np.array(np.meshgrid(X, Y)).reshape(2, -1).T]
    if len(T):
        C = corners(T)
        for y in np.unique(np.round(Y, 12)):
            mn, mx = _band_extent(C, y - 0.5, y + 0.5, axis=1)
            ok = ~np.isnan(mn)
            for s in range(stack + 1):
                cand.append(np.stack([mx[ok] + 0.5 + s, np.full(ok.sum(), y)], 1))
                cand.append(np.stack([mn[ok] - 0.5 - s, np.full(ok.sum(), y)], 1))
        for x in np.unique(np.round(X, 12)):
            mn, mx = _band_extent(C, x - 0.5, x + 0.5, axis=0)
            ok = ~np.isnan(mn)
            for s in range(stack + 1):
                cand.append(np.stack([np.full(ok.sum(), x), mx[ok] + 0.5 + s], 1))
                cand.append(np.stack([np.full(ok.sum(), x), mn[ok] - 0.5 - s], 1))
    Q = np.concatenate(cand)
    Q = Q[(Q[:, 0] >= 0.5 - EPS) & (Q[:, 0] <= Wb - 0.5 + EPS) & (Q[:, 1] >= 0.5 - EPS) & (Q[:, 1] <= Hb - 0.5 + EPS)]
    Q = np.unique(np.round(Q, 10), axis=0)
    if len(T) and len(Q):
        Q = Q[_clear_of(T, Q)]
    return Q


def _clear_of(T: np.ndarray, Q: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Mask of upright candidates that do not overlap any tilted square (separating axes)."""
    keep = np.ones(len(Q), dtype=bool)
    CT = corners(T)
    c, s = np.cos(T[:, 2]), np.sin(T[:, 2])
    for k in range(len(T)):
        near = np.nonzero((np.abs(Q[:, 0] - T[k, 0]) < 1.21) & (np.abs(Q[:, 1] - T[k, 1]) < 1.21) & keep)[0]
        if not len(near):
            continue
        q = Q[near]
        cq = np.stack([q + [0.5, 0.5], q + [-0.5, 0.5], q + [-0.5, -0.5], q + [0.5, -0.5]], 1)  # (r, 4, 2)
        sep = np.full(len(near), -np.inf)
        for ax in (np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([c[k], s[k]]), np.array([-s[k], c[k]])):
            pa = CT[k] @ ax
            pb = cq @ ax
            g = np.maximum(pb.min(1) - pa.max(), pa.min() - pb.max(1))
            sep = np.maximum(sep, g)
        keep[near[sep < -tol]] = False
    return keep


def max_independent(Q: np.ndarray, time_limit: float = 20.0) -> np.ndarray:
    """Indices of a maximum set of non overlapping uprights among candidates Q."""
    m = len(Q)
    if m == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(Q[:, 0], kind="stable")
    xs = Q[order, 0]
    rows, cols = [], []
    r = 0
    for a in range(m):
        i = order[a]
        hi = np.searchsorted(xs, Q[i, 0] + 1 - 1e-9, side="left")
        js = order[a:hi]
        dy = Q[js, 1] - Q[i, 1]
        for sel in ((dy >= 0) & (dy < 1 - 1e-9), (dy <= 0) & (dy > -1 + 1e-9)):
            mem = js[sel]
            if len(mem) > 1:
                rows += [r] * len(mem)
                cols += list(mem)
                r += 1
    c = -np.ones(m)
    cons = []
    if r:
        A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(r, m)).tocsr()
        cons = [LinearConstraint(A, -np.inf, 1)]
    res = milp(c, constraints=cons, integrality=np.ones(m), bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "disp": False})
    if res.x is None:
        return np.zeros(0, dtype=np.int64)
    return np.nonzero(res.x > 0.5)[0]


def refill(T: np.ndarray, Wb: float, Hb: float, time_limit: float = 20.0) -> np.ndarray:
    """Tilted core plus the maximum upright fill, as one packing (n, 3)."""
    Q = candidates(T, Wb, Hb)
    pick = max_independent(Q, time_limit)
    U = np.column_stack([Q[pick], np.zeros(len(pick))])
    return np.vstack([T, U]) if len(T) else U
