"""Kaggle submission I/O.

Format (verified against the competition pages, Oct 2026):
    id,n,square_id,x,y,angle      one row per square, id = 0..rows-1, angle in radians.

Export policy: every packing is translated to the origin, its centres are scaled apart by
(1 + EPS) so each exact contact opens by at least EPS, and values are written as float64
reprs. Then the file is identical whether Kaggle passes our strings or a pandas float64
round trip to the metric, and no contact depends on more than double precision.
EPS = 1e-12 costs about 1e-9 in total score.
"""
from __future__ import annotations

import csv
import math

import numpy as np

from .geometry import inflate, min_gap, normalize, side

EPS = 1e-12
COLUMNS = ["id", "n", "square_id", "x", "y", "angle"]


def export_ready(P: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Inflate by eps, or by enough to also cover a residual float overlap (LP tolerance).

    Inflation by e opens every contact by at least e, so e = eps + 4 * overlap suffices.
    """
    P = normalize(P)
    g0 = min_gap(P) if len(P) > 1 else np.inf
    e = eps if g0 >= 0 else eps + 4.0 * (-g0)
    if e > 1e-7:
        raise ValueError(f"packing overlaps by {-g0:.3e}; refusing to hide it by inflation")
    Q = normalize(inflate(P, e))
    g = min_gap(Q) if len(Q) > 1 else np.inf
    if g < 0.5 * eps:
        raise ValueError(f"after inflation by {e:.2e} min gap is {g:.3e} < {0.5 * eps:.1e}")
    return Q


def write(packings: dict[int, np.ndarray], path: str, eps: float = EPS) -> dict[int, float]:
    """Write all packings; returns the float side of each exported puzzle."""
    sides = {}
    rid = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for n in sorted(packings):
            Q = export_ready(packings[n], eps)
            sides[n] = side(Q)
            for k, (x, y, t) in enumerate(Q):
                w.writerow([rid, n, k, repr(float(x)), repr(float(y)), repr(float(t))])
                rid += 1
    return sides


def read(path: str) -> dict[int, list[dict]]:
    """Rows grouped by n, values kept as the exact strings in the file."""
    out: dict[int, list[dict]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out.setdefault(int(row["n"]), []).append(row)
    return out


def read_packings(path: str) -> dict[int, np.ndarray]:
    rows = read(path)
    out = {}
    for n, rs in rows.items():
        rs = sorted(rs, key=lambda r: int(r["square_id"]))
        out[n] = np.array([[float(r["x"]), float(r["y"]), float(r["angle"])] for r in rs])
    return out


def total_score(sides: dict[int, float], nmax: int = 1000) -> float:
    tot = 0.0
    for n in range(1, nmax + 1):
        L = sides.get(n, math.ceil(math.sqrt(n) - 1e-12))
        tot += L / math.sqrt(n) - 1
    return tot
