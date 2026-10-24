"""Float64 geometry for packings of unit squares.

A packing is an (n, 3) float array of rows (cx, cy, theta): centre and rotation in radians,
the same convention as the Kaggle metric. Nothing here certifies a contact; the official
metric (interval arithmetic) is the final judge, see sqk.validate.
"""
from __future__ import annotations

import math

import numpy as np

H = 0.5  # half side
REACH = math.sqrt(2.0)  # two unit squares can only touch if centres are closer than this


def as_packing(P) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"packing must be (n, 3), got {P.shape}")
    return P


def fold_angle(theta: np.ndarray) -> np.ndarray:
    """A unit square is invariant under quarter turns: map angles into [-pi/4, pi/4)."""
    q = math.pi / 2
    return np.mod(theta + math.pi / 4, q) - math.pi / 4


def corners(P: np.ndarray) -> np.ndarray:
    """(n, 4, 2) corners, in the metric's order."""
    c, s = np.cos(P[:, 2]), np.sin(P[:, 2])
    dx1, dy1 = H * c - H * s, H * s + H * c
    dx2, dy2 = -H * c - H * s, -H * s + H * c
    x, y = P[:, 0], P[:, 1]
    return np.stack(
        [
            np.stack([x + dx1, y + dy1], 1),
            np.stack([x + dx2, y + dy2], 1),
            np.stack([x - dx1, y - dy1], 1),
            np.stack([x - dx2, y - dy2], 1),
        ],
        1,
    )


def half_extent(theta: np.ndarray) -> np.ndarray:
    """Half width of a unit square's axis aligned bounding box."""
    return H * (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))


def bbox(P: np.ndarray) -> tuple[float, float, float, float]:
    C = corners(P).reshape(-1, 2)
    lo, hi = C.min(0), C.max(0)
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


def side(P: np.ndarray) -> float:
    """The metric's L: max(span_x, span_y) over all corners."""
    x0, y0, x1, y1 = bbox(P)
    return max(x1 - x0, y1 - y0)


def normalize(P: np.ndarray) -> np.ndarray:
    """Translate so the bounding box starts at the origin."""
    x0, y0, _, _ = bbox(P)
    Q = P.copy()
    Q[:, 0] -= x0
    Q[:, 1] -= y0
    return Q


def neighbor_pairs(P: np.ndarray, reach: float = REACH + 1e-9) -> np.ndarray:
    """All pairs (i < j) whose centres are within `reach`, by cell hashing. O(n)."""
    n = len(P)
    if n < 2:
        return np.zeros((0, 2), dtype=np.int64)
    cell = np.floor(P[:, :2] / reach).astype(np.int64)
    cell -= cell.min(0)
    w = int(cell[:, 0].max()) + 3
    key = (cell[:, 0] + 1) * 1 + (cell[:, 1] + 1) * w
    order = np.argsort(key, kind="stable")
    skey = key[order]
    out = []
    for dk in (0, 1, w - 1, w, w + 1):  # half of the 3x3 stencil, the rest by symmetry
        lo = np.searchsorted(skey, skey + dk, side="left")
        hi = np.searchsorted(skey, skey + dk, side="right")
        for a in range(len(order)):
            for b in range(lo[a], hi[a]):
                if dk == 0 and b <= a:
                    continue
                out.append((order[a], order[b]))
    if not out:
        return np.zeros((0, 2), dtype=np.int64)
    pr = np.array(out, dtype=np.int64)
    pr = np.sort(pr, axis=1)
    pr = np.unique(pr, axis=0)
    d = np.hypot(P[pr[:, 0], 0] - P[pr[:, 1], 0], P[pr[:, 0], 1] - P[pr[:, 1], 1])
    return pr[d < reach]


def axes_of(theta: np.ndarray) -> np.ndarray:
    """(m, 2, 2): the two edge normals of each square."""
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([np.stack([c, s], 1), np.stack([-s, c], 1)], 1)


def pair_gaps(P: np.ndarray, pairs: np.ndarray, return_axis: bool = False):
    """Separating axis gap per pair: > 0 separated, 0 touching, < 0 overlap depth.

    With return_axis, also the axis (unit vector, oriented from i to j) attaining it.
    """
    if len(pairs) == 0:
        z = np.zeros(0)
        return (z, np.zeros((0, 2))) if return_axis else z
    i, j = pairs[:, 0], pairs[:, 1]
    C = corners(P)
    A, B = C[i], C[j]
    cand = np.concatenate([axes_of(P[i, 2]), axes_of(P[j, 2])], 1)  # (m, 4, 2)
    pa = np.einsum("mkd,mad->mak", A, cand)  # (m, 4 axes, 4 corners)
    pb = np.einsum("mkd,mad->mak", B, cand)
    g_ij = pb.min(2) - pa.max(2)  # j beyond i along +axis
    g_ji = pa.min(2) - pb.max(2)  # j beyond i along -axis
    g = np.maximum(g_ij, g_ji)
    k = np.argmax(g, 1)
    best = g[np.arange(len(g)), k]
    if not return_axis:
        return best
    ax = cand[np.arange(len(g)), k]
    flip = g_ji[np.arange(len(g)), k] > g_ij[np.arange(len(g)), k]
    ax[flip] *= -1.0
    return best, ax


def min_gap(P: np.ndarray) -> float:
    pr = neighbor_pairs(P)
    g = pair_gaps(P, pr)
    return float(g.min()) if len(g) else math.inf


def inflate(P: np.ndarray, eps: float) -> np.ndarray:
    """Scale centres about the bounding box centre by (1 + eps).

    For two squares separated (or touching) along unit axis a, the gap grows by
    eps * a.(c_j - c_i) >= eps * (h_i(a) + h_j(a)) >= eps, so every contact opens by at
    least eps while L grows by eps * (L - 1) at most.
    """
    x0, y0, x1, y1 = bbox(P)
    mid = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    Q = P.copy()
    Q[:, :2] = mid + (P[:, :2] - mid) * (1.0 + eps)
    return Q


def grid(n: int) -> np.ndarray:
    k = math.isqrt(n - 1) + 1 if n > 0 else 0
    idx = np.arange(n)
    return np.stack([idx % k + 0.5, idx // k + 0.5, np.zeros(n)], 1)


def score(n: int, L: float) -> float:
    return L / math.sqrt(n) - 1.0


def grid_score(n: int) -> float:
    return math.ceil(math.sqrt(n) - 1e-12) / math.sqrt(n) - 1.0
