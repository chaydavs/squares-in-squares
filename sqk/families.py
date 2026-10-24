"""Closed form constructions and the two monotone transfer rules.

Transfer rules (both valid for any packing):
  * deletion   s(n) <= s(n + k): drop k squares.
  * L border   a packing of N in side s, plus t unit strips along two sides, holds
               N + t (2 floor(s) + t) squares in side s + t.  Each step adds 2 frac(s) of
               waste, so it degrades when frac(s) > 0.5 (see RESEARCH.md).

Families:
  * Goebel squares: b x b block at 45 degrees, staircases in the corners.
        n = 2a^2 + 2a + b^2, s = a + 1 + b / sqrt 2, valid when a - 1 < b / sqrt 2 < a + 1.
  * Goebel strips: one 45 degree strip on the anti diagonal.
        n = (a + 1) a + 2 + b, b = 1 + floor((a - 1) sqrt 2), s = a + 1 + sqrt(2) / 2.
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import grid, normalize, side

R2 = math.sqrt(2.0)


def gobel_square(a: int, b: int) -> np.ndarray:
    S = a + 1 + b / R2
    c = S / 2
    sq = []
    for i in range(b):
        for j in range(b):
            u, v = i - (b - 1) / 2, j - (b - 1) / 2
            sq.append((c + (u - v) / R2, c + (u + v) / R2, math.pi / 4))
    for i in range(a):
        for j in range(a - i):
            x, y = i + 0.5, j + 0.5
            sq += [(x, y, 0.0), (S - x, y, 0.0), (x, S - y, 0.0), (S - x, S - y, 0.0)]
    return np.array(sq)


def gobel_square_params(nmax: int = 1200):
    for a in range(0, 40):
        for b in range(1, 60):
            if a - 1 < b / R2 < a + 1 and 2 * a * a + 2 * a + b * b <= nmax:
                yield a, b


def gobel_strip(a: int) -> np.ndarray:
    """Strip of b diamonds on x + y = s, staircases below and above, two corner squares."""
    s = a + 1 + R2 / 2
    b = 1 + math.isqrt(2 * (a - 1) * (a - 1))  # 1 + floor((a - 1) sqrt 2)
    sq = []
    for i in range(a):
        for j in range(a - i):  # cells with i + j <= a - 1, and their half turn images
            sq.append((i + 0.5, j + 0.5, 0.0))
            sq.append((s - i - 0.5, s - j - 0.5, 0.0))
    sq.append((s - 0.5, 0.5, 0.0))
    sq.append((0.5, s - 0.5, 0.0))
    # diamonds: centres on x + y = s, d = x - y spaced sqrt 2, symmetric about d = 0
    for k in range(b):
        d = (k - (b - 1) / 2) * R2
        sq.append(((s + d) / 2, (s - d) / 2, math.pi / 4))
    return np.array(sq)


def extend(P: np.ndarray, t: int) -> np.ndarray:
    """Add t unit strips along the top and the right of a normalized packing."""
    s = side(P)
    fs = math.floor(s + 1e-9)
    S = s + t
    fS = math.floor(S + 1e-9)
    new = []
    for r in range(t):
        y = s + r + 0.5
        new += [(i + 0.5, y, 0.0) for i in range(fS)]
    for c in range(t):
        x = s + c + 0.5
        new += [(x, j + 0.5, 0.0) for j in range(fs)]
    return np.vstack([P, np.array(new).reshape(-1, 3)])


def extend_count(N: int, s: float, t: int) -> int:
    return N + t * (2 * math.floor(s + 1e-9) + t)


def delete_to(P: np.ndarray, n: int) -> np.ndarray:
    """Keep n squares, dropping those that reach furthest up or right first.

    Deleting never increases L. Dropping extreme squares sometimes decreases it.
    """
    if len(P) == n:
        return P
    P = normalize(P)
    reach = np.maximum(P[:, 0], P[:, 1])
    keep = np.sort(np.argsort(-reach, kind="stable")[len(P) - n :])
    return normalize(P[keep])


__all__ = ["gobel_square", "gobel_square_params", "gobel_strip", "extend", "extend_count", "delete_to", "grid"]
