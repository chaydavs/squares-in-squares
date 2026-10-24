"""L frame constructions: two leaning stack strips along the walls, exact upright refill.

Container [0, S]^2 with S = K + f. A vertical strip V of width w + f hugs the right wall over
the full height: bars of w + 1 squares tilted by tV with (w + 1) cos tV + sin tV = w + f,
stacked upward with pitch 1 / cos tV. A horizontal strip H of height w2 + f hugs the top wall
between the left wall and V: bars of w2 + 1 squares, stacked rightward. Every row crosses V
and every column left of V crosses H, so the fractional offset f is absorbed everywhere.
The rest (and the strip ends) is filled with upright squares by the exact MILP refill.

Why it wins at large K: the strips have total length about 2K - w (the anti diagonal gadget
chain only gets about K), and each unit of strip length at bloat f gains f - sin(t) squares.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from . import fill, lp
from .geometry import min_gap, side


def tilt_for(w: int, f: float) -> float:
    """Largest tilt t in (0, pi/4) with (w + 1) cos t + sin t = w + f (the tight width fit)."""
    g = lambda t: (w + 1) * math.cos(t) + math.sin(t) - (w + f)
    lo = math.atan(1.0 / (w + 1))  # maximum of the width function
    if g(lo) < 0:
        return math.nan
    return brentq(g, lo, math.pi / 4) if g(math.pi / 4) < 0 else math.nan


def bars_v(x0: float, x1: float, y_lo: float, y_hi: float, w: int, t: float, shift: float = 0.0):
    """Stack of bars of w + 1 squares spanning x in [x0, x1], stacked in y within [y_lo, y_hi]."""
    p = 1.0 / math.cos(t)
    ext_y = (w + 1) * math.sin(t) + math.cos(t)  # vertical extent of one bar
    cx = 0.5 * (x0 + x1)
    first = y_lo + ext_y / 2 + shift
    out = []
    i = 0
    while True:
        cy = first + i * p
        if cy + ext_y / 2 > y_hi + 1e-12:
            break
        for j in range(w + 1):
            o = j - w / 2
            out.append((cx + o * math.cos(t), cy + o * math.sin(t), t))
        i += 1
    return np.array(out).reshape(-1, 3)


def bars_h(y0: float, y1: float, x_lo: float, x_hi: float, w: int, t: float, shift: float = 0.0):
    """Transpose of bars_v: bars spanning y in [y0, y1], stacked in x within [x_lo, x_hi]."""
    B = bars_v(y0, y1, x_lo, x_hi, w, t, shift)
    if len(B) == 0:
        return B
    return np.stack([B[:, 1], B[:, 0], -B[:, 2]], 1)  # reflect across y = x


def build(K: int, f: float, w: int, w2: int, sv: float = 0.0, sh: float = 0.0, tl: float = 30.0):
    """Core of both strips plus exact refill. Returns the packing (unchecked) or None."""
    S = K + f
    tv, th = tilt_for(w, f), tilt_for(w2, f)
    if not (math.isfinite(tv) and math.isfinite(th)):
        return None
    V = bars_v(K - w, S, 0.0, S, w, tv, sv)
    H = bars_h(K - w2, S, 0.0, K - w, w2, th, sh)
    T = np.vstack([V, H])
    if len(T) and min_gap(T) < -1e-9:
        return None
    return fill.refill(T, S, S, time_limit=tl)


def certify(P: np.ndarray, trust: float = 0.05):
    if min_gap(P) < -1e-9:
        return None
    return lp.compact(P, iters=8, trust=trust)
