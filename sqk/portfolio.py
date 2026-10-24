"""Per n best packing, with provenance.

Offers are cheap (count, side, recipe) triples; only the winner for each n is realized
and checked. A packing of N squares in side s serves every n <= N (deletion), and its
L border extensions serve larger n.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np

from . import families
from .geometry import grid, min_gap, normalize, side

NMAX = 1000


@dataclass
class Entry:
    n: int
    side: float
    P: np.ndarray
    source: str


class Portfolio:
    def __init__(self, nmax: int = NMAX):
        self.nmax = nmax
        self.best: dict[int, Entry] = {n: Entry(n, float(math.isqrt(n - 1) + 1), grid(n), "grid") for n in range(1, nmax + 1)}

    # ------------------------------------------------------------------ offers
    def offer(self, n: int, P: np.ndarray, source: str, check: bool = True, tol: float = 1e-9) -> bool:
        """Accept P for n if it is valid (float check) and strictly smaller."""
        if len(P) != n:
            raise ValueError(f"{source}: {len(P)} squares offered for n={n}")
        L = side(P)
        if L >= self.best[n].side - 1e-12:
            return False
        if check:
            g = min_gap(P)
            if g < -tol:
                print(f"reject {source} for n={n}: overlap {g:.3e}")
                return False
        self.best[n] = Entry(n, L, normalize(P), source)
        return True

    def total(self, lo: int = 1, hi: int | None = None) -> float:
        hi = hi or self.nmax
        return sum(self.best[n].side / math.sqrt(n) - 1 for n in range(lo, hi + 1))

    # ------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        arrays = {f"n{n}": e.P for n, e in self.best.items()}
        np.savez_compressed(path + ".npz", **arrays)
        meta = {n: {"side": e.side, "source": e.source} for n, e in self.best.items()}
        json.dump(meta, open(path + ".json", "w"), indent=0)

    @classmethod
    def load(cls, path: str) -> "Portfolio":
        meta = json.load(open(path + ".json"))
        arr = np.load(path + ".npz")
        pf = cls(nmax=max(int(k) for k in meta))
        for k, d in meta.items():
            n = int(k)
            pf.best[n] = Entry(n, d["side"], arr[f"n{n}"], d["source"])
        return pf


# ----------------------------------------------------------------- transfers
def plan_transfers(bases: dict[str, np.ndarray], nmax: int = NMAX) -> dict[int, tuple[float, tuple]]:
    """Best (side, recipe) per n using deletion and L border extension of every base.

    recipe = ("base", key) or ("ext", key, t); deletion is implicit (count >= n).
    """
    best: dict[int, tuple[float, tuple]] = {n: (float(math.isqrt(n - 1) + 1), ("grid",)) for n in range(1, nmax + 1)}

    def take(count: int, s: float, recipe: tuple) -> None:
        # a packing of `count` serves every n <= count; walk down while it helps
        for n in range(min(count, nmax), 0, -1):
            if s < best[n][0] - 1e-12:
                best[n] = (s, recipe)
            elif s > best[n][0] + 1e-9 and n < count - 2000:
                break

    for key, P in bases.items():
        N, s = len(P), side(P)
        take(N, s, ("base", key))
        fs = math.floor(s + 1e-9)
        if s - fs < 1e-9:
            continue
        t = 1
        while N + t * (2 * fs + t) - (2 * (fs + t) + 2) <= nmax:
            take(families.extend_count(N, s, t), s + t, ("ext", key, t))
            t += 1
    return best


def realize(n: int, recipe: tuple, bases: dict[str, np.ndarray]) -> np.ndarray:
    if recipe[0] == "grid":
        return grid(n)
    P = normalize(bases[recipe[1]])
    if recipe[0] == "ext":
        P = families.extend(P, recipe[2])
    return families.delete_to(P, n)
