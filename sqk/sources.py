"""Loaders for public packing sources. Every packing returned is an (n, 3) float array.

* Squares Project (github.com/jlevy/squares, data CC BY 4.0, credit Joshua Levy, the
  squares project): known-best witnesses for n <= 324.
* Francisco Couzo's certificates (github.com/franciscouzo/square-packing): exact rational
  witnesses, x y t with t = tan(theta / 2), for n up to about 10000.

Values are evaluated in mpmath at 50 digits and rounded once to float64.
"""
from __future__ import annotations

import glob
import os
import re
from fractions import Fraction

import mpmath as mp
import numpy as np

mp.mp.dps = 50


def _num(tok: str) -> mp.mpf:
    tok = tok.strip()
    if "/" in tok:
        p, q = tok.split("/")
        return mp.mpf(int(p)) / mp.mpf(int(q))
    return mp.mpf(tok)


def read_couzo_cert(path: str) -> tuple[int, float, np.ndarray]:
    """Parse one certificate: comment lines, then `n side`, then n lines `x y t`."""
    toks: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks.extend(line.split())
    n = int(toks[0])
    side = float(_num(toks[1]))
    body = toks[2:]
    if len(body) != 3 * n:
        raise ValueError(f"{path}: expected {3 * n} values after the header, got {len(body)}")
    P = np.empty((n, 3))
    for k in range(n):
        x, y, t = (_num(v) for v in body[3 * k : 3 * k + 3])
        P[k] = (float(x), float(y), float(2 * mp.atan(t)))
    return n, side, P


def load_couzo(cert_dir: str, nmax: int | None = None) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for path in sorted(glob.glob(os.path.join(cert_dir, "n*.cert"))):
        m = re.fullmatch(r"n(\d+)\.cert", os.path.basename(path))
        if not m:
            continue
        n = int(m.group(1))
        if nmax is not None and n > nmax:
            continue
        try:
            n2, side, P = read_couzo_cert(path)
        except Exception as e:  # keep going; report the bad file
            print(f"skip {path}: {e}")
            continue
        assert n2 == n, (path, n2)
        out[n] = {"P": P, "side_claimed": side, "source": f"couzo:{os.path.basename(path)}"}
    return out


def load_squares_project(witness_json: str) -> dict[int, dict]:
    """Read the json produced by scripts/import_squares_project.py."""
    import json

    K = json.load(open(witness_json))
    out = {}
    for k, d in K.items():
        P = np.array([[float(v) for v in s] for s in d["squares"]])
        out[int(k)] = {
            "P": P,
            "side_claimed": float(d["side_claimed"]),
            "source": f"squares-project:n-{int(k):03d}",
        }
    return out
