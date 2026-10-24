"""Merge exact rational certificates from public repositories into the portfolio.

Formats: `x y t` text certificates (.cert, .cert.gz; Couzo, Evan Daniel) and PingYou's
.cert.json ({"n", "s_exact", "squares": [[x, y, t], ...]}), t = tan(theta / 2).

    python3 -I scripts/harvest_certs.py --in data/portfolio_v4 --out data/portfolio_v5 \
        --roots /root/ext/evand_square-packing /root/ext/BuddenPing_square-packing-certificates
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqk.families import delete_to  # noqa: E402
from sqk.portfolio import Portfolio  # noqa: E402
from sqk.sources import _num  # noqa: E402


def read_text_cert(path):
    op = gzip.open if path.endswith(".gz") else open
    toks = []
    with op(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                toks += line.split()
    n = int(toks[0])
    body = toks[2:]
    if len(body) != 3 * n:
        raise ValueError("bad length")
    P = np.empty((n, 3))
    for k in range(n):
        x, y, t = (_num(v) for v in body[3 * k:3 * k + 3])
        P[k] = (float(x), float(y), float(2 * mp.atan(t)))
    return P


def read_json_cert(path):
    d = json.load(open(path))
    sq = d["squares"]
    P = np.empty((len(sq), 3))
    for k, (x, y, t) in enumerate(sq):
        P[k] = (float(_num(x)), float(_num(y)), float(2 * mp.atan(_num(t))))
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/portfolio_v4")
    ap.add_argument("--out", default="data/portfolio_v5")
    ap.add_argument("--roots", nargs="+", required=True)
    a = ap.parse_args()
    pf = Portfolio.load(a.inp)
    start = pf.total()
    seen = 0
    for root in a.roots:
        for dp, _, fs in os.walk(root):
            if "/.git" in dp:
                continue
            for fn in fs:
                path = os.path.join(dp, fn)
                try:
                    if fn.endswith(".cert") or fn.endswith(".cert.gz"):
                        P = read_text_cert(path)
                    elif fn.endswith(".cert.json"):
                        P = read_json_cert(path)
                    else:
                        continue
                except Exception:
                    continue
                seen += 1
                N = len(P)
                src = "cert:" + os.path.relpath(path, os.path.dirname(root))
                if N <= 1000:
                    if pf.offer(N, P, src):
                        print(f"n={N}: {pf.best[N].side:.10f} from {src}", flush=True)
                # deletion: a certificate with N squares also serves n < N; try the next few
                for n in range(max(1, N - 3), min(N, 1001)):
                    if n < N and pf.offer(n, delete_to(P, n), src + f":del{N - n}"):
                        print(f"n={n}: {pf.best[n].side:.10f} from {src} (delete {N - n})", flush=True)
    pf.save(a.out)
    print(f"{seen} certificates read; total {start:.10f} -> {pf.total():.10f}")


if __name__ == "__main__":
    main()
