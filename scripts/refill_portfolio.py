"""Exact upright refill of whole packings: can the packing for n hold more squares at the same
side (or a slightly larger one)? Every extra square improves s(n + k) when that side beats it.

    python3 -I scripts/refill_portfolio.py --in data/portfolio_v4lp --out data/portfolio_v5 --ns 325-1000
"""
from __future__ import annotations

import argparse
import multiprocessing as mp_
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqk import fill, lp  # noqa: E402
from sqk.families import delete_to  # noqa: E402
from sqk.geometry import fold_angle, min_gap, normalize, side  # noqa: E402
from sqk.portfolio import Portfolio  # noqa: E402


def _ns(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def work(args):
    n, P, src, deltas = args
    res = []
    try:
        P = normalize(P)
        L = side(P)
        for tau in (1e-7, 0.12):
            T = P[np.abs(fold_angle(P[:, 2])) >= tau]
            for dl in deltas:
                R = fill.refill(T, L + dl, L + dl, time_limit=60)
                if len(R) > n and min_gap(R) > -1e-9:
                    R = lp.compact(R, iters=6, trust=0.05)
                    res.append((len(R), side(R), R, f"{src}+refill(t{tau},dl{dl})"))
    except Exception as e:
        print("error", n, e, flush=True)
    return n, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/portfolio_v4lp")
    ap.add_argument("--out", default="data/portfolio_v5")
    ap.add_argument("--ns", default="325-1000")
    ap.add_argument("--deltas", default="0,0.003,0.01")
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    pf = Portfolio.load(a.inp)
    start = pf.total()
    deltas = [float(x) for x in a.deltas.split(",")]
    jobs = [(n, pf.best[n].P, pf.best[n].source, deltas) for n in _ns(a.ns) if pf.best[n].source != "grid"]
    print(f"{len(jobs)} packings", flush=True)
    t0 = time.time()
    with mp_.Pool(a.procs) as pool:
        for n, res in pool.imap_unordered(work, jobs):
            for N, s, R, src in res:
                for m in range(n + 1, min(N, 1000) + 1):
                    if s < pf.best[m].side - 1e-9 and pf.offer(m, delete_to(R, m), src + f":n{m}"):
                        print(f"n={m}: {pf.best[m].side:.9f} from refill of n={n} (holds {N})  total {pf.total():.10f}", flush=True)
    pf.save(a.out)
    print(f"total {start:.10f} -> {pf.total():.10f} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
