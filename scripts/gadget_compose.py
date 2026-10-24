"""Build a gadget library from every packing we hold, compose with the DP, offer results.

    python3 -I scripts/gadget_compose.py --in data/portfolio_v2lp --out data/portfolio_v3 \
        --couzo-cache /root/sqk/data/couzo_certs.pkl --lib data/gadgets.pkl

For every K and every target j = n - K^2, binary search the smallest bloat threshold at
which the DP reaches sum d >= j, realize that chain, LP compact it, check it, and offer it.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp_
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqk import gadgets as G  # noqa: E402
from sqk import lp  # noqa: E402
from sqk.geometry import min_gap, side  # noqa: E402
from sqk.portfolio import Portfolio  # noqa: E402


def _extract(args):
    P, src = args
    try:
        return G.extract(P, src)
    except Exception as e:  # keep going on odd packings
        return []


def build_library(packs: list[tuple[np.ndarray, str]], procs: int) -> G.Library:
    lib = G.Library()
    with mp_.Pool(procs) as pool:
        for gs in pool.imap_unordered(_extract, packs, chunksize=4):
            for g in gs:
                lib.add(g)
    return lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/portfolio_v2lp")
    ap.add_argument("--out", default="data/portfolio_v3")
    ap.add_argument("--couzo-cache", default="/root/sqk/data/couzo_certs.pkl")
    ap.add_argument("--lib", default="data/gadgets.pkl")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--kmin", type=int, default=10)
    ap.add_argument("--kmax", type=int, default=31)
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    t0 = time.time()
    pf = Portfolio.load(a.inp)
    if os.path.exists(a.lib) and not a.rebuild:
        lib = pickle.load(open(a.lib, "rb"))
    else:
        packs = [(e.P, e.source) for e in pf.best.values() if e.source != "grid"]
        if os.path.exists(a.couzo_cache):
            for n, d in pickle.load(open(a.couzo_cache, "rb")).items():
                packs.append((d["P"], d["source"]))
        print(f"extracting gadgets from {len(packs)} packings", flush=True)
        lib = build_library(packs, a.procs)
        pickle.dump(lib, open(a.lib, "wb"))
    gs = G.pareto([g for g in lib.best.values() if g.f < 1 - 1e-9])
    print(f"library {len(lib)} gadgets, {len(gs)} on the Pareto front ({time.time() - t0:.0f}s)", flush=True)
    fs = sorted({g.f for g in gs})
    start = pf.total()
    for K in range(a.kmin, a.kmax + 1):
        # D(K, F) is monotone in F: binary search the threshold for each target j
        cache = {}

        def D(i):
            if i not in cache:
                cache[i] = G.compose_fast(gs, K, fs[i])
            return cache[i]

        top = D(len(fs) - 1)[0]
        wins = 0
        for j in range(1, min(top, 2 * K) + 1):
            n = K * K + j
            if n > 1000:
                break
            lo, hi = 0, len(fs) - 1
            if D(hi)[0] < j:
                continue
            while lo < hi:
                mid = (lo + hi) // 2
                if D(mid)[0] >= j:
                    hi = mid
                else:
                    lo = mid + 1
            Dv, chain = D(lo)
            f = max(g.f for g, _ in chain)
            if K + f >= pf.best[n].side - 1e-9:
                continue
            P = G.realize(chain, K)
            if len(P) < n or min_gap(P) < -1e-9:
                print(f"  K={K} j={j}: realized chain invalid (len {len(P)}, gap {min_gap(P):.2e})", flush=True)
                continue
            P = lp.compact(P, iters=6, trust=0.05)
            from sqk.families import delete_to
            P = delete_to(P, n)
            src = "gadgets:" + "+".join(f"{g.w}x{g.h}+{g.d}" for g, _ in chain)
            if pf.offer(n, P, src):
                wins += 1
        print(f"K={K}: max sum d {top}, {wins} puzzles improved, total {pf.total():.10f} ({time.time() - t0:.0f}s)", flush=True)
    pf.save(a.out)
    print(f"total {start:.10f} -> {pf.total():.10f}")


if __name__ == "__main__":
    main()
