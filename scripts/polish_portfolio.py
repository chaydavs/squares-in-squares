"""LP compaction (and optional dual guided rotation) over every non grid entry.

    python3 -I scripts/polish_portfolio.py --in data/portfolio --out data/portfolio_lp --ns 101-1000 --procs 2
"""
from __future__ import annotations

import argparse
import multiprocessing as mp_
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqk import lp  # noqa: E402
from sqk.geometry import min_gap, side  # noqa: E402
from sqk.portfolio import Portfolio  # noqa: E402


def _parse_ns(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def _work(args):
    n, P, rot = args
    t = time.time()
    try:
        R = lp.compact(P, trust=0.1)
        if rot:
            R = lp.rotate_polish(R, rounds=rot, step0=1e-4, trust=0.05)
        ok = min_gap(R) > -1e-9
        return n, R if ok else None, time.time() - t
    except Exception as e:  # one bad case must not stop the sweep
        print(f"n={n}: {e}")
        return n, None, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/portfolio")
    ap.add_argument("--out", default="data/portfolio_lp")
    ap.add_argument("--ns", default="2-1000")
    ap.add_argument("--rot", type=int, default=0, help="rotation rounds after the LP (0 = LP only)")
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    pf = Portfolio.load(a.inp)
    jobs = [(n, pf.best[n].P, a.rot) for n in _parse_ns(a.ns) if pf.best[n].source != "grid"]
    print(f"{len(jobs)} packings to polish")
    gained = 0.0
    with mp_.Pool(a.procs) as pool:
        for n, R, dt in pool.imap_unordered(_work, jobs):
            if R is None:
                continue
            old = pf.best[n].side
            src = pf.best[n].source
            if pf.offer(n, R, src if src.endswith("+lp") else src + "+lp"):
                d = (old - pf.best[n].side) / n**0.5
                gained += d
                print(f"n={n}: {old:.10f} -> {pf.best[n].side:.10f}  score -{d:.2e}  ({dt:.0f}s)", flush=True)
    pf.save(a.out)
    print(f"total {pf.total():.10f}  gained {gained:.3e}")


if __name__ == "__main__":
    main()
