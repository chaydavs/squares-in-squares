"""Strip near upright squares from every gadget and refill them exactly (MILP), at the
gadget's own bloat and slightly larger ones; certify gains with the box LP.

    python3 -I scripts/refill_gadgets.py --lib data/gadgets.pkl --out data/gadgets_refill.pkl
"""
from __future__ import annotations

import argparse
import multiprocessing as mp_
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqk import fill  # noqa: E402
from sqk import gadget_search as GS  # noqa: E402
from sqk.gadgets import Gadget, _corner_flags, pareto  # noqa: E402
from sqk.geometry import fold_angle, min_gap  # noqa: E402

DF = (0.0, 0.003, 0.006, 0.01, 0.015, 0.02, 0.03, 0.04)


def work(g: Gadget):
    out = []
    try:
        for tau in (1e-7, 0.05, 0.12):
            near = np.abs(fold_angle(g.P[:, 2])) < tau
            T0 = g.P[~near]
            for df in DF:
                F = g.f + df
                if F >= 0.999:
                    continue
                for anchor in (0, 1):  # core stays at bottom left, or moves with the top right
                    T = T0.copy()
                    if anchor:
                        T[:, :2] += df
                    R = fill.refill(T, g.w + F, g.h + F, time_limit=10)
                    d = len(R) - g.w * g.h
                    if d <= g.d:
                        continue
                    if min_gap(R) < -1e-9:
                        continue
                    c = GS.certify(R, g.w, g.h, trust=0.05, rounds=6)
                    if c is None:
                        continue
                    Rc, Fc = c
                    if Fc < 1 - 1e-9:
                        tl, br = _corner_flags(Rc, g.w, g.h, Fc)
                        out.append(Gadget(g.w, g.h, d, Fc, Rc, g.source + f"+refill(t{tau},df{df},a{anchor})", tl, br))
    except Exception as e:  # keep the sweep going
        print("error", g.key, e, flush=True)
    return g.key, g.f, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="data/gadgets.pkl")
    ap.add_argument("--out", default="data/gadgets_refill.pkl")
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    lib = pickle.load(open(a.lib, "rb"))
    gs = pareto([g for g in lib.best.values() if g.f < 1 - 1e-9])
    gs = [g for g in gs if g.w <= g.h]  # one orientation; transposes are added back by Library.add
    print(f"{len(gs)} gadgets to refill", flush=True)
    t0 = time.time()
    new = 0
    with mp_.Pool(a.procs) as pool:
        for key, f0, out in pool.imap_unordered(work, gs, chunksize=2):
            for g in out:
                if lib.add(g):
                    new += 1
                    print(f"  {key} f={f0:.4f} -> {g.w}x{g.h}+{g.d} f={g.f:.4f}  ({g.source.split('+')[-1]})", flush=True)
    pickle.dump(lib, open(a.out, "wb"))
    print(f"{new} new or better gadgets in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
