"""Build the per n portfolio from every public source plus our own constructions.

    python3 -I scripts/build_portfolio.py --couzo PATH/certificates/best --out data/portfolio
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqk import families  # noqa: E402
from sqk.geometry import normalize, side  # noqa: E402
from sqk.portfolio import NMAX, Portfolio, plan_transfers, realize  # noqa: E402
from sqk.sources import load_couzo, load_squares_project  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--couzo", default="/root/ext/couzo/certificates/best")
    ap.add_argument("--couzo-cache", default="data/couzo_certs.pkl")
    ap.add_argument("--squares", default="data/squares_project_known_best.json")
    ap.add_argument("--extra", nargs="*", default=[], help="previous portfolios to merge (path without extension)")
    ap.add_argument("--out", default="data/portfolio")
    a = ap.parse_args()

    t0 = time.time()
    if os.path.exists(a.couzo_cache):
        couzo = pickle.load(open(a.couzo_cache, "rb"))
    else:
        couzo = load_couzo(a.couzo)
        pickle.dump(couzo, open(a.couzo_cache, "wb"))
    sp = load_squares_project(a.squares)
    print(f"sources: couzo {len(couzo)} certs, squares project {len(sp)} witnesses ({time.time() - t0:.0f}s)")

    bases: dict[str, object] = {}
    for n, d in couzo.items():
        bases[d["source"]] = normalize(d["P"])
    for n, d in sp.items():
        bases[d["source"]] = normalize(d["P"])
    for a_, b_ in families.gobel_square_params(NMAX + 200):
        bases[f"gobel-square:a={a_},b={b_}"] = normalize(families.gobel_square(a_, b_))
    for a_ in range(2, 32):
        bases[f"gobel-strip:a={a_}"] = normalize(families.gobel_strip(a_))
    for extra in a.extra:
        old = Portfolio.load(extra)
        for n, e in old.best.items():
            if e.source != "grid":
                bases[f"{e.source}"] = e.P

    plan = plan_transfers(bases, NMAX)
    pf = Portfolio(NMAX)
    for n in range(1, NMAX + 1):
        s_plan, recipe = plan[n]
        if recipe[0] == "grid":
            continue
        P = realize(n, recipe, bases)
        src = recipe[1] if recipe[0] == "base" else f"{recipe[1]}+L{recipe[2]}"
        if len(P) != n or abs(side(P) - s_plan) > 1e-7:
            print(f"n={n}: realized side {side(P):.9f} differs from plan {s_plan:.9f} ({src})")
        pf.offer(n, P, src)
    pf.save(a.out)
    lo, hi = pf.total(1, 324), pf.total(325, NMAX)
    print(f"portfolio total {pf.total():.6f}  (n<=324 {lo:.4f}, n>324 {hi:.4f})  in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
