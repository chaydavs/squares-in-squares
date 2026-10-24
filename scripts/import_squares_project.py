"""Load the Squares Project best known packings (n <= 324) into (cx, cy, theta) arrays.

Usage: python3 -I load_known.py <witness_dir> <out.npz>
Theta is in radians. Values are evaluated with mpmath at 60 digits, then stored as float64
plus a high precision string copy for output.
"""
import sys, glob, json
from fractions import Fraction
import yaml
import mpmath as mp

mp.mp.dps = 60


def num(s):
    s = str(s)
    if "/" in s:
        p, q = s.split("/")
        return mp.mpf(int(p)) / mp.mpf(int(q))
    return mp.mpf(s)


def make_eval(scalar):
    kind = scalar["kind"]
    if kind == "algebraic-number-field":
        coeffs = [num(c) for c in scalar["minimal_polynomial"]]  # leading first
        lo, hi = (num(v) for v in scalar["isolating_interval"])
        poly = lambda x: mp.polyval(coeffs, x)
        alpha = mp.findroot(poly, (lo, hi), solver="anderson")
        def ev(v):
            if isinstance(v, list):
                return sum(num(c) * alpha ** i for i, c in enumerate(v))
            return num(v)
        return ev
    return num


def load(path):
    w = yaml.safe_load(open(path))["witness"]
    ev = make_eval(w["scalar"])
    rep = w["representation"]
    unit = w["coordinates"].get("angle_unit")
    out = []
    for sq in w["squares"]:
        if rep == "center-angle":
            cx, cy = ev(sq["center"][0]), ev(sq["center"][1])
            a = ev(sq["angle"])
            th = a * mp.pi / 180 if unit == "degrees" else a
        elif rep == "center-basis":
            cx, cy = ev(sq["center"][0]), ev(sq["center"][1])
            bx, by = ev(sq["basis"][0]), ev(sq["basis"][1])
            th = mp.atan2(by, bx)
        elif rep == "corners":
            (x0, y0), (x1, y1), (x2, y2) = [(ev(a), ev(b)) for a, b in sq["corners"][:3]]
            cx, cy = (x0 + x2) / 2, (y0 + y2) / 2
            th = mp.atan2(y1 - y0, x1 - x0)
        else:
            raise ValueError(rep)
        out.append((cx, cy, th))
    side = w["side"]
    side = ev(side) if not isinstance(side, list) else ev(side)
    return w["n"], side, out


if __name__ == "__main__":
    wdir, outp = sys.argv[1], sys.argv[2]
    data = {}
    for f in sorted(glob.glob(f"{wdir}/n-*.yaml")):
        n, side, sqs = load(f)
        assert len(sqs) == n, (f, len(sqs))
        data[n] = {
            "side_claimed": mp.nstr(side, 30),
            "squares": [[mp.nstr(c, 30) for c in s] for s in sqs],
        }
    json.dump(data, open(outp, "w"))
    print("loaded", len(data), "cases")
