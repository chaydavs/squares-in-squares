"""Our float geometry must agree with the official metric wherever the metric is decisive.

Run: python3 -I tests/test_geometry_vs_metric.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqk import official_metric as M  # noqa: E402
from sqk.families import gobel_square, gobel_strip  # noqa: E402
from sqk.geometry import corners, inflate, min_gap, normalize, pair_gaps, side  # noqa: E402
from sqk.submission import export_ready  # noqa: E402


def metric_side_and_ok(P):
    sq = [M.Square(x, y, t, x_str=repr(x), y_str=repr(y), angle_str=repr(t)) for x, y, t in map(lambda r: map(float, r), P)]
    ok, why = M.verify_puzzle(sq)
    return float(M.measure_side_mp(sq)), ok


def test_corners_match_metric():
    rng = np.random.default_rng(1)
    P = np.column_stack([rng.uniform(0, 5, 50), rng.uniform(0, 5, 50), rng.uniform(-3, 3, 50)])
    C = corners(P)
    for k in range(50):
        ref = M.compute_corners_float(*P[k])
        assert np.allclose(C[k], np.array(ref), atol=1e-15)


def test_pairs_random():
    """Random pairs near contact: our sign agrees with the metric whenever |gap| > 1e-9."""
    rng = np.random.default_rng(2)
    agree = 0
    for _ in range(3000):
        t1, t2 = rng.uniform(-math.pi, math.pi, 2)
        d = rng.uniform(0.9, 1.5)
        phi = rng.uniform(0, 2 * math.pi)
        P = np.array([[0.0, 0.0, t1], [d * math.cos(phi), d * math.sin(phi), t2]])
        g = pair_gaps(P, np.array([[0, 1]]))[0]
        if abs(g) < 1e-9:
            continue
        a = M.Square(*P[0]); b = M.Square(*P[1])
        res = M.sat_check_float(a, b, undecided_tol=0.0)
        assert (res == "separated") == (g > 0), (P, g, res)
        agree += 1
    assert agree > 2000


def test_exported_families_pass_metric():
    for P in (gobel_square(3, 4), gobel_strip(5), gobel_strip(9)):
        Q = export_ready(P)
        L, ok = metric_side_and_ok(Q)
        assert ok and abs(L - side(Q)) < 1e-12


def test_touching_irrational_needs_gap():
    """An exact 45 degree contact written to float precision is not provably disjoint;
    after export_ready it is."""
    P = gobel_square(1, 1)  # n = 5
    _, ok_raw = metric_side_and_ok(P)
    _, ok_exp = metric_side_and_ok(export_ready(P))
    assert ok_exp
    print(f"  raw n=5 float coordinates accepted by metric: {ok_raw} (exported: {ok_exp})")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
