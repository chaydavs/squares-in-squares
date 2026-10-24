"""Score a submission file with the unchanged official metric (sqk/official_metric.py).

Two modes are run, because we do not control how Kaggle hands the file to the metric:
  * strings:  the exact text in the file;
  * pandas:   pd.read_csv (float64) then to_csv, which is what score() re-serializes.
A puzzle counts only if it passes both. Writes a per n report next to the file.

    python3 -I scripts/validate.py submission.csv [--ns 1-1000] [--procs 2]
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import multiprocessing as mp_
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from sqk import official_metric as M  # noqa: E402


def _parse_ns(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _work(args):
    n, rows = args
    try:
        return M.evaluate_single_puzzle(n, rows)
    except Exception as e:  # mirror the metric's own worker guard
        return (n, M.infeasible_score(n), None, f"worker error: {e}")


def evaluate(puzzles: dict[int, list[dict]], ns: list[int], procs: int) -> dict[int, tuple]:
    work = [(n, puzzles[n]) for n in ns if n in puzzles]
    with mp_.Pool(procs) as pool:
        res = pool.map(_work, work, chunksize=1)
    out = {r[0]: r for r in res}
    for n in ns:
        if n not in out:
            out[n] = (n, M.infeasible_score(n), None, "missing from submission")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--ns", default="1-1000")
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--skip-pandas", action="store_true")
    a = ap.parse_args()
    ns = _parse_ns(a.ns)
    t0 = time.time()

    text = open(a.csv).read()
    raw = M._parse_submission_csv(text)
    res_str = evaluate(raw, ns, a.procs)
    print(f"strings: {sum(r[1] for r in res_str.values()):.10f}  ({time.time() - t0:.0f}s)")

    if a.skip_pandas:
        res_pd = res_str
    else:
        df = pd.read_csv(io.StringIO(text))
        rt = M._parse_submission_csv(df.drop(columns=["id"]).to_csv(index=False))
        res_pd = evaluate(rt, ns, a.procs)
        print(f"pandas round trip: {sum(r[1] for r in res_pd.values()):.10f}  ({time.time() - t0:.0f}s)")

    report = os.path.splitext(a.csv)[0] + "_validation.csv"
    bad = 0
    total = 0.0
    with open(report, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "score", "L", "status_strings", "status_pandas"])
        for n in ns:
            s1, s2 = res_str[n], res_pd[n]
            ok = s1[3] == "ok" and s2[3] == "ok"
            sc = max(s1[1], s2[1])
            total += sc
            if not ok and s1[3] != "missing from submission":
                bad += 1
                print(f"  n={n}: {s1[3]} | {s2[3]}")
            w.writerow([n, f"{sc:.12f}", s1[2] or "", s1[3], s2[3]])
    print(f"worst case total over n in range: {total:.10f}; puzzles failing a mode: {bad}; report {report}")


if __name__ == "__main__":
    main()
