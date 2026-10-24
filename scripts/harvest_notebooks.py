"""Merge packings from public Kaggle notebook outputs into the portfolio.

For each public notebook of the competition: fetch its output archive, read every CSV with
the submission header, and offer each puzzle to the portfolio. A puzzle is accepted only if
it is strictly smaller and passes the float separating axis check; the official metric then
validates the final file. Downloads are treated as data and deleted after use. Every
accepted packing keeps its notebook as the source, for credit.

    python3 -I scripts/harvest_notebooks.py --in data/portfolio_lp --out data/portfolio_v2
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from kaggle_mcp import TOKEN, Client  # noqa: E402
from sqk.portfolio import Portfolio  # noqa: E402

COMP = "squares-in-squares"
NEED = {"n", "square_id", "x", "y", "angle"}


def call(c, tool, req):
    r = c.rpc("tools/call", {"name": tool, "arguments": {"request": req}})
    t = r["result"]["content"][0]["text"]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return t


def list_notebooks(c):
    out = []
    for page in range(1, 10):
        j = call(c, "search_notebooks", {"competition": COMP, "hasCompetition": True, "group": "Everyone",
                                         "sortBy": "DateRun", "pageSize": 100, "hasPageSize": True, "page": page})
        ks = j if isinstance(j, list) else next((v for v in j.values() if isinstance(v, list)), []) if isinstance(j, dict) else []
        if not ks:
            break
        out += [k["ref"] for k in ks]
    return list(dict.fromkeys(out))


def fetch_zip(c, ref, dest):
    owner, slug = ref.split("/", 1)
    j = call(c, "download_notebook_output", {"ownerSlug": owner, "kernelSlug": slug})
    if not isinstance(j, dict) or "url" not in j:
        return False
    req = urllib.request.Request(j["url"])
    req.add_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(req, timeout=900) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    return True


ALIASES = {"square_id": ("square_id", "sid", "i", "idx", "k"), "angle": ("angle", "theta", "rot", "a")}


def packings_from_csv(fh) -> dict[int, np.ndarray]:
    rd = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"))
    if not rd.fieldnames:
        return {}
    cols = set(rd.fieldnames)
    sid = next((a for a in ALIASES["square_id"] if a in cols), None)
    ang = next((a for a in ALIASES["angle"] if a in cols), None)
    if not ({"n", "x", "y"} <= cols and sid and ang):
        return {}
    rows: dict[int, list] = {}
    for r in rd:
        try:
            rows.setdefault(int(r["n"]), []).append((int(r[sid]), float(r["x"]), float(r["y"]), float(r[ang])))
        except (ValueError, TypeError):
            continue
    out = {}
    for n, rs in rows.items():
        if 1 <= n <= 1000 and len(rs) == n and len({r[0] for r in rs}) == n:
            rs.sort()
            out[n] = np.array([[x, y, t] for _, x, y, t in rs])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/portfolio_lp")
    ap.add_argument("--out", default="data/portfolio_v2")
    ap.add_argument("--log", default="data/harvest_log.json")
    a = ap.parse_args()
    pf = Portfolio.load(a.inp)
    start = pf.total()
    c = Client()
    c.start()
    refs = list_notebooks(c)
    print(f"{len(refs)} public notebooks", flush=True)
    log = {}
    for ref in refs:
        tmp = tempfile.mkdtemp(prefix="nb_")
        try:
            z = os.path.join(tmp, "out.zip")
            if not fetch_zip(c, ref, z):
                log[ref] = "no output"
                continue
            won = []
            with zipfile.ZipFile(z) as zf:
                for name in zf.namelist():
                    if not name.lower().endswith(".csv") or zf.getinfo(name).file_size > 200_000_000:
                        continue
                    with zf.open(name) as fh:
                        packs = packings_from_csv(fh)
                    for n, P in packs.items():
                        if pf.offer(n, P, f"kaggle-nb:{ref}:{os.path.basename(name)}"):
                            won.append(n)
            log[ref] = sorted(set(won))
            print(f"{ref}: {len(set(won))} puzzles improved; total now {pf.total():.10f}", flush=True)
        except Exception as e:  # one bad notebook must not stop the harvest
            log[ref] = f"error: {e}"
            print(f"{ref}: error {e}", flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    pf.save(a.out)
    json.dump(log, open(a.log, "w"), indent=1)
    print(f"total {start:.10f} -> {pf.total():.10f}")


if __name__ == "__main__":
    main()
