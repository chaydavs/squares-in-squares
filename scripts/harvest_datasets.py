"""Merge packings from public Kaggle datasets into the portfolio (same rules as notebooks).

    python3 -I scripts/harvest_datasets.py --in data/portfolio_v3 --out data/portfolio_v4 \
        --datasets kmmmvs/squares-in-squares-best-packings dariushafshar/squares-in-squares-certified-packings
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from harvest_notebooks import call, packings_from_csv  # noqa: E402
from kaggle_mcp import Client  # noqa: E402
from sqk.portfolio import Portfolio  # noqa: E402

DEFAULT = [
    "kmmmvs/squares-in-squares-best-packings",
    "dariushafshar/squares-in-squares-certified-packings",
    "lightningv08/squares-in-squares-layout-atlas",
    "alexchilton/squares-in-squares-starter-kit",
    "prvsiyan/a-square-deal-frozen-public-inputs",
]


def list_files(c, owner, slug):
    out, token = [], None
    for _ in range(20):
        req = {"ownerSlug": owner, "datasetSlug": slug, "pageSize": 200, "hasPageSize": True}
        if token:
            req.update({"pageToken": token, "hasPageToken": True})
        j = call(c, "list_dataset_files", req)
        if not isinstance(j, dict):
            break
        out += next((v for v in j.values() if isinstance(v, list)), [])
        token = j.get("next_page_token") or j.get("nextPageToken")
        if not token:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/portfolio_v3")
    ap.add_argument("--out", default="data/portfolio_v4")
    ap.add_argument("--datasets", nargs="*", default=DEFAULT)
    ap.add_argument("--log", default="data/harvest_datasets_log.json")
    a = ap.parse_args()
    pf = Portfolio.load(a.inp)
    start = pf.total()
    c = Client()
    c.start()
    log = {}
    for ref in a.datasets:
        owner, slug = ref.split("/")
        files = [f for f in list_files(c, owner, slug)
                 if str(f.get("name", "")).lower().endswith(".csv") and int(f.get("total_bytes") or 0) < 200_000_000]
        for f in files:
            name = f["name"]
            tmp = tempfile.mkdtemp(prefix="ds_")
            try:
                j = call(c, "download_dataset", {"ownerSlug": owner, "datasetSlug": slug, "fileName": name})
                if not isinstance(j, dict) or "url" not in j:
                    continue
                path = os.path.join(tmp, "f.csv")
                with urllib.request.urlopen(j["url"], timeout=900) as r, open(path, "wb") as fh:
                    shutil.copyfileobj(r, fh)
                with open(path, "rb") as fh:
                    packs = packings_from_csv(fh)
                won = [n for n, P in packs.items() if pf.offer(n, P, f"kaggle-ds:{ref}:{name}")]
                log[f"{ref}:{name}"] = {"puzzles": len(packs), "improved": sorted(won)}
                if packs:
                    print(f"{ref}:{name}: {len(packs)} puzzles read, {len(won)} improved; total {pf.total():.10f}", flush=True)
            except Exception as e:  # keep going
                log[f"{ref}:{name}"] = f"error: {e}"
                print(f"{ref}:{name}: error {e}", flush=True)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    pf.save(a.out)
    json.dump(log, open(a.log, "w"), indent=1)
    print(f"total {start:.10f} -> {pf.total():.10f}")


if __name__ == "__main__":
    main()
