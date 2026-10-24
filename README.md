# sqk: Kaggle Squares in Squares

Code and notes for the Kaggle competition
[Squares in Squares](https://www.kaggle.com/competitions/squares-in-squares): pack n unit
squares (n = 1..1000) into the smallest square; score = sum of L / sqrt(n) − 1.

Read [RESEARCH.md](RESEARCH.md) first. It explains the problem from first principles,
what the record data says, how the record holders work, where the points are, and the plan.

## Setup

    pip install -r requirements.txt
    git clone --depth 1 https://github.com/franciscouzo/square-packing.git ../ext/couzo

Squares Project witnesses for n ≤ 324 are already converted in
`data/squares_project_known_best.json` (CC BY 4.0, Joshua Levy, the squares project).
To regenerate it from a clone of github.com/jlevy/squares:

    python3 -I scripts/import_squares_project.py ../squares/packing/witnesses/known-best data/squares_project_known_best.json

## Build, polish, validate, submit

    python3 -I scripts/build_portfolio.py --couzo ../ext/couzo/certificates/best --out data/portfolio
    python3 -I scripts/polish_portfolio.py --in data/portfolio --out data/portfolio_lp
    python3 -I -c "import sys; sys.path.insert(0,'.'); from sqk.portfolio import Portfolio; from sqk import submission as S; pf=Portfolio.load('data/portfolio_lp'); S.write({n:e.P for n,e in pf.best.items()}, 'data/submission.csv')"
    python3 -I scripts/validate.py data/submission.csv

`validate.py` runs the unchanged official metric twice (exact strings, and a pandas
float64 round trip) and writes a per n report. Upload only files where every puzzle passes
both modes.

## Tests

    python3 -I tests/test_geometry_vs_metric.py

## Credits

Packings come from Francisco Couzo's certificates, the Squares Project, David Ellsworth's
catalogue after Erich Friedman, and the original discoverers credited there. The official
metric (`sqk/official_metric.py`) is Kaggle's, Apache 2.0.
