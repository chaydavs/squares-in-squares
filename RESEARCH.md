# Squares in Squares: research notes and strategy

Started 23 October 2026, for the Kaggle competition
[Squares in Squares](https://www.kaggle.com/competitions/squares-in-squares) (final deadline
18 July 2027; milestone prizes at the 17 January and 18 April 2027 cutoffs).

These notes record what we understood before writing any search code: the mathematics,
what the record data says, how the people who hold the records work, and where the points
in this competition actually are. Everything quantitative here was computed from public
data in this repository (`data/`) and can be rerun.

## 1. The problem and the metric

Pack n unit squares (n = 1..1000), each free to translate and rotate, with disjoint
interiors, into the smallest axis aligned square. For every n we submit a centre and an
angle per square. Kaggle measures

    L(n) = max(span of all corner x, span of all corner y),   score(n) = L / sqrt(n) − 1

and the leaderboard is the sum. Lower is better. Three facts about the metric matter more
than anything else in it.

**The score is half the relative wasted area.** Write the waste as W = L² − n. Then
score = sqrt(1 + W / n) − 1 ≈ W / (2n). Every idea below is a way to waste less area.

**A bad puzzle costs only itself.** Wrong row count, overlap, or an undecidable contact
scores that n as the trivial grid, ceil(sqrt n) / sqrt n − 1. The whole file fails only on
missing columns or an n outside 1..1000. The grid for every n sums to 28.93767.

**Validity is certified, not tolerated.** Pairs within 1e−8 go to mpmath interval
arithmetic at 50 to 500 digits, and the pair passes only if the overlap is proved ≤ 0.
An exact contact between irrational numbers cannot be proved and is rejected. So every
contact must be opened by a deliberate gap. We scale centres apart by 1 + 1e−12 (each
contact opens by at least 1e−12, the score moves by about 1e−9 in total) and write float64
reprs, so the file scores identically whether Kaggle passes our strings or a pandas
float64 round trip. `scripts/validate.py` runs the unchanged official metric in both modes.

Format: header `id,n,square_id,x,y,angle`, one row per square, angle in radians.
Ten submissions a day, one final selection, teams of up to five, external public data
allowed.

## 2. First principles

### 2.1 Bounds

* Area: s(n) ≥ sqrt(n). Grid: s(n) ≤ ceil(sqrt(n)). Equality for every perfect square.
* Monotone: s(n) ≤ s(n + 1), since deleting a square never hurts. So a good packing of N
  squares serves every n ≤ N.
* L border: a packing of N squares in side s, plus t unit strips along two sides, holds
  N + t(2 floor(s) + t) squares in side s + t. Each step adds 2 frac(s) of waste.
* Proved exact values: perfect squares; k² − 1 (Karakuş 2026); k² − 2 (Lean proof);
  k² − 3 for every k ≥ 6 (Daniel 2026); k² − 4 for k ≥ 5 (reported); a handful of small
  cases (5, 10, 11, 13, 22, 33, 46, ...). Everything near the top of a block is the grid,
  and that is final for everybody.

### 2.2 Blocks, and the only quantity that matters

Write n = m² + j with 1 ≤ j ≤ 2m, and the side as s = m + δ. The grid wastes
2m + 1 − j. A tilted packing wastes some W. Tilting wins exactly when W < 2m + 1 − j.

Across every record in the public catalogues, **W grows roughly in proportion to m,
almost independently of j**:

| block m | 9 | 12 | 15 | 17 | 20 | 24 | 27 | 31 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best W / m in the block (public) | 0.99 | 1.01 | 0.97 | 0.94 | 0.92 | 0.87 | 0.85 | 0.83 |
| median W / m in the block (public) | 1.05 | 1.05 | 1.04 | 1.03 | 1.03 | 1.02 | 1.01 | 1.00 |

So a tilted packing beats the grid while j is below about (2 − W/m) m, roughly the lower
half of every block (Chris Deotte's "u < 0.55"). The number of puzzles that can be improved
is about m + 2 per block, and those are the only puzzles where anyone can gain points.

### 2.3 Why tilting helps: slack absorption

An upright grid in a container of side m + δ wastes δ along every row and every column,
2δm in total. Every good construction is a way to absorb that slack.

* **Göbel's 45° trick** (n = 5 and its families). A diamond uses √2 of width where an
  upright square uses 1, so a diagonal strip or block of diamonds can take up a fractional
  offset of √2/2. Göbel squares: n = 2a² + 2a + b², s = a + 1 + b/√2. Göbel strips:
  n = (a + 1)a + 2 + b, b = 1 + floor((a − 1)√2), s = a + 1 + √2/2. Best when δ is near
  0.5 or 0.7.
* **Leaning stacks** (Erdős and Graham 1975; Chung and Graham 2009, 2020). In a strip of
  width M + δ, a bar of M + 1 squares tilted by θ fits when (M + 1)cos θ + sin θ ≤ M + δ.
  Bars lean against each other like books on a shelf, so every bar holds one square more
  than an upright column. The price is a thin triangle at each bar end, about sin θ of
  area per unit length of strip, plus a large triangle of about (M + 1)² tan θ / 2 at
  each strip end, part of which upright squares can refill. The tilt can never drop below
  about 2/(M + 1) (the sin θ term), so wider strips waste less per unit length but more at
  their ends.
* Asymptotically this gives W(s) = O(s^0.636) (Erdős and Graham), O(s^0.6306 log s)
  (Chung and Graham 2009) and O(s^0.6) (Chung and Graham 2020); Roth and Vaughan proved
  W(s) ≥ c sqrt(s ‖s‖), so waste can never be bounded. At s ≤ 32 the end effects dominate
  and the records sit at W ≈ 0.83m to 1.05m.

The record data agrees with this picture. Waste depends on δ the way the band formula says
it should (blocks 18 to 31, mean W/m over non grid n):

| δ | 0.50 to 0.60 | 0.60 to 0.70 | 0.70 to 0.80 | 0.80 to 0.90 | 0.90 to 0.95 | 0.95 to 1.00 |
| --- | --- | --- | --- | --- | --- | --- |
| public frontier | 1.007 | 1.044 | 1.051 | 1.010 | 0.953 | 0.908 |
| Kaggle leaders | 1.003 | 1.034 | 1.012 | 0.964 | 0.919 | 0.889 |

Near δ = 1 a shallow tilt suffices and waste is lowest. Around δ = 0.6 to 0.8 neither the
45° families nor shallow stacks fit naturally, and waste peaks. That middle band is where
the leaders found most of their gains and where we expect most of ours.

### 2.4 What record packings look like

* Low j (δ ≈ 0.5 to 0.7): 45° Göbel blocks and strips (n = 290, 295, 297, 734, 744).
* m = 17, high j: two bands about five squares wide at about 20°, in a V or L across the
  container (n = 304 to 308), found by simulated annealing.
* m ≥ 24, high j: an **L frame** of two large patches of leaning stacks, about 6 to 8
  squares wide and tilted 5° to 10°, one vertical and one horizontal, with near upright
  squares filling the patch ends and plain grid everywhere else (n = 604, 761, 935, 754).
  This is Chung and Graham's construction at finite size, and Ryu's 2026 L frame for
  k² − c is the same idea.

## 3. How the people at the top of the field work

* **Construct, then refine.** Most catalogue entries above n ≈ 100 extend or delete from a
  smaller record, or instantiate a family. Search only discovers the primitive pattern.
* **The record engine is GPU simulated annealing** (Thomas Schadt's program, run in
  modified form by David Ellsworth; closed source). Record basins are needles: at n = 51,
  4 of 3004 classified basins refined to the record, about 5 GPU hours per hit, and runs
  at n ≈ 300 took weeks. Seeds are rarely cold: neighbouring records with squares removed
  and others straightened, analytic constructions, hand edits.
* **Plain annealing fails on this problem.** Single square moves against a max span
  objective get stuck just above the trivial grid (Schadt's own note, and the Squares
  Project's measurements). Only the two to four squares on the bounding box feel the
  objective. Systems that work change the move set rather than the cooling schedule:
  inflation or wall pressure formulations, contact group moves, "contract, act, re
  expand", fixed angle LP over all centres, tying angles into a few classes and then
  releasing them, swaps, and remove and straighten seeding from neighbours.
* **Fixed angle LP is exact for translations.** Choose one separating axis per nearby
  pair; non overlap and containment become linear; minimizing L is an LP. Its duals price
  every contact and give the gradient of L with respect to each angle. The objective has
  kinks where the active contact set changes, so angle descent stalls; tie angles into
  classes or move contact groups together.
* **Exactification** (contact equations, Ellsworth's Jacobian determinant condition,
  integer relation recovery) produces the closed forms in the catalogue. Kaggle does not
  need exact forms, only provable gaps.
* **General global solvers do not scale here.** SCIP and Xpress match records only up to
  n ≈ 16.
* **Kaggle Santa 2025** (the closest previous competition): winners combined lattice or
  crystal seeds, GPU batched L-BFGS on an overlap penalty, simulated annealing with
  remove, loosen, reinsert moves, building N from the N ± 1 solutions, and a final
  legalize step. About 1000 GPU hours for first place.

## 4. The competition on 24 October 2026

| reference | total |
| --- | --- |
| trivial grid everywhere | 28.93767 |
| our first construction baseline (catalogue n ≤ 324, Göbel families, L borders) | 23.17696 |
| public frontier (Couzo's certificates and the Squares Project), rebuilt here | 22.35077 |
| **our submission v1**: public frontier plus our LP sweep, official metric both modes | **22.3499485521** |
| best public notebooks | about 22.36 |
| leaderboard first place (Umong Sain) | 22.22476 |
| best known per puzzle across all teams (union) | 22.19429 |

The 0.156 between the public frontier and the per puzzle best lies entirely above
n = 324, and it grows with m (0.004 in block 18, 0.017 in block 31). For n ≤ 324 everyone
already holds the catalogue. 453 puzzles are still the grid for every team.

Public notebooks reach 22.36 by merging every public witness, composing cropped
rectangular "gadgets" from up to 18,232 donor packings along staircase fault lines with a
dynamic program, and polishing with the fixed angle LP and dual guided rotations of
contact groups.

### 4.1 What we measured ourselves

* **The fixed angle LP is exact and fast.** At the record's own angles it reproduces
  Trump's s(11), Bidwell's s(17), s(29), s(55) and s(303) to within the 1e−10 safety
  margin per contact, in 0.01 to 0.1 s; it pulls a perturbed n = 17 packing back into
  Bidwell's basin.
* **The public packings are local optima, with a little slack in places.** One LP sweep
  over all 546 non grid packings found real slack in 32 of them (sides down by up to
  0.0026, for example n = 639, 858, 918, 861), worth 0.00082 in total. Dual guided
  rotation added nothing measurable.
* **Transferring structure without search does not beat the frontier.** Repeating one
  slab in each direction moves a packing to the next block and the LP repairs every
  overlap, but the bands have no room for an extra stack at unit spacing. They spread,
  and the side lands 0.01 to 0.06 above the public packing for the same n. This is the
  record holders' experience too: construction gives the seed, search finds the record.
* **The float trap is real.** Göbel's n = 5 written as plain floats is rejected by the
  official metric (it would score as the grid). Exported with our gap it passes.

## 5. Strategy

What we know about the points:

* n ≤ 324 is shared by everyone at the catalogue (12.2024). Beating a catalogue record is
  a world record; possible but rare.
* n > 324 decides the competition. Matching the best in block efficiency across every j
  would be worth roughly 1 point over the public frontier; the leaders have taken about
  0.16 of it so far.

Plan, in order of expected gain per effort:

1. **Submission v1: the validated public frontier**, plus our LP sweep (22.3499485521).
   Puts us around rank 20 to 25 on 24 October. Done in this repository.
2. **Leaning stack L frame generator for blocks 18 to 31.** Parametrize the patch width
   M, tilt, length and end treatment; build exact candidates for every (m, δ); polish
   with the LP and dual guided rotation; target the δ = 0.6 to 0.9 hump first. Also use
   these as seeds for everything below.
3. **Gadget composition**: a dynamic program that tiles a container with cropped pieces
   of good packings (every certificate up to n = 10,000 is a donor), then LP polish.
4. **Search with structured seeds**: simulated annealing or billiards with contact
   group moves, run on Kaggle GPUs (30 hours a week free), seeded from items 2 and 3 and
   from neighbouring n with squares removed and straightened.
5. **Watch the per puzzle leaderboard** (Kaggle staff refresh it daily) to see where we
   trail, and keep a ledger of every accepted improvement with its provenance.

Discipline: never upload a file the official metric has not passed in both modes; keep
the best packing per n forever (monotone portfolio); credit every source.

## 5.1 Status and where to resume (24 October 2026, 00:40 ET)

| version | how | Kaggle score | rank |
| --- | --- | --- | --- |
| v1 | public frontier + LP sweep | 22.3499485520 | 41 |
| v2 | + public notebook outputs (Deotte FTMC and others) + LP | 22.3056860415 | 9 |
| v4 | + gadget recombination + public Kaggle datasets + LP | 22.3043827908 | 9 |

Leader 22.2248. The whole gap is above n = 324 (`data/targets.json` ranks puzzles by score gap).

**Key finding: we are one square short.** At the leaders' bloat, our gadget DP reaches
their count on 61 of 397 puzzles and is short by exactly one square on 267 (two on 60).
Angran Zhang's bests repeat the same fractional side across blocks (0.79613 at n = 747,
803, 861; 0.79903 at 498, 544, 592), so he composes a fixed set of strip gadgets that hold
one more square than ours at the same bloat. The difference sits in the strip end caps.

Tried, with results:
* Squeeze and relax of whole packings, gadget basin hopping: no gains (jammed optima).
* Exact upright refill (`sqk/fill.py`, MILP): recovers every record's uprights exactly, so
  no free square at the same side.
* Strip stretching (`sqk/stretch.py`): reproduces existing gadgets; each bar adds about
  0.017 to 0.035 bloat at fixed width (pitch 1/cos θ > 1).
* L frame from scratch (`sqk/lframe.py`): strips along two walls give about twice the
  strip length of a diagonal chain, but upright-only end caps waste too much (795 of 803
  at K = 28). End caps need squares at intermediate tilts.

Resume with `sqk/endcap.py` (freeze a strip's periodic middle, insert one square into an
end region, relax that region only, certify with the box LP), then rerun
`scripts/gadget_compose.py` and submit with `scripts/kaggle_submit.py`. Re-harvest public
sources first (`scripts/harvest_notebooks.py`, `scripts/harvest_datasets.py`); they moved
us from rank 41 to 9 in one night.

## 6. Code map

| path | what it does |
| --- | --- |
| `sqk/geometry.py` | corners, side, neighbour pairs, separating axis gaps, inflation |
| `sqk/sources.py` | loaders for Couzo certificates and Squares Project witnesses |
| `sqk/families.py` | grid, Göbel squares and strips, L border, deletion |
| `sqk/portfolio.py` | best packing per n with provenance; transfer planning |
| `sqk/lp.py` | fixed angle LP compaction and dual guided rotation |
| `sqk/grow.py` | slab insertion transfer to the next block, with LP repair (experimental) |
| `sqk/submission.py` | Kaggle CSV writer and reader |
| `sqk/official_metric.py` | the competition metric v17, unchanged (Apache 2.0) |
| `scripts/build_portfolio.py` | merge all sources into `data/portfolio` |
| `scripts/polish_portfolio.py` | LP sweep over the portfolio |
| `scripts/validate.py` | official metric on a CSV, strings and pandas round trip |

## Sources

* Kaggle: competition overview, rules, metric source v17, discussion 751561 (Deotte),
  per puzzle leaderboard notebook (inversion), public notebooks by lightningv08, kirilltym,
  cdeotte; Santa 2025 write ups 672465, 672750, 671181.
* [The Squares Project](https://github.com/jlevy/squares) (Joshua Levy; data CC BY 4.0):
  witnesses for n ≤ 324, tutorials, research notes on algorithms and annealing.
* [Francisco Couzo, square-packing](https://github.com/franciscouzo/square-packing):
  exact certificates and bounds up to n = 10,000.
* David Ellsworth's Squares in Squares catalogue (after Erich Friedman), including the
  Göbel strips, Göbel squares, rigid and s(n² − n − 1) pages.
* E. Friedman, Packing unit squares in squares: a survey and new results (EJC DS7).
* P. Erdős and R. Graham (1975); K. Roth and R. Vaughan (1978); F. Chung and R. Graham
  (2009, 2020); S. Ryu (2026 preprints on k² − c).
* T. Gensane and P. Ryckelynck (2005), inflation and stochastic billiards.
