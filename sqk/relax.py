"""Penalty relaxation: squeeze the container, relax overlaps, legalize with the exact LP.

Energy at a fixed target side L (container [0, L]^2), over centres and angles:
    sum over neighbour pairs  relu(-sep_ij)^2      sep = separating axis gap (< 0 is overlap)
  + sum over corners          relu(-c)^2 + relu(c - L)^2
Minimized with L-BFGS (torch, CPU or GPU). A relaxed state with small residual overlap is
then made exactly valid by the fixed angle LP (sqk.lp), which also minimizes L for those
angles. Accept only if the certified side beats the incumbent.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import lp
from .geometry import min_gap, neighbor_pairs, normalize, side

H = 0.5
DT = torch.float64


def _corners(X, Y, T):
    c, s = torch.cos(T), torch.sin(T)
    dx1, dy1 = H * c - H * s, H * s + H * c
    dx2, dy2 = -H * c - H * s, -H * s + H * c
    cx = torch.stack([X + dx1, X + dx2, X - dx1, X - dx2], -1)
    cy = torch.stack([Y + dy1, Y + dy2, Y - dy1, Y - dy2], -1)
    return cx, cy


def energy(X, Y, T, L, I, J, wall_w=1.0):
    cx, cy = _corners(X, Y, T)
    e_wall = (torch.relu(-cx) ** 2 + torch.relu(cx - L) ** 2 + torch.relu(-cy) ** 2 + torch.relu(cy - L) ** 2).sum()
    if len(I) == 0:
        return wall_w * e_wall
    ti, tj = T[I], T[J]
    ax = torch.stack([torch.cos(ti), -torch.sin(ti), torch.cos(tj), -torch.sin(tj)], 1)  # (m, 4)
    ay = torch.stack([torch.sin(ti), torch.cos(ti), torch.sin(tj), torch.cos(tj)], 1)
    pa = cx[I][:, None, :] * ax[:, :, None] + cy[I][:, None, :] * ay[:, :, None]  # (m, 4 axes, 4 corners)
    pb = cx[J][:, None, :] * ax[:, :, None] + cy[J][:, None, :] * ay[:, :, None]
    g = torch.maximum(pb.min(2).values - pa.max(2).values, pa.min(2).values - pb.max(2).values)
    sep = g.max(1).values
    return (torch.relu(-sep) ** 2).sum() + wall_w * e_wall


def relax(P: np.ndarray, L: float, iters: int = 300, reach: float = 1.9) -> tuple[np.ndarray, float]:
    """Minimize the penalty at side L from P (already scaled into [0, L]). Returns (Q, energy)."""
    pr = neighbor_pairs(P, reach)
    I = torch.tensor(pr[:, 0])
    J = torch.tensor(pr[:, 1])
    X = torch.tensor(P[:, 0], dtype=DT, requires_grad=True)
    Y = torch.tensor(P[:, 1], dtype=DT, requires_grad=True)
    T = torch.tensor(P[:, 2], dtype=DT, requires_grad=True)
    opt = torch.optim.LBFGS([X, Y, T], lr=1.0, max_iter=iters, tolerance_grad=1e-14,
                            tolerance_change=1e-16, history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        e = energy(X, Y, T, L, I, J)
        e.backward()
        return e

    opt.step(closure)
    with torch.no_grad():
        e = float(energy(X, Y, T, L, I, J))
    Q = np.stack([X.detach().numpy(), Y.detach().numpy(), T.detach().numpy()], 1)
    return Q, e


def legalize(Q: np.ndarray, trust: float = 0.08, rounds: int = 6) -> np.ndarray | None:
    """Exact repair and compaction by the fixed angle LP."""
    R = normalize(Q)
    for _ in range(rounds):
        r = lp.solve(R, trust=trust)
        if r.status != 0:
            return None
        R = r.P
        if min_gap(R) > -1e-9:
            break
    if min_gap(R) <= -1e-9:
        return None
    return lp.compact(R, iters=8, trust=trust)


def shrink(P: np.ndarray, rounds: int = 20, delta0: float = 0.02, noise: float = 0.0,
           rng: np.random.Generator | None = None, verbose: bool = False) -> np.ndarray:
    """Repeatedly squeeze by delta, relax, legalize; keep certified improvements."""
    rng = rng or np.random.default_rng(0)
    best = normalize(P)
    Lb = side(best)
    delta = delta0
    for rd in range(rounds):
        L = Lb - delta
        Q = best.copy()
        Q[:, :2] *= L / Lb
        if noise:
            Q[:, :2] += rng.normal(0, noise, (len(Q), 2))
            Q[:, 2] += rng.normal(0, noise, len(Q))
        Q, e = relax(Q, L)
        R = legalize(Q) if e < 1e-2 else None
        Lr = side(R) if R is not None else math.inf
        ok = Lr < Lb - 1e-10
        if verbose:
            print(f"  shrink {rd}: target {L:.6f} energy {e:.2e} certified {Lr:.9f} {'ACCEPT' if ok else ''}", flush=True)
        if ok:
            best, Lb = R, Lr
            delta *= 1.5
        else:
            delta *= 0.5
            if delta < 1e-5:
                break
    return best
