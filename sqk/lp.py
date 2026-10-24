"""Fixed angle LP compaction, and dual guided angle steps.

With every angle fixed and one separating axis chosen per neighbouring pair, both
non-overlap and containment are linear in the centres:

    pair (i, j), unit axis a (oriented i -> j):   a.(c_j - c_i) >= h_i(a) + h_j(a) + margin
    walls:  x_i - e_i >= 0,  x_i + e_i <= L,  y_i - e_i >= 0,  y_i + e_i <= L

with h_k(a) = (|a.u_k| + |a.v_k|) / 2 the support of square k along a and e_k its
half extent. Minimizing L is an LP (HiGHS). Any feasible point is a valid packing for the
listed pairs; a box trust region on the centres keeps unlisted pairs apart.

The LP duals price each contact. For an angle theta_p, dL*/dtheta_p = -sum_k mu_k dr_k/dtheta_p
where r_k = A_k z* - b_k is constraint k's residual and mu_k <= 0 is HiGHS's marginal.
Axes stay attached to the square that owns them, so they rotate with it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .geometry import REACH, axes_of, half_extent, min_gap, neighbor_pairs, normalize, pair_gaps, side

H = 0.5


@dataclass
class PairAxes:
    pairs: np.ndarray  # (m, 2)
    owner: np.ndarray  # (m,) 0 if the axis belongs to square i, 1 if to square j
    k: np.ndarray  # (m,) which edge normal of the owner (0: u, 1: v)
    sign: np.ndarray  # (m,) +1 or -1 so that the axis points from i to j


def choose_axes(P: np.ndarray, reach: float) -> PairAxes:
    pr = neighbor_pairs(P, reach)
    i, j = pr[:, 0], pr[:, 1]
    C = None
    from .geometry import corners

    C = corners(P)
    A, B = C[i], C[j]
    cand = np.concatenate([axes_of(P[i, 2]), axes_of(P[j, 2])], 1)  # (m, 4, 2)
    pa = np.einsum("mkd,mad->mak", A, cand)
    pb = np.einsum("mkd,mad->mak", B, cand)
    g_ij = pb.min(2) - pa.max(2)
    g_ji = pa.min(2) - pb.max(2)
    g = np.maximum(g_ij, g_ji)
    best = np.argmax(g, 1)
    rows = np.arange(len(pr))
    sign = np.where(g_ji[rows, best] > g_ij[rows, best], -1.0, 1.0)
    return PairAxes(pr, (best >= 2).astype(np.int64), best % 2, sign)


def _axis(theta_owner: np.ndarray, k: np.ndarray, sign: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta_owner), np.sin(theta_owner)
    ax = np.where(k[:, None] == 0, np.stack([c, s], 1), np.stack([-s, c], 1))
    return ax * sign[:, None]


def _support(a: np.ndarray, theta: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return H * (np.abs(a[:, 0] * c + a[:, 1] * s) + np.abs(-a[:, 0] * s + a[:, 1] * c))


def pair_rows(theta: np.ndarray, pa: PairAxes) -> tuple[np.ndarray, np.ndarray]:
    """Axis per pair and right hand side: a.c_i - a.c_j <= -(h_i + h_j)."""
    i, j = pa.pairs[:, 0], pa.pairs[:, 1]
    own = np.where(pa.owner == 0, theta[i], theta[j])
    a = _axis(own, pa.k, pa.sign)
    rhs = -(_support(a, theta[i]) + _support(a, theta[j]))
    return a, rhs


@dataclass
class LPResult:
    P: np.ndarray
    L: float
    status: int
    grad: np.ndarray | None = None


def solve(P: np.ndarray, trust: float = 0.1, margin: float = 1e-10, pa: PairAxes | None = None,
          want_grad: bool = False, tol: float = 1e-10) -> LPResult:
    """One LP: minimize L over centres, angles fixed. Returns the packing translated to [0, L]^2."""
    P = normalize(P)
    n = len(P)
    th = P[:, 2]
    reach = REACH + 2 * math.sqrt(2) * trust + 1e-6
    if pa is None:
        pa = choose_axes(P, reach)
    m = len(pa.pairs)
    a, rhs = pair_rows(th, pa)
    e = half_extent(th)
    nv = 2 * n + 1  # x (n), y (n), L
    iL = 2 * n

    # pair rows: a.c_i - a.c_j <= rhs - margin
    i, j = pa.pairs[:, 0], pa.pairs[:, 1]
    r = np.repeat(np.arange(m), 4)
    cols = np.stack([i, n + i, j, n + j], 1).ravel()
    vals = np.stack([a[:, 0], a[:, 1], -a[:, 0], -a[:, 1]], 1).ravel()
    Ap = sp.csr_matrix((vals, (r, cols)), shape=(m, nv))
    bp = rhs - margin

    # walls: -x_i <= -e_i ; x_i - L <= -e_i ; same for y
    idx = np.arange(n)
    rows = np.concatenate([idx, n + idx, 2 * n + idx, 2 * n + idx, 3 * n + idx, 3 * n + idx])
    cols = np.concatenate([idx, idx, iL + 0 * idx, n + idx, n + idx, iL + 0 * idx])
    # rows 0..n-1: -x ; n..2n-1: x - L ; 2n..3n-1: -y ; 3n..4n-1: y - L
    rows = np.concatenate([idx, n + idx, n + idx, 2 * n + idx, 3 * n + idx, 3 * n + idx])
    cols = np.concatenate([idx, idx, np.full(n, iL), n + idx, n + idx, np.full(n, iL)])
    vals = np.concatenate([-np.ones(n), np.ones(n), -np.ones(n), -np.ones(n), np.ones(n), -np.ones(n)])
    Aw = sp.csr_matrix((vals, (rows, cols)), shape=(4 * n, nv))
    bw = np.concatenate([-e, -e, -e, -e])

    A = sp.vstack([Ap, Aw]).tocsr()
    b = np.concatenate([bp, bw])
    c = np.zeros(nv)
    c[iL] = 1.0
    x0 = np.concatenate([P[:, 0], P[:, 1]])
    bounds = [(x0[k] - trust, x0[k] + trust) for k in range(2 * n)] + [(0, None)]
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs",
                  options={"primal_feasibility_tolerance": tol, "dual_feasibility_tolerance": tol})
    if res.status != 0:
        return LPResult(P, side(P), res.status)
    z = res.x
    Q = np.stack([z[:n], z[n:2 * n], th], 1)
    out = LPResult(normalize(Q), side(Q), 0)
    if want_grad:
        out.grad = _angle_gradient(z, th, pa, e, res.ineqlin.marginals, n, m)
    return out


def _angle_gradient(z, th, pa, e, mu, n, m, h=1e-7):
    """dL*/dtheta by the envelope theorem, residual derivatives by central differences."""
    x, y = z[:n], z[n:2 * n]
    grad = np.zeros(n)
    i, j = pa.pairs[:, 0], pa.pairs[:, 1]
    dc = np.stack([x[i] - x[j], y[i] - y[j]], 1)  # c_i - c_j

    def resid(theta):
        a, rhs = pair_rows(theta, pa)
        return np.einsum("md,md->m", a, dc) - rhs

    mu_p = mu[:m]
    # each pair depends on th_i and th_j: perturb all i angles together is wrong (shared
    # squares), so differentiate per endpoint with the other endpoint held fixed.
    for end in (0, 1):
        idx = pa.pairs[:, end]
        tp = th.copy()
        tm = th.copy()
        # perturb only the endpoint's angle, pair by pair, via a local copy of thetas
        ti_p, ti_m = th[idx] + h, th[idx] - h
        a_p, rhs_p = _pair_rows_local(th, pa, end, ti_p)
        a_m, rhs_m = _pair_rows_local(th, pa, end, ti_m)
        dr = (np.einsum("md,md->m", a_p, dc) - rhs_p - (np.einsum("md,md->m", a_m, dc) - rhs_m)) / (2 * h)
        np.add.at(grad, idx, -mu_p * dr)
    # walls: rows -x+e... residual r = (-x_i + e_i), (x_i - L + e_i), same for y: dr/dth = e'
    de = (half_extent(th + h) - half_extent(th - h)) / (2 * h)
    mw = mu[m:].reshape(4, n)
    grad += -(mw.sum(0)) * de
    return grad


def _pair_rows_local(th, pa, end, t_end):
    """pair_rows with the angle of endpoint `end` of every pair replaced by t_end."""
    i, j = pa.pairs[:, 0], pa.pairs[:, 1]
    ti = t_end if end == 0 else th[i]
    tj = t_end if end == 1 else th[j]
    own = np.where(pa.owner == 0, ti, tj)
    a = _axis(own, pa.k, pa.sign)
    rhs = -(_support(a, ti) + _support(a, tj))
    return a, rhs


def compact(P: np.ndarray, iters: int = 20, trust: float = 0.1, verbose: bool = False) -> np.ndarray:
    """Alternate axis choice and LP until L stops improving."""
    best = normalize(P)
    Lb = side(best)
    for it in range(iters):
        r = solve(best, trust=trust)
        if r.status != 0 or min_gap(r.P) < -1e-9:
            break
        if verbose:
            print(f"  lp {it}: L={r.L:.12f}")
        if r.L < Lb - 1e-12:
            best, Lb = r.P, r.L
        else:
            break
    return best


def rotate_polish(P: np.ndarray, rounds: int = 30, step0: float = 1e-3, trust: float = 0.05,
                  only_tilted: bool = False, verbose: bool = False) -> np.ndarray:
    """Dual guided descent on the angles, re-solving the LP at every trial."""
    best = compact(P, trust=trust)
    Lb = side(best)
    step = step0
    for rd in range(rounds):
        r = solve(best, trust=trust, want_grad=True)
        if r.status != 0 or r.grad is None:
            break
        g = r.grad
        if only_tilted:
            g = np.where(np.abs(np.sin(2 * best[:, 2])) > 1e-9, g, 0.0)
        gn = np.linalg.norm(g)
        if gn < 1e-14:
            break
        improved = False
        for _ in range(8):
            Q = r.P.copy()
            Q[:, 2] = Q[:, 2] - step * g / gn
            q = solve(Q, trust=trust)
            if q.status == 0 and q.L < Lb - 1e-12 and min_gap(q.P) > -1e-9:
                best, Lb = q.P, q.L
                improved = True
                step *= 1.5
                break
            step *= 0.4
        if verbose:
            print(f"  rot {rd}: L={Lb:.12f} step={step:.2e} {'+' if improved else '-'}")
        if not improved and step < 1e-9:
            break
    return best
