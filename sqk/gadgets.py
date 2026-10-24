"""Fault line gadgets: extraction from packings, LP tightening, and composition.

Model (after Chris Deotte's multi gadget fault DP, see RESEARCH.md). In a container of side
K + f, upright squares on the integer lattice anchored at the bottom left ("blue") and
upright squares on the lattice shifted by (f, f) anchored at the top right ("green") meet
along a monotone staircase from the top left corner to the bottom right corner. Gadgets sit
on the staircase. A gadget (w, h, d) occupies a w x h block of cells and holds w h + d
squares inside the box [0, w + f] x [0, h + f], whose bottom left touches blue cells and
whose top right touches green cells. A chain of gadgets with sum w <= K and sum h <= K
gives K^2 + sum d squares in side K + max f.

Consecutive gadgets touch corner to corner; their boxes then share an f x f corner square.
That is safe when at least one of the two gadgets leaves its corner of that square empty;
otherwise a buffer column (or row) separates them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import lp
from .geometry import corners, fold_angle, min_gap, side

TOL = 1e-6


@dataclass
class Gadget:
    w: int
    h: int
    d: int
    f: float
    P: np.ndarray  # local frame: box [0, w + f] x [0, h + f]
    source: str = ""
    tl_free: bool = True  # top left f x f corner square of the box is empty
    br_free: bool = True  # bottom right corner square is empty

    @property
    def key(self):
        return (self.w, self.h, self.d)


def _corner_flags(P: np.ndarray, w: int, h: int, f: float) -> tuple[bool, bool]:
    """Does any square reach into the top left [0,f]x[h,h+f] or bottom right [w,w+f]x[0,f] corner?"""
    if len(P) == 0:
        return True, True
    C = corners(P)
    xs, ys = C[..., 0], C[..., 1]
    # conservative: a square's bounding box intersecting the corner square counts as occupying it
    tl = ~((xs.min(1) >= f - 1e-12) | (ys.max(1) <= h + 1e-12))
    br = ~((xs.max(1) <= w + 1e-12) | (ys.min(1) >= f - 1e-12))
    return not tl.any(), not br.any()


def tighten(P: np.ndarray, w: int, h: int, f0: float, trust: float = 0.1, iters: int = 8) -> tuple[np.ndarray, float] | None:
    """Minimize the gadget bloat F by LP: squares inside [0, w + F] x [0, h + F], angles fixed."""
    Q, F = P, f0
    for _ in range(iters):
        r = lp.solve(Q, trust=trust, box=(w, h))
        if r.status != 0 or min_gap(r.P) < -1e-9:
            break
        if r.L < F - 1e-12:
            Q, F = r.P, r.L
        else:
            break
    return Q, F


def extract(P: np.ndarray, source: str = "", do_tighten: bool = True, maxdim: int = 31) -> list[Gadget]:
    """Cut a fault line packing into gadgets. Returns [] when P does not have that structure."""
    from .geometry import normalize

    P = normalize(P)
    L = side(P)
    K = int(math.floor(L + 1e-9))
    f = L - K
    if f < 1e-6 or f > 1 - 1e-6 or K < 2:
        return []
    up = np.abs(fold_angle(P[:, 2])) < 1e-7
    cell = -np.ones((K, K), dtype=np.int64)  # index of the lattice square in each cell
    kind = np.zeros(len(P), dtype=np.int64)  # 0 core, 1 blue, 2 green
    for i in np.nonzero(up)[0]:
        x, y = P[i, 0] - 0.5, P[i, 1] - 0.5
        for k, (ox, oy) in ((1, (0.0, 0.0)), (2, (f, f))):
            u, v = x - ox, y - oy
            iu, iv = round(u), round(v)
            if abs(u - iu) < TOL and abs(v - iv) < TOL and 0 <= iu < K and 0 <= iv < K and cell[iu, iv] < 0:
                cell[iu, iv] = i
                kind[i] = k
                break
    free = cell < 0
    core = np.nonzero(kind == 0)[0]
    if len(core) == 0:
        return []
    # components of free cells, 4 connectivity
    comp = -np.ones((K, K), dtype=np.int64)
    nc = 0
    for a in range(K):
        for b in range(K):
            if free[a, b] and comp[a, b] < 0:
                stack = [(a, b)]
                comp[a, b] = nc
                while stack:
                    u, v = stack.pop()
                    for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        p, q = u + du, v + dv
                        if 0 <= p < K and 0 <= q < K and free[p, q] and comp[p, q] < 0:
                            comp[p, q] = nc
                            stack.append((p, q))
                nc += 1
    boxes = []
    for c in range(nc):
        us, vs = np.nonzero(comp == c)
        boxes.append([us.min(), vs.min(), us.max() + 1, vs.max() + 1])
    # merge overlapping cell boxes
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    boxes.pop(j)
                    merged = True
                    break
            if merged:
                break
    out = []
    C = corners(P)
    for x0, y0, x1, y1 in boxes:
        w, h = x1 - x0, y1 - y0
        if w > maxdim or h > maxdim:
            continue  # cannot fit any container we score (K <= 31)
        # members: core squares whose centre lies in the box, plus lattice squares in its cells
        cx, cy = P[:, 0], P[:, 1]
        inbox = (cx > x0) & (cx < x1 + f) & (cy > y0) & (cy < y1 + f)
        lat = np.zeros(len(P), dtype=bool)
        sub = cell[x0:x1, y0:y1]
        lat[sub[sub >= 0]] = True
        mem = (inbox & (kind == 0)) | lat
        idx = np.nonzero(mem)[0]
        d = len(idx) - w * h
        if d <= 0:
            continue
        Cm = C[idx]
        if (Cm[..., 0].min() < x0 - 1e-7 or Cm[..., 0].max() > x1 + f + 1e-7 or
                Cm[..., 1].min() < y0 - 1e-7 or Cm[..., 1].max() > y1 + f + 1e-7):
            continue  # a member sticks out: not a clean gadget
        G = P[idx].copy()
        G[:, 0] -= x0
        G[:, 1] -= y0
        fg = f
        if do_tighten:
            G, fg = tighten(G, w, h, f)
        tl, br = _corner_flags(G, w, h, fg)
        out.append(Gadget(w, h, d, fg, G, source, tl, br))
    return out


def transpose(g: Gadget) -> Gadget:
    """Reflect across the main diagonal: (w, h) -> (h, w); angles theta -> -theta."""
    Q = g.P[:, [1, 0, 2]].copy()
    Q[:, 2] = -Q[:, 2]
    return Gadget(g.h, g.w, g.d, g.f, Q, g.source + ":T", g.tl_free, g.br_free)


@dataclass
class Library:
    best: dict = field(default_factory=dict)  # (w, h, d) -> Gadget with the smallest f

    def add(self, g: Gadget) -> bool:
        if g.f >= 1 - 1e-9 or g.d <= 0:
            return False
        added = False
        for h in (g, transpose(g)):
            k = h.key
            if k not in self.best or h.f < self.best[k].f - 1e-12:
                self.best[k] = h
                added = True
        return added

    def __len__(self):
        return len(self.best)


try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def _dp_kernel(K, gw, gh, gd, gtl, gbr):
    NEG = -(10**9)
    best = np.full((K + 1, K + 1, 3), NEG, dtype=np.int64)
    bk = np.full((K + 1, K + 1, 3, 4), -1, dtype=np.int64)  # prevW, prevH, prev corner, gadget index
    for W in range(K + 1):
        for H in range(K + 1):
            best[W, H, 2] = 0
    for s in range(K * 2 + 1):
        for W in range(max(0, s - K), min(K, s) + 1):
            H = s - W
            for gi in range(len(gw)):
                for prev in range(3):
                    ok = prev != 0 or gtl[gi]
                    cw = gw[gi] + (0 if ok else 1)
                    if cw > W or gh[gi] > H:
                        continue
                    v = best[W - cw, H - gh[gi], prev]
                    if v == NEG:
                        continue
                    c = 1 if gbr[gi] else 0
                    if v + gd[gi] > best[W, H, c]:
                        best[W, H, c] = v + gd[gi]
                        bk[W, H, c, 0] = W - cw
                        bk[W, H, c, 1] = H - gh[gi]
                        bk[W, H, c, 2] = prev
                        bk[W, H, c, 3] = gi
    return best, bk


if njit is not None:
    _dp_kernel = njit(cache=True)(_dp_kernel)


def pareto(gs: list[Gadget]) -> list[Gadget]:
    """Drop gadgets dominated in (w, h, f smaller is better; d larger; corners freer)."""
    gs = sorted(gs, key=lambda g: (g.f, -g.d))
    keep: list[Gadget] = []
    for g in gs:
        dom = False
        for k in keep:
            if (k.w <= g.w and k.h <= g.h and k.d >= g.d and k.f <= g.f
                    and (k.tl_free or not g.tl_free) and (k.br_free or not g.br_free)):
                dom = True
                break
        if not dom:
            keep.append(g)
    return keep


def compose_fast(gs: list[Gadget], K: int, fmax: float):
    """compose_dp with a compiled kernel over a prepared gadget list."""
    use = [g for g in gs if g.f <= fmax + 1e-15 and g.w <= K and g.h <= K]
    if not use:
        return 0, []
    gw = np.array([g.w for g in use], dtype=np.int64)
    gh = np.array([g.h for g in use], dtype=np.int64)
    gd = np.array([g.d for g in use], dtype=np.int64)
    gtl = np.array([g.tl_free for g in use], dtype=np.bool_)
    gbr = np.array([g.br_free for g in use], dtype=np.bool_)
    best, bk = _dp_kernel(K, gw, gh, gd, gtl, gbr)
    c = int(np.argmax(best[K, K]))
    D = int(best[K, K, c])
    chain = []
    W, H = K, K
    while bk[W, H, c, 3] >= 0:
        W0, H0, prev, gi = (int(v) for v in bk[W, H, c])
        chain.append((use[gi], W - W0 - use[gi].w))
        W, H, c = W0, H0, prev
    chain.reverse()
    return D, chain


def compose_dp(lib: Library, K: int, fmax: float):
    """Max sum d over gadget chains with sum w <= K, sum h <= K, every f <= fmax.

    State (W, H, last bottom right corner free?). A corner to corner touch between a
    gadget with an occupied bottom right corner and one with an occupied top left corner
    costs a buffer column (+1 to W). Returns (D, chain list of gadgets).
    """
    gs = [g for g in lib.best.values() if g.f <= fmax + 1e-15 and g.w <= K and g.h <= K]
    if not gs:
        return 0, []
    NEG = -10**9
    best = np.full((K + 1, K + 1, 3), NEG, dtype=np.int64)  # last corner: 0 occupied, 1 free, 2 start
    best[:, :, 2] = 0
    back = {}
    order = sorted(((W, H) for W in range(K + 1) for H in range(K + 1)), key=lambda t: t[0] + t[1])
    for W, H in order:
        for gi, g in enumerate(gs):
            for prev in (0, 1, 2):
                ok = prev != 0 or g.tl_free
                cw = g.w + (0 if ok else 1)
                if cw > W or g.h > H:
                    continue
                v = best[W - cw, H - g.h, prev]
                if v == NEG:
                    continue
                c = 1 if g.br_free else 0
                if v + g.d > best[W, H, c]:
                    best[W, H, c] = v + g.d
                    back[(W, H, c)] = (W - cw, H - g.h, prev, gi, cw - g.w)
    D = int(best[K, K].max())
    c = int(np.argmax(best[K, K]))
    chain = []
    s = (K, K, c)
    while s in back:
        W0, H0, prev, gi, buf = back[s]
        chain.append((gs[gi], buf))
        s = (W0, H0, prev)
    chain.reverse()
    return D, chain


def realize(chain, K: int) -> np.ndarray:
    """Place a gadget chain on the fault line of a K + f container (f = max chain bloat)."""
    f = max(g.f for g, _ in chain)
    used = np.zeros((K, K), dtype=bool)
    parts = []
    x, ytop = 0, K
    boxes = []
    for g, buf in chain:
        x += buf
        y0 = ytop - g.h
        Q = g.P.copy()
        # a gadget tightened to bloat g.f < f still fits: anchor it at its blue (bottom left) corner
        Q[:, 0] += x
        Q[:, 1] += y0
        parts.append(Q)
        used[x:x + g.w, y0:ytop] = True
        boxes.append((x, y0, g.w, g.h))
        x += g.w
        ytop = y0
    # seam: cells below/left of the chain are blue, above/right are green
    low = np.zeros(K, dtype=np.int64)  # column u: rows below low[u] are blue
    xs = 0
    for (x0, y0, w, h) in boxes:
        low[xs:x0] = y0 + h if xs < x0 else low[xs:x0]
        low[x0:x0 + w] = y0
        xs = x0 + w
    if xs < K:
        low[xs:] = boxes[-1][1] if boxes else 0
    sq = []
    for u in range(K):
        for v in range(K):
            if used[u, v]:
                continue
            if v < low[u]:
                sq.append((u + 0.5, v + 0.5, 0.0))
            else:
                sq.append((u + 0.5 + f, v + 0.5 + f, 0.0))
    return np.vstack(parts + [np.array(sq).reshape(-1, 3)])
