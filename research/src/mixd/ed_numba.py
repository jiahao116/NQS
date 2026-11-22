"""Matrix-free exact diagonalization of spin-1/2 Heisenberg models (numba + scipy).

Independent of QuSpin: used as a second ED path to cross-check QuSpin and quantax.
Works in a fixed-magnetization sector (n_up spins up) with Lin-table O(1) state
lookup. Feasible on a laptop up to ~24-26 sites (dimension a few million).

H = sum_b J_b S_i . S_j  with spin-1/2 operators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh


@nb.njit(cache=True)
def _popcount(v: np.int64) -> np.int64:
    c = 0
    while v:
        v &= v - 1
        c += 1
    return c


@nb.njit(cache=True)
def _build_basis(n_sites: int, n_up: int, n_low: int):
    n_high = n_sites - n_low
    size_low = 1 << n_low
    size_high = 1 << n_high
    pop_low = np.empty(size_low, dtype=np.int64)
    rank_low = np.empty(size_low, dtype=np.int64)
    counter = np.zeros(n_low + 1, dtype=np.int64)
    for low in range(size_low):
        p = _popcount(low)
        pop_low[low] = p
        rank_low[low] = counter[p]
        counter[p] += 1
    offset = np.full(size_high, -1, dtype=np.int64)
    total = 0
    for high in range(size_high):
        need = n_up - _popcount(high)
        if 0 <= need <= n_low:
            offset[high] = total
            total += counter[need]
    states = np.empty(total, dtype=np.int64)
    k = 0
    for high in range(size_high):
        need = n_up - _popcount(high)
        if 0 <= need <= n_low:
            for low in range(size_low):
                if pop_low[low] == need:
                    states[k] = (high << n_low) | low
                    k += 1
    return states, offset, rank_low


@nb.njit(cache=True, inline="always")
def _lookup(s, n_low, low_mask, offset, rank_low):
    return offset[s >> n_low] + rank_low[s & low_mask]


@nb.njit(cache=True, parallel=True)
def _matvec(x, states, offset, rank_low, n_low, bi, bj, bJ):
    dim = states.shape[0]
    low_mask = (1 << n_low) - 1
    y = np.zeros(dim, dtype=np.float64)
    nb_bonds = bi.shape[0]
    for a in nb.prange(dim):
        s = states[a]
        acc = 0.0
        diag = 0.0
        for k in range(nb_bonds):
            i = bi[k]
            j = bj[k]
            si = (s >> i) & 1
            sj = (s >> j) & 1
            if si == sj:
                diag += 0.25 * bJ[k]
            else:
                diag -= 0.25 * bJ[k]
                t = s ^ ((1 << i) | (1 << j))
                b = _lookup(t, n_low, low_mask, offset, rank_low)
                acc += 0.5 * bJ[k] * x[b]
        y[a] = acc + diag * x[a]
    return y


@nb.njit(cache=True, parallel=True)
def _correlations(psi, states, offset, rank_low, n_low, n_sites):
    """C[i, j] = <S_i . S_j> for all pairs (real ground state assumed)."""
    dim = states.shape[0]
    low_mask = (1 << n_low) - 1
    n_pairs = n_sites * (n_sites - 1) // 2
    pi = np.empty(n_pairs, dtype=np.int64)
    pj = np.empty(n_pairs, dtype=np.int64)
    k = 0
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            pi[k] = i
            pj[k] = j
            k += 1
    vals = np.zeros(n_pairs, dtype=np.float64)
    for p in nb.prange(n_pairs):
        i = pi[p]
        j = pj[p]
        acc = 0.0
        for a in range(dim):
            s = states[a]
            w = psi[a]
            if w == 0.0:
                continue
            si = (s >> i) & 1
            sj = (s >> j) & 1
            if si == sj:
                acc += 0.25 * w * w
            else:
                acc -= 0.25 * w * w
                t = s ^ ((1 << i) | (1 << j))
                b = _lookup(t, n_low, low_mask, offset, rank_low)
                acc += 0.5 * w * psi[b]
        vals[p] = acc
    C = np.zeros((n_sites, n_sites), dtype=np.float64)
    for p in range(n_pairs):
        C[pi[p], pj[p]] = vals[p]
        C[pj[p], pi[p]] = vals[p]
    for i in range(n_sites):
        C[i, i] = 0.75
    return C


@dataclass
class EDResult:
    energies: np.ndarray
    psi0: np.ndarray
    dim: int
    n_up: int
    correlations: np.ndarray | None = None


def heisenberg_ed(
    n_sites: int,
    bonds: list[tuple[int, int, float]],
    n_up: int | None = None,
    k: int = 1,
    with_correlations: bool = False,
    tol: float = 1e-12,
    seed: int = 1234,
) -> EDResult:
    if n_sites > 62:
        raise ValueError("bit representation limited to 62 sites")
    if n_up is None:
        n_up = n_sites // 2
    n_low = n_sites // 2
    states, offset, rank_low = _build_basis(n_sites, n_up, n_low)
    dim = states.shape[0]
    bi = np.array([b[0] for b in bonds], dtype=np.int64)
    bj = np.array([b[1] for b in bonds], dtype=np.int64)
    bJ = np.array([b[2] for b in bonds], dtype=np.float64)

    if dim <= 2000:
        H = np.empty((dim, dim))
        eye = np.eye(dim)
        for c in range(dim):
            H[:, c] = _matvec(eye[:, c].copy(), states, offset, rank_low, n_low, bi, bj, bJ)
        w, v = np.linalg.eigh(H)
        energies, vecs = w[:k], v[:, :k]
    else:
        op = LinearOperator(
            (dim, dim),
            matvec=lambda x: _matvec(
                np.ascontiguousarray(x, dtype=np.float64).ravel(),
                states, offset, rank_low, n_low, bi, bj, bJ,
            ),
            dtype=np.float64,
        )
        v0 = np.random.default_rng(seed).standard_normal(dim)
        w, v = eigsh(op, k=k, which="SA", v0=v0, tol=tol)
        order = np.argsort(w)
        energies, vecs = w[order], v[:, order]

    psi0 = vecs[:, 0] / np.linalg.norm(vecs[:, 0])
    res = EDResult(energies=np.asarray(energies), psi0=psi0, dim=dim, n_up=n_up)
    if with_correlations:
        res.correlations = _correlations(psi0, states, offset, rank_low, n_low, n_sites)
    return res
