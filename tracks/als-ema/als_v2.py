"""Explicit-feedback ALS-WR on the core e-means engine (T06 v2).

Full-batch ALS-WR (Zhou et al. 2008): alternately solve every user then every item with
ridge lambda * n_ratings * I. EMA variant (the e-means recipe):

  * mini-batch of users: solve those users exactly against the current item factors;
  * batch item statistics  A_i = sum_u u u^T,  b_i = sum_u r_ui u,  n_i = #ratings, scaled by n_users/B
    so they estimate full-data sums (lambda * n_i is not scale-invariant, so scaling matters);
  * EMA (cosine step) of (n_i, b_i, A_i) in ``EMAStats``; items re-solved from the EMA stats.

Ratings are mean-centred by the global train mean. Loss = RMSE.
"""

from __future__ import annotations

import numba
import numpy as np

from core.plugin import ClassicAlgoPlugin
from core.revival import RevivalEvent
from core.stats import BatchStats, EMAStats


class CSR:
    def __init__(self, rows, cols, vals, n_rows):
        order = np.argsort(rows, kind="stable")
        self.indices = cols[order].astype(np.int64)
        self.data = vals[order].astype(np.float64)
        self.indptr = np.zeros(n_rows + 1, np.int64)
        np.add.at(self.indptr, rows + 1, 1)
        self.indptr = np.cumsum(self.indptr)
        self.n_rows = n_rows


@numba.njit(cache=True, parallel=True)
def solve_rows(row_ids, indptr, indices, data, other, lam):
    """Ridge solve for each row r: (sum v v^T + lam n_r I) x = sum r v."""
    f = other.shape[1]
    out = np.zeros((len(row_ids), f))
    for q in numba.prange(len(row_ids)):
        r = row_ids[q]
        s, e = indptr[r], indptr[r + 1]
        n = e - s
        if n == 0:
            continue
        A = np.zeros((f, f))
        b = np.zeros(f)
        for p in range(s, e):
            v = other[indices[p]]
            A += np.outer(v, v)
            b += data[p] * v
        for j in range(f):
            A[j, j] += lam * n
        out[q] = np.linalg.solve(A, b)
    return out


@numba.njit(cache=True)
def item_batch_stats(user_ids, U_batch, indptr, indices, data, n_items):
    f = U_batch.shape[1]
    A = np.zeros((n_items, f, f))
    b = np.zeros((n_items, f))
    cnt = np.zeros(n_items)
    for q in range(len(user_ids)):
        u = user_ids[q]
        uv = U_batch[q]
        for p in range(indptr[u], indptr[u + 1]):
            i = indices[p]
            cnt[i] += 1.0
            for a in range(f):
                b[i, a] += data[p] * uv[a]
                for c in range(f):
                    A[i, a, c] += uv[a] * uv[c]
    return cnt, b, A


@numba.njit(cache=True, parallel=True)
def solve_from_stats(cnt, b, A, lam, fallback):
    n, f = b.shape
    out = fallback.copy()
    for i in numba.prange(n):
        if cnt[i] < 1e-8:
            continue
        M = A[i].copy()
        for j in range(f):
            M[j, j] += lam * cnt[i] + 1e-10
        out[i] = np.linalg.solve(M, b[i])
    return out


@numba.njit(cache=True, parallel=True)
def _sq_err(u, i, r, U, V):
    out = np.empty(len(u))
    for p in numba.prange(len(u)):
        out[p] = (U[u[p]] @ V[i[p]] - r[p]) ** 2
    return out


def rmse(triples, U, V):
    u, i, r = triples
    return float(np.sqrt(_sq_err(u, i, r, U, V).mean()))


def init_factors(n_users, n_items, f, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.1, (n_users, f)), rng.normal(0, 0.1, (n_items, f))


def als_full_epoch(R_u: CSR, R_i: CSR, U, V, lam):
    U = solve_rows(np.arange(R_u.n_rows), R_u.indptr, R_u.indices, R_u.data, V, lam)
    V = solve_rows(np.arange(R_i.n_rows), R_i.indptr, R_i.indices, R_i.data, U, lam)
    return U, V


class ALSPlugin(ClassicAlgoPlugin):
    """x passed to the engine is ``arange(n_users)[:, None]`` (user ids)."""

    def __init__(self, R_u: CSR, n_items, f, lam, seed=0):
        self.R_u, self.n_items, self.f, self.lam, self.seed = R_u, n_items, f, lam, seed
        self.U = self.V = None

    def init(self, x0):
        self.U, self.V = init_factors(self.R_u.n_rows, self.n_items, self.f, self.seed)
        eye = np.tile(np.eye(self.f), (self.n_items, 1, 1))
        return EMAStats.from_params(self.V, n0=1e-10, extra_inits={"A": eye})

    def assign(self, batch):
        users = np.asarray(batch[:, 0], dtype=np.int64)
        Ub = solve_rows(users, self.R_u.indptr, self.R_u.indices, self.R_u.data, self.V, self.lam)
        self.U[users] = Ub
        cnt, b, A = item_batch_stats(users, Ub, self.R_u.indptr, self.R_u.indices, self.R_u.data, self.n_items)
        scale = self.R_u.n_rows / len(users)
        return BatchStats(counts=cnt * scale, sums=b * scale, extras={"A": A * scale})

    def params_from_stats(self, stats):
        # EMAStats keeps sums of b; A in extras; counts = n_i
        self.V = solve_from_stats(stats.counts, stats.sums, stats.extras["A"], self.lam, self.V)
        return self.V

    def set_params(self, params):
        pass

    def revive(self, stats, step, rng=None):
        e = np.empty(0, np.int64)
        return RevivalEvent(e, e, 0.0)

    def loss(self, x):
        return rmse(x, self.U, self.V)
