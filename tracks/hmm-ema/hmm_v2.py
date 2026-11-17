"""Discrete HMM on the core e-means engine (T07 v2).

Sequences are stored as a padded int matrix (n_seqs, L_max) with -1 padding so the generic
``run_ema_engine`` can slice mini-batches of sequences. Sufficient statistics per state j:

  counts[j]          expected occupancy  sum_t gamma_t(j)
  sums[j, v]         expected emissions  sum_t gamma_t(j) [o_t = v]
  extras["trans"]    expected transitions (S, S)   sum_t xi_t(i, j)
  extras["init"]     expected initial state (S,)   gamma_0(j)

The M-step normalises each block, so batch statistics need no n/B rescaling. Full-batch
Baum-Welch uses the exact same E-step and M-step code (``em_step``), so the only difference
between reference and EMA is the update rule.
"""

from __future__ import annotations

import numba
import numpy as np

from core.device import gpu_port
from core.plugin import ClassicAlgoPlugin
from core.revival import RevivalEvent
from core.stats import BatchStats, EMAStats

SMOOTH = 1e-6
# CUDA port (CLASSIC_ALGOS_GPU_PORTS=hmm), routed by 5090 bench vs 4 CPU threads: forward-backward wins from 64 sequences
# (x1.3, x6-14 at V=5000); the forward-only log-likelihood is faster in numba at every size, so it never routes
HMM_GPU_MIN_SEQS = 64


def _fb_batch(seqs, log_pi, A, B):
    """Scaled forward-backward over a padded batch. Returns stats and total log-lik."""
    if gpu_port("hmm") and len(seqs) >= HMM_GPU_MIN_SEQS:
        return _fb_torch(seqs, log_pi, A, B, want_stats=True)
    return _fb_batch_numba(seqs, log_pi, A, B)


def _ll_batch(seqs, log_pi, A, B):
    return _ll_batch_numba(seqs, log_pi, A, B)


def _fb_torch(seqs, log_pi, A, B, want_stats):
    """torch CUDA port: all sequences advance one time step together (float64, padding masked).
    Same scaled recursions as the numba kernels; returns their outputs."""
    import torch

    dev = "cuda"
    seqs = np.asarray(seqs, dtype=np.int64)
    n, L = seqs.shape
    S, V = B.shape
    O = torch.as_tensor(seqs, device=dev)
    valid = O >= 0
    # length = first padding position (padding is a suffix in the numba kernels' reading too)
    Tq = torch.where(valid.all(1), torch.full((n,), L, device=dev), (~valid).int().argmax(1))
    tpos = torch.arange(L, device=dev)
    m = tpos[None, :] < Tq[:, None]  # (n, L)
    At = torch.as_tensor(np.asarray(A, dtype=np.float64), device=dev)
    Bt = torch.as_tensor(np.asarray(B, dtype=np.float64), device=dev)
    pi = torch.exp(torch.as_tensor(np.asarray(log_pi, dtype=np.float64), device=dev))
    Bo = Bt.T[torch.clamp(O, min=0)]  # (n, L, S) emission prob of each observed symbol
    alpha = torch.zeros((n, L, S), dtype=torch.float64, device=dev)
    c = torch.ones((n, L), dtype=torch.float64, device=dev)
    a = pi[None, :] * Bo[:, 0]
    c0 = a.sum(1)
    c[:, 0] = torch.where(m[:, 0], c0, 1.0)
    alpha[:, 0] = a / c[:, 0:1]
    for t in range(1, L):
        a = (alpha[:, t - 1] @ At) * Bo[:, t]
        ct = torch.where(m[:, t], a.sum(1), 1.0)
        c[:, t] = ct
        alpha[:, t] = torch.where(m[:, t:t + 1], a / ct[:, None], alpha[:, t - 1])
    ll = torch.where(m, torch.log(c), 0.0).sum()
    if not want_stats:
        return float(ll), float(m.sum())
    beta = torch.ones((n, L, S), dtype=torch.float64, device=dev)
    for t in range(L - 2, -1, -1):
        nxt = m[:, t + 1:t + 2]  # beta_t uses step t+1 only if it exists for this sequence
        b = ((Bo[:, t + 1] * beta[:, t + 1]) @ At.T) / c[:, t + 1:t + 2]
        beta[:, t] = torch.where(nxt, b, 1.0)
    g = alpha * beta * m[:, :, None]  # (n, L, S)
    occ = g.sum((0, 1))
    emis = torch.zeros((V, S), dtype=torch.float64, device=dev)
    emis.index_add_(0, O[m], g[m])
    init = g[:, 0].sum(0)
    # xi_t(i, j) = alpha_t(i) A_ij B_j(o_t+1) beta_t+1(j) / c_t+1, for t+1 inside the sequence
    q = Bo[:, 1:] * beta[:, 1:] / c[:, 1:, None] * m[:, 1:, None]
    trans = At * torch.einsum("nti,ntj->ij", alpha[:, :-1], q)
    return (occ.cpu().numpy(), emis.T.contiguous().cpu().numpy(), trans.cpu().numpy(), init.cpu().numpy(), float(ll))


@numba.njit(cache=True, parallel=True)
def _fb_batch_numba(seqs, log_pi, A, B):
    """Scaled forward-backward over a padded batch. Returns stats and total log-lik."""
    n, L = seqs.shape
    S, V = B.shape
    pi = np.exp(log_pi)
    occ = np.zeros((n, S))
    emis = np.zeros((n, S, V))
    trans = np.zeros((n, S, S))
    init = np.zeros((n, S))
    ll = np.zeros(n)
    for q in numba.prange(n):
        T = 0
        while T < L and seqs[q, T] >= 0:
            T += 1
        if T == 0:
            continue
        alpha = np.zeros((T, S))
        c = np.zeros(T)
        o = seqs[q]
        for j in range(S):
            alpha[0, j] = pi[j] * B[j, o[0]]
        c[0] = alpha[0].sum()
        alpha[0] /= c[0]
        for t in range(1, T):
            for j in range(S):
                s = 0.0
                for i in range(S):
                    s += alpha[t - 1, i] * A[i, j]
                alpha[t, j] = s * B[j, o[t]]
            c[t] = alpha[t].sum()
            alpha[t] /= c[t]
        beta = np.ones((T, S))
        for t in range(T - 2, -1, -1):
            for i in range(S):
                s = 0.0
                for j in range(S):
                    s += A[i, j] * B[j, o[t + 1]] * beta[t + 1, j]
                beta[t, i] = s / c[t + 1]
        for t in range(T):
            for j in range(S):
                g = alpha[t, j] * beta[t, j]
                occ[q, j] += g
                emis[q, j, o[t]] += g
                if t == 0:
                    init[q, j] += g
        for t in range(T - 1):
            for i in range(S):
                for j in range(S):
                    trans[q, i, j] += alpha[t, i] * A[i, j] * B[j, o[t + 1]] * beta[t + 1, j] / c[t + 1]
        tot = 0.0
        for t in range(T):
            tot += np.log(c[t])
        ll[q] = tot
    return occ.sum(0), emis.sum(0), trans.sum(0), init.sum(0), ll.sum()


@numba.njit(cache=True, parallel=True)
def _ll_batch_numba(seqs, log_pi, A, B):
    n, L = seqs.shape
    S = B.shape[0]
    pi = np.exp(log_pi)
    ll = np.zeros(n)
    toks = np.zeros(n)
    for q in numba.prange(n):
        a = np.empty(S)
        a2 = np.empty(S)
        o = seqs[q]
        if L == 0 or o[0] < 0:
            continue
        for j in range(S):
            a[j] = pi[j] * B[j, o[0]]
        c = a.sum()
        a /= c
        tot = np.log(c)
        t = 1
        while t < L and o[t] >= 0:
            for j in range(S):
                s = 0.0
                for i in range(S):
                    s += a[i] * A[i, j]
                a2[j] = s * B[j, o[t]]
            c = a2.sum()
            a[:] = a2 / c
            tot += np.log(c)
            t += 1
        ll[q] = tot
        toks[q] = t
    return ll.sum(), toks.sum()


@numba.njit(cache=True, parallel=True)
def _viterbi_batch(seqs, log_pi, logA, logB):
    n, L = seqs.shape
    S = logB.shape[0]
    out = -np.ones((n, L), dtype=np.int64)
    for q in numba.prange(n):
        o = seqs[q]
        T = 0
        while T < L and o[T] >= 0:
            T += 1
        if T == 0:
            continue
        dp = np.empty((T, S))
        bp = np.zeros((T, S), dtype=np.int64)
        for j in range(S):
            dp[0, j] = log_pi[j] + logB[j, o[0]]
        for t in range(1, T):
            for j in range(S):
                best, arg = -1e300, 0
                for i in range(S):
                    v = dp[t - 1, i] + logA[i, j]
                    if v > best:
                        best, arg = v, i
                dp[t, j] = best + logB[j, o[t]]
                bp[t, j] = arg
        j = int(np.argmax(dp[T - 1]))
        for t in range(T - 1, -1, -1):
            out[q, t] = j
            j = bp[t, j]
    return out


def _norm_rows(c):
    """Row-normalise with smoothing RELATIVE to row mass (scale-invariant). An absolute pseudo-count
    swamps the engine's n0=1e-10 initial statistics and maps a random init to the exactly uniform,
    symmetric HMM, which is an EM fixed point (bug found on UCI HAR, 2026-11-13)."""
    c = np.atleast_2d(c)
    mass = c.sum(1, keepdims=True)
    c = c + SMOOTH * mass / c.shape[1] + 1e-300
    return c / c.sum(1, keepdims=True)


def params_from(counts, emis, trans, init):
    return np.log(_norm_rows(init)[0]), _norm_rows(trans), _norm_rows(emis)


def random_init(S, V, seed, emis_prior=None):
    """Dirichlet init; emissions centred on the unigram distribution (standard for POS induction)."""
    rng = np.random.default_rng(seed)
    pi = rng.dirichlet(np.ones(S))
    A = rng.dirichlet(np.ones(S), size=S)
    base = emis_prior if emis_prior is not None else np.full(V, 1.0 / V)
    B = base[None, :] * rng.uniform(0.5, 1.5, (S, V))
    B /= B.sum(1, keepdims=True)
    return np.log(pi), A, B


class HMMPlugin(ClassicAlgoPlugin):
    def __init__(self, S, V, seed=0, tau=0.1, delta=0.05, emis_prior=None):
        self.S, self.V, self.seed, self.tau, self.delta = S, V, seed, tau, delta
        self.emis_prior = emis_prior
        self.log_pi = self.A = self.B = None

    def init(self, x0):
        self.log_pi, self.A, self.B = random_init(self.S, self.V, self.seed, self.emis_prior)
        st = EMAStats.from_params(self.B, n0=1e-10,
                                  extra_inits={"trans": self.A, "init": np.exp(self.log_pi)[:, None]})
        return st

    def assign(self, batch):
        seqs = np.asarray(batch, dtype=np.int64)
        occ, emis, trans, init, _ = _fb_batch(seqs, self.log_pi, self.A, self.B)
        return BatchStats(counts=occ, sums=emis, extras={"trans": trans, "init": init[:, None]})

    def params_from_stats(self, stats):
        self.log_pi, self.A, self.B = params_from(stats.counts, stats.sums, stats.extras["trans"],
                                                  stats.extras["init"][:, 0])
        return self.B

    def set_params(self, params):
        pass  # finalize_mode='none' for HMMs

    def revive(self, stats, step, rng=None):
        """Split the heaviest-occupancy donors into dead states: emissions, outgoing AND incoming
        transitions are halved, emissions perturbed by +-delta so the twins can separate."""
        counts = stats.counts
        mean = counts.mean()
        dead = np.flatnonzero(counts < self.tau * mean)
        empty = np.empty(0, dtype=np.int64)
        if len(dead) == 0 or mean <= 0:
            return RevivalEvent(empty, empty, self.delta)
        rng = rng or np.random.default_rng()
        live = np.flatnonzero(counts >= self.tau * mean)
        donors = rng.choice(live, size=len(dead), replace=len(dead) > len(live), p=counts[live] / counts[live].sum())
        tr, ini = stats.extras["trans"], stats.extras["init"]
        sign = rng.choice([-1.0, 1.0], size=stats.sums.shape[1])
        for dj, di in zip(dead, donors):
            for arr in (stats.counts, ini):
                arr[dj] = arr[di] = 0.5 * arr[di]
            e = 0.5 * stats.sums[di]
            stats.sums[dj] = e * (1 + self.delta * sign)
            stats.sums[di] = e * (1 - self.delta * sign)
            tr[dj] = tr[di] = 0.5 * tr[di]              # outgoing
            tr[:, dj] = tr[:, di] = 0.5 * tr[:, di]     # incoming
        return RevivalEvent(dead, donors, self.delta)

    def loss(self, x):
        return nll_per_token(x, self.log_pi, self.A, self.B)


def nll_per_token(seqs, log_pi, A, B):
    ll, toks = _ll_batch(np.asarray(seqs, dtype=np.int64), log_pi, A, B)
    return float(-ll / max(toks, 1))


def em_step(seqs, log_pi, A, B, chunk=4096):
    occ = emis = trans = init = None
    for s in range(0, len(seqs), chunk):
        o, e, t, i, _ = _fb_batch(seqs[s:s + chunk], log_pi, A, B)
        if occ is None:
            occ, emis, trans, init = o, e, t, i
        else:
            occ += o; emis += e; trans += t; init += i
    return params_from(occ, emis, trans, init)


def many_to_one(seqs, tags, log_pi, A, B):
    """Many-to-one accuracy: map each induced state to its most frequent gold tag."""
    pred = _viterbi_batch(np.asarray(seqs, np.int64), log_pi, np.log(A), np.log(B))
    m = seqs >= 0
    p, g = pred[m], tags[m]
    S = A.shape[0]
    G = int(g.max()) + 1
    conf = np.zeros((S, G))
    np.add.at(conf, (p, g), 1)
    return float(conf.max(1).sum() / m.sum())


def pad(seqs, max_len=None):
    L = max_len or max(len(s) for s in seqs)
    out = -np.ones((len(seqs), L), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[:L]
        out[i, :len(s)] = s
    return out
