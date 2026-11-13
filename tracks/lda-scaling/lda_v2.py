"""Variational LDA on the core e-means engine (T05 v2).

One numba E-step (Hoffman et al. 2010, Alg. 2 inner loop) is shared by every method:
  batch VB        lambda = eta + sum_{all docs} sstats             (reference, 25 passes)
  online LDA      lambda <- (1-rho_t) lambda + rho_t (eta + D/|S| sstats),  rho_t = (tau0+t)^-kappa
  cosine EMA      same EMA with the e-means cosine step + topic revival      (ours)

The EMA state is ``EMAStats`` with sums = lambda - eta (init: Gamma(100, 1/100)), counts = topic mass.

Evaluation = document-completion perplexity: each test doc's tokens are split 50/50 (by unique
word id, fixed seed); theta is inferred on the first half with lambda fixed; perplexity is computed
on the second half with beta = lambda / rowsum(lambda).
"""

from __future__ import annotations

import math

import numba
import numpy as np

from core.plugin import ClassicAlgoPlugin
from core.revival import revive_dead_units
from core.stats import BatchStats, EMAStats

ALPHA = 0.1  # doc-topic prior (1/K-style priors are set per K below)
ETA = 0.01


@numba.njit(cache=True)
def _digamma(x):
    r = 0.0
    while x < 6.0:
        r -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    return r + math.log(x) - 0.5 / x - f * (1.0 / 12 - f * (1.0 / 120 - f * (1.0 / 252 - f * (1.0 / 240 - f / 132))))


@numba.njit(cache=True, parallel=True)
def exp_elog_beta(lam):
    K, V = lam.shape
    out = np.empty_like(lam)
    for k in numba.prange(K):
        s = _digamma(lam[k].sum())
        for v in range(V):
            out[k, v] = math.exp(_digamma(lam[k, v]) - s)
    return out


@numba.njit(cache=True, parallel=True)
def e_step(docs, indptr, ids, cts, eeb, alpha, max_iter, tol, want_sstats):
    """Returns (gamma (n,K), sstats (K,V) premultiplied by eeb)."""
    K, V = eeb.shape
    n = len(docs)
    gamma = np.empty((n, K))
    nthreads = numba.get_num_threads()
    sst = np.zeros((nthreads, K, V)) if want_sstats else np.zeros((1, 1, 1))
    for q in numba.prange(n):
        d = docs[q]
        s, e = indptr[d], indptr[d + 1]
        w = ids[s:e]
        c = cts[s:e]
        L = e - s
        g = np.ones(K)
        eet = np.empty(K)
        phin = np.empty(L)
        for it in range(max_iter):
            gs = _digamma(g.sum())
            for k in range(K):
                eet[k] = math.exp(_digamma(g[k]) - gs)
            for j in range(L):
                acc = 1e-100
                for k in range(K):
                    acc += eet[k] * eeb[k, w[j]]
                phin[j] = acc
            newg = np.empty(K)
            for k in range(K):
                acc = 0.0
                for j in range(L):
                    acc += c[j] / phin[j] * eeb[k, w[j]]
                newg[k] = alpha + eet[k] * acc
            diff = np.abs(newg - g).mean()
            g = newg
            if diff < tol:
                break
        gamma[q] = g
        if want_sstats:
            gs = _digamma(g.sum())
            for k in range(K):
                eet[k] = math.exp(_digamma(g[k]) - gs)
            for j in range(L):
                acc = 1e-100
                for k in range(K):
                    acc += eet[k] * eeb[k, w[j]]
                phin[j] = acc
            tid = numba.get_thread_id()
            for j in range(L):
                for k in range(K):
                    sst[tid, k, w[j]] += eet[k] * c[j] / phin[j]
    if want_sstats:
        out = sst.sum(0) * eeb
        return gamma, out
    return gamma, sst[0]


@numba.njit(cache=True, parallel=True)
def _heldout_ll(docs, indptr, ids, cts, theta, beta):
    tot = np.zeros(len(docs))
    ntok = np.zeros(len(docs))
    for q in numba.prange(len(docs)):
        d = docs[q]
        for p in range(indptr[d], indptr[d + 1]):
            pr = 0.0
            for k in range(theta.shape[1]):
                pr += theta[q, k] * beta[k, ids[p]]
            tot[q] += cts[p] * math.log(max(pr, 1e-300))
            ntok[q] += cts[p]
    return tot.sum(), ntok.sum()


class Corpus:
    """CSR bag of words."""

    def __init__(self, indptr, ids, cts, V):
        self.indptr, self.ids, self.cts, self.V = indptr.astype(np.int64), ids.astype(np.int64), cts.astype(np.float64), V
        self.n = len(indptr) - 1

    def subset(self, rows):
        rows = np.asarray(rows)
        lens = self.indptr[rows + 1] - self.indptr[rows]
        ip = np.concatenate([[0], np.cumsum(lens)])
        sel = np.concatenate([np.arange(self.indptr[r], self.indptr[r + 1]) for r in rows]) if len(rows) else np.zeros(0, np.int64)
        return Corpus(ip, self.ids[sel], self.cts[sel], self.V)

    def completion_split(self, seed=0):
        rng = np.random.default_rng(seed)
        obs, held = [], []
        for d in range(self.n):
            s, e = self.indptr[d], self.indptr[d + 1]
            m = rng.random(e - s) < 0.5
            obs.append(np.arange(s, e)[m])
            held.append(np.arange(s, e)[~m])
        def build(parts):
            ip = np.concatenate([[0], np.cumsum([len(p) for p in parts])])
            sel = np.concatenate(parts)
            return Corpus(ip, self.ids[sel], self.cts[sel], self.V)
        return build(obs), build(held)


def perplexity(lam, obs: Corpus, held: Corpus, alpha):
    eeb = exp_elog_beta(lam)
    docs = np.arange(obs.n)
    gamma, _ = e_step(docs, obs.indptr, obs.ids, obs.cts, eeb, alpha, 100, 1e-3, False)
    theta = gamma / gamma.sum(1, keepdims=True)
    beta = lam / lam.sum(1, keepdims=True)
    ll, nt = _heldout_ll(docs, held.indptr, held.ids, held.cts, theta, beta)
    return float(math.exp(-ll / nt))


def npmi(lam, train: Corpus, top=10, max_docs=20000):
    """Mean NPMI of each topic's top words, document co-occurrence on (a subset of) train."""
    tops = np.argsort(-lam, axis=1)[:, :top]
    words = np.unique(tops)
    idx = {w: i for i, w in enumerate(words)}
    D = min(train.n, max_docs)
    occ = np.zeros((D, len(words)), dtype=bool)
    for d in range(D):
        for w in train.ids[train.indptr[d]:train.indptr[d + 1]]:
            j = idx.get(int(w))
            if j is not None:
                occ[d, j] = True
    p1 = occ.mean(0)
    co = (occ.T.astype(np.float64) @ occ.astype(np.float64)) / D
    scores = []
    for t in tops:
        vals = []
        for a in range(top):
            for b in range(a + 1, top):
                i, j = idx[int(t[a])], idx[int(t[b])]
                pij = co[i, j]
                if pij <= 0:
                    vals.append(-1.0)
                else:
                    vals.append(math.log(pij / (p1[i] * p1[j])) / -math.log(pij))
        scores.append(np.mean(vals))
    return float(np.mean(scores))


def init_lambda(K, V, seed):
    return np.random.default_rng(seed).gamma(100.0, 0.01, (K, V))


class LDAPlugin(ClassicAlgoPlugin):
    """Engine input x = arange(D)[:, None] (doc ids)."""

    def __init__(self, corpus: Corpus, K, seed=0, alpha=None, max_iter=50, tau=0.1):
        self.c, self.K, self.seed = corpus, K, seed
        self.alpha = alpha if alpha is not None else 1.0 / K
        self.max_iter, self.tau = max_iter, tau
        self.lam = None
        self.eeb = None

    def init(self, x0):
        self.lam = init_lambda(self.K, self.c.V, self.seed) + ETA
        self.eeb = exp_elog_beta(self.lam)
        return EMAStats.from_params(self.lam - ETA, n0=1.0)

    def assign(self, batch):
        docs = np.asarray(batch[:, 0], dtype=np.int64)
        _, sst = e_step(docs, self.c.indptr, self.c.ids, self.c.cts, self.eeb, self.alpha, self.max_iter, 1e-3, True)
        sst *= self.c.n / len(docs)
        return BatchStats(counts=sst.sum(1), sums=sst)

    def params_from_stats(self, stats):
        self.lam = ETA + np.maximum(stats.sums, 0.0)
        self.eeb = exp_elog_beta(self.lam)
        return self.lam

    def set_params(self, params):
        pass

    def revive(self, stats, step, rng=None):
        return revive_dead_units(stats, tau=self.tau, delta=0.05, rng=rng)

    def loss(self, x):
        raise NotImplementedError


def batch_vb_step(corpus: Corpus, lam, alpha, max_iter=50):
    eeb = exp_elog_beta(lam)
    _, sst = e_step(np.arange(corpus.n), corpus.indptr, corpus.ids, corpus.cts, eeb, alpha, max_iter, 1e-3, True)
    return ETA + sst
