"""Parity of the torch CUDA ports (CLASSIC_ALGOS_GPU_PORTS) against the CPU reference paths, on fixed seeds.

A port may be used for results only while its test here passes. Skipped without CUDA.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA ports need a CUDA GPU")


def _load(track: str, mod: str):
    """Import tracks/<track>/<mod>.py under its own bare name, as the tracks do: numba's on-disk cache pickles the
    module name, so a made-up name poisons the cache for the real runs. The five ported modules have unique names."""
    tdir = ROOT / "tracks" / track
    if str(tdir) not in sys.path:
        sys.path.insert(0, str(tdir))
    name = mod
    if name in sys.modules and Path(sys.modules[name].__file__).resolve() == (tdir / f"{mod}.py").resolve():
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, tdir / f"{mod}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# size-routing thresholds, zeroed so the small test problems really take the CUDA path
_THRESHOLDS = {"gmm": ("gmm", "GMM_DIAG_GPU_MIN_BATCH"), "dictlearn": ("dictlearn", "FISTA_GPU_MIN_BATCH"),
               "lda": ("lda_v2", "LDA_GPU_MIN_WORK"), "hmm": ("hmm_v2", "HMM_GPU_MIN_SEQS"),
               "als": ("als_v2", "ALS_GPU_MIN_ROWS")}


def _both(monkeypatch, port, fn):
    mod, const = _THRESHOLDS[port]
    if mod in sys.modules and hasattr(sys.modules[mod], const):
        monkeypatch.setattr(sys.modules[mod], const, 0)
    monkeypatch.delenv("CLASSIC_ALGOS_GPU_PORTS", raising=False)
    cpu = fn()
    monkeypatch.setenv("CLASSIC_ALGOS_GPU_PORTS", port)
    gpu = fn()
    monkeypatch.delenv("CLASSIC_ALGOS_GPU_PORTS", raising=False)
    return cpu, gpu


def test_routing_threshold_really_switches(monkeypatch):
    """Guard for _both: with the threshold zeroed a tiny GMM batch must reach the torch path."""
    g = _load("gmm-ema", "gmm")
    called = []
    monkeypatch.setattr(g, "_log_resp_torch", lambda *a: called.append(1) or (np.zeros((2, 1)), np.zeros(2)))
    x = np.zeros((2, 3))
    _both(monkeypatch, "gmm", lambda: g.compute_log_responsibilities(x, np.ones(1), np.zeros((1, 3)), np.ones((1, 3))))
    assert called == [1]


def test_switch(monkeypatch):
    from core.device import gpu_port, gpu_ports

    monkeypatch.delenv("CLASSIC_ALGOS_GPU_PORTS", raising=False)
    assert gpu_ports() == ()
    monkeypatch.setenv("CLASSIC_ALGOS_GPU_PORTS", "gmm, lda")
    assert gpu_port("gmm") and gpu_port("lda") and not gpu_port("hmm")
    monkeypatch.setenv("CLASSIC_ALGOS_GPU_PORTS", "all")
    assert set(gpu_ports()) == {"gmm", "dictlearn", "lda", "hmm", "als"}


# ------------------------------------------------------------------ GMM

def _gmm_problem(cov, k=12, d=16, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)) * 3.0 + 50.0  # offset: exercises cancellation in the expanded diag form
    means = x[rng.choice(n, k, replace=False)]
    w = rng.dirichlet(np.ones(k))
    if cov == "diag":
        cv = rng.uniform(0.5, 4.0, size=(k, d))
    else:
        A = rng.normal(size=(k, d, d))
        cv = A @ A.transpose(0, 2, 1) / d + 0.2 * np.eye(d)
        cv[3] = np.outer(A[3, 0], A[3, 0])  # rank-1: forces the non-PD fallback
    return x, w, means, cv


@pytest.mark.parametrize("cov", ["diag", "full"])
def test_gmm_estep_and_stats(monkeypatch, cov):
    g = _load("gmm-ema", "gmm")
    x, w, mu, cv = _gmm_problem(cov)
    (lr_c, lp_c), (lr_g, lp_g) = _both(monkeypatch, "gmm",
                                      lambda: g.compute_log_responsibilities(x, w, mu, cv, cov_type=cov, eps_cov=1e-6))
    np.testing.assert_allclose(lp_g, lp_c, rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(np.exp(lr_g), np.exp(lr_c), rtol=1e-7, atol=1e-9)
    gamma = np.exp(lr_c)
    s_c, s_g = _both(monkeypatch, "gmm", lambda: g.compute_batch_stats(x, gamma, cov_type=cov))
    np.testing.assert_allclose(s_g.counts, s_c.counts, rtol=1e-10)
    np.testing.assert_allclose(s_g.sums, s_c.sums, rtol=1e-10, atol=1e-8)
    key = "sq_sums" if cov == "diag" else "cov_sums"
    np.testing.assert_allclose(s_g.extras[key], s_c.extras[key], rtol=1e-10, atol=1e-6)


@pytest.mark.parametrize("cov", ["diag", "full"])
def test_gmm_em_trajectory(monkeypatch, cov):
    """10 EM iterations from the same init end at the same NLL (errors must not compound)."""
    g = _load("gmm-ema", "gmm")
    from core.stats import EMAStats

    x, *_ = _gmm_problem(cov, n=4000, seed=1)

    def run():
        p = g.init_gmm_parameters(x, 8, cov_type=cov, eps_cov=1e-3, rng=np.random.default_rng(3))
        for _ in range(10):
            gm, _ = g.compute_responsibilities(x, p.weights, p.means, p.covars, cov_type=cov, eps_cov=1e-3)
            bs = g.compute_batch_stats(x, gm, cov_type=cov)
            p = g.m_step_from_stats(EMAStats(bs.counts, bs.sums, bs.extras), cov_type=cov, eps_cov=1e-3)
        return g.evaluate_gmm_nll(x, p), p.means

    (nll_c, m_c), (nll_g, m_g) = _both(monkeypatch, "gmm", run)
    assert abs(nll_g - nll_c) <= 1e-8 * abs(nll_c)
    np.testing.assert_allclose(m_g, m_c, rtol=1e-7, atol=1e-7)


# ------------------------------------------------------------------ dictionary learning (FISTA)

@pytest.mark.parametrize("nn", [False, True])
def test_fista(monkeypatch, nn):
    dl = _load("dictlearn-scaling", "dictlearn")
    rng = np.random.default_rng(0)
    D = rng.normal(size=(48, 64))
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    if nn:
        D = np.abs(D)
    x = rng.normal(size=(1024, 64)) if not nn else np.abs(rng.normal(size=(1024, 64)))
    for max_iter, tol in ((50, 1e-4), (100, 1e-5)):
        a_c, a_g = _both(monkeypatch, "dictlearn",
                         lambda: dl.fista_sparse_code(x, D, 0.1, max_iter=max_iter, tol=tol, non_negative=nn))
        assert a_c.dtype == a_g.dtype == np.float32
        np.testing.assert_allclose(a_g, a_c, rtol=1e-5, atol=1e-6)


# ------------------------------------------------------------------ LDA E-step

def _lda_corpus(n=300, V=400, seed=0):
    lda = _load("lda-scaling", "lda_v2")
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        L = int(rng.integers(5, 60))
        ids = np.unique(rng.zipf(1.3, L) % V)
        rows.append((ids, rng.integers(1, 6, len(ids)).astype(np.float64)))
    indptr = np.concatenate([[0], np.cumsum([len(r[0]) for r in rows])])
    return lda, lda.Corpus(indptr, np.concatenate([r[0] for r in rows]), np.concatenate([r[1] for r in rows]), V)


def test_lda_estep_iterates_to_tol():
    """Regression: with parallel=True the old kernel aliased g/newg and stopped every doc after 2 iterations.
    The numba E-step must match a plain-numpy fixed-point iteration run to the same tolerance."""
    from scipy.special import digamma

    lda, c = _lda_corpus()
    lam = lda.init_lambda(20, c.V, 0) + lda.ETA
    eeb = lda.exp_elog_beta(lam)
    g_nb, _ = lda._e_step_numba(np.arange(c.n), c.indptr, c.ids, c.cts, eeb, 0.05, 50, 1e-3, False)
    for d in range(0, c.n, 37):
        s, e = c.indptr[d], c.indptr[d + 1]
        w, cc = c.ids[s:e], c.cts[s:e]
        g = np.ones(20)
        for _ in range(50):
            eet = np.exp(digamma(g) - digamma(g.sum()))
            newg = 0.05 + eet * (eeb[:, w] @ (cc / (eet @ eeb[:, w] + 1e-100)))
            diff, g = np.abs(newg - g).mean(), newg
            if diff < 1e-3:
                break
        np.testing.assert_allclose(g_nb[d], g, rtol=1e-8)


def test_lda_estep(monkeypatch):
    lda, c = _lda_corpus()
    lam = lda.init_lambda(20, c.V, 0) + lda.ETA
    eeb = lda.exp_elog_beta(lam)
    docs = np.arange(c.n)
    (g_c, s_c), (g_g, s_g) = _both(monkeypatch, "lda",
                                   lambda: lda.e_step(docs, c.indptr, c.ids, c.cts, eeb, 0.05, 50, 1e-3, True))
    np.testing.assert_allclose(g_g, g_c, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(s_g, s_c, rtol=1e-7, atol=1e-12)
    # chunked path (many token chunks) gives the same answer
    g_k, s_k = lda._e_step_torch(docs[::-1].copy(), c.indptr, c.ids, c.cts, eeb, 0.05, 50, 1e-3, True, max_tokens=97)
    np.testing.assert_allclose(g_k, g_c[::-1], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(s_k, s_c, rtol=1e-7, atol=1e-12)


def test_lda_batch_vb_and_perplexity(monkeypatch):
    lda, c = _lda_corpus(seed=1)
    obs, held = c.completion_split(0)

    def run():
        lam = lda.init_lambda(10, c.V, 2) + lda.ETA
        for _ in range(5):
            lam = lda.batch_vb_step(c, lam, 0.1)
        return lda.perplexity(lam, obs, held, 0.1)

    p_c, p_g = _both(monkeypatch, "lda", run)
    assert abs(p_g - p_c) <= 1e-8 * p_c


# ------------------------------------------------------------------ HMM forward-backward

def _hmm_problem(n=500, L=40, S=6, V=25, seed=0):
    hmm = _load("hmm-ema", "hmm_v2")
    rng = np.random.default_rng(seed)
    seqs = -np.ones((n, L), np.int64)
    for i in range(n):
        T = 0 if i % 97 == 5 else int(rng.integers(1, L + 1))  # includes empty and length-1 sequences
        seqs[i, :T] = rng.integers(0, V, T)
    return hmm, seqs, hmm.random_init(S, V, seed)


def test_hmm_forward_backward(monkeypatch):
    hmm, seqs, (lp, A, B) = _hmm_problem()
    cpu, gpu = _both(monkeypatch, "hmm", lambda: hmm._fb_batch(seqs, lp, A, B))
    for a, b, name in zip(gpu, cpu, ("occ", "emis", "trans", "init", "ll")):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10, err_msg=name)
    # the torch forward pass (kept, though _ll_batch routes to numba) against the numba likelihood
    ll_c, tok_c = hmm._ll_batch_numba(seqs, lp, A, B)
    ll_g, tok_g = hmm._fb_torch(seqs, lp, A, B, want_stats=False)
    np.testing.assert_allclose(ll_g, ll_c, rtol=1e-11)
    assert tok_g == tok_c


def test_hmm_em_trajectory(monkeypatch):
    hmm, seqs, init = _hmm_problem(seed=1)

    def run():
        lp, A, B = init
        for _ in range(8):
            lp, A, B = hmm.em_step(seqs, lp, A, B, chunk=128)
        return hmm.nll_per_token(seqs, lp, A, B), B

    (n_c, B_c), (n_g, B_g) = _both(monkeypatch, "hmm", run)
    assert abs(n_g - n_c) <= 1e-10 * abs(n_c)
    np.testing.assert_allclose(B_g, B_c, rtol=1e-8, atol=1e-12)


# ------------------------------------------------------------------ ALS

def _als_problem(nu=400, ni=150, f=8, seed=0):
    als = _load("als-ema", "als_v2")
    rng = np.random.default_rng(seed)
    nnz = 6000
    u = rng.integers(0, nu - 5, nnz)  # last 5 users have no ratings
    i = rng.integers(0, ni - 3, nnz)  # last 3 items have no ratings
    r = rng.normal(size=nnz)
    return als, als.CSR(u, i, r, nu), als.CSR(i, u, r, ni), *als.init_factors(nu, ni, f, seed)


def test_als_kernels(monkeypatch):
    als, Ru, Ri, U, V = _als_problem()
    rows = np.arange(Ru.n_rows)
    x_c, x_g = _both(monkeypatch, "als", lambda: als.solve_rows(rows, Ru.indptr, Ru.indices, Ru.data, V, 0.05))
    np.testing.assert_allclose(x_g, x_c, rtol=1e-9, atol=1e-12)
    users = np.random.default_rng(1).choice(Ru.n_rows, 120, replace=False)
    (c_c, b_c, A_c), (c_g, b_g, A_g) = _both(
        monkeypatch, "als", lambda: als.item_batch_stats(users, U[users], Ru.indptr, Ru.indices, Ru.data, Ri.n_rows))
    np.testing.assert_allclose(c_g, c_c)
    np.testing.assert_allclose(b_g, b_c, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(A_g, A_c, rtol=1e-10, atol=1e-13)
    v_c = als.solve_from_stats(c_c, b_c, A_c, 0.05, V)
    v_g = als._solve_from_stats_torch(c_c, b_c, A_c, 0.05, V)  # kept, not routed (no gain on the 5090)
    np.testing.assert_allclose(v_g, v_c, rtol=1e-9, atol=1e-12)


def test_als_full_epochs(monkeypatch):
    als, Ru, Ri, U0, V0 = _als_problem(seed=2)
    trip = (np.repeat(np.arange(Ru.n_rows), np.diff(Ru.indptr)), Ru.indices, Ru.data)

    def run():
        U, V = U0, V0
        for _ in range(5):
            U, V = als.als_full_epoch(Ru, Ri, U, V, 0.05)
        return als.rmse(trip, U, V)

    r_c, r_g = _both(monkeypatch, "als", run)
    assert abs(r_g - r_c) <= 1e-9 * r_c
