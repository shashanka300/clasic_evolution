"""T07 v2: numba HMM must agree with hmmlearn exactly."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tracks" / "hmm-ema"))

from hmm_v2 import HMMPlugin, em_step, nll_per_token, pad, random_init  # noqa: E402
from core.plugin import run_ema_engine  # noqa: E402
from core.schedule import CosineSchedule  # noqa: E402


def _data(seed=0, S=4, V=12, n=200):
    rng = np.random.default_rng(seed)
    seqs = [rng.integers(0, V, rng.integers(3, 25)) for _ in range(n)]
    return seqs, pad(seqs)


def _hmmlearn(log_pi, A, B, n_iter):
    from hmmlearn.hmm import CategoricalHMM
    m = CategoricalHMM(n_components=A.shape[0], n_iter=n_iter, init_params="", params="ste", tol=-1e9, implementation="scaling")
    m.startprob_, m.transmat_, m.emissionprob_ = np.exp(log_pi), A.copy(), B.copy()
    return m


def test_loglik_matches_hmmlearn():
    seqs, X = _data()
    lp, A, B = random_init(4, 12, 1)
    m = _hmmlearn(lp, A, B, 1)
    cat = np.concatenate(seqs)[:, None]
    lens = [len(s) for s in seqs]
    assert np.isclose(-m.score(cat, lens) / len(cat), nll_per_token(X, lp, A, B), rtol=1e-9)


def test_em_step_matches_hmmlearn():
    seqs, X = _data()
    lp, A, B = random_init(4, 12, 2)
    m = _hmmlearn(lp, A, B, 1)
    m.fit(np.concatenate(seqs)[:, None], [len(s) for s in seqs])
    lp2, A2, B2 = em_step(X, lp, A, B)
    assert np.allclose(A2, m.transmat_, atol=1e-5)
    assert np.allclose(B2, m.emissionprob_, atol=1e-5)
    assert np.allclose(np.exp(lp2), m.startprob_, atol=1e-5)


def test_engine_runs_and_improves():
    _, X = _data(n=400)
    plug = HMMPlugin(4, 12, seed=0)
    plug.init(X)
    before = plug.loss(X)
    run_ema_engine(plug, X, epochs=2, batch_size=32, schedule=CosineSchedule(0.3, 25), finalize_mode="none")
    assert plug.loss(X) < before


def test_revival_gives_incoming_transitions():
    _, X = _data(n=50)
    plug = HMMPlugin(4, 12, seed=0)
    st = plug.init(X)
    st.counts[:] = [10, 10, 10, 1e-9]
    st.extras["trans"][:, 3] = 0.0
    ev = plug.revive(st, 0, np.random.default_rng(0))
    assert list(ev.dead_indices) == [3]
    assert st.extras["trans"][:, 3].sum() > 0


def test_init_params_survive_first_mstep():
    """Regression: with n0=1e-10 the first params_from_stats must return the random init, not uniform."""
    _, X = _data()
    plug = HMMPlugin(4, 12, seed=3)
    st = plug.init(X)
    B0, A0 = plug.B.copy(), plug.A.copy()
    plug.params_from_stats(st)
    assert np.allclose(plug.B, B0, rtol=1e-4) and np.allclose(plug.A, A0, rtol=1e-4)
