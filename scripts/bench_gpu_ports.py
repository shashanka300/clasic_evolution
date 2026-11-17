#!/usr/bin/env python3
"""CPU vs CUDA-port timing of each hot loop on real data shapes. Usage: bench_gpu_ports.py [gmm dictlearn lda hmm als]

CPU side runs with whatever OMP/NUMBA thread count the environment gives (set NT to match a queue job).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def load(track, mod):
    tdir = ROOT / "tracks" / track
    sys.path.insert(0, str(tdir))
    spec = importlib.util.spec_from_file_location(mod, tdir / f"{mod}.py")  # bare name: numba cache pickles it
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def timeit(fn, reps=3):
    import torch
    fn()  # warm-up / JIT
    torch.cuda.synchronize()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t.append(time.perf_counter() - t0)
    return min(t)


def both(name, fn):
    os.environ.pop("CLASSIC_ALGOS_GPU_PORTS", None)
    tc = timeit(fn)
    os.environ["CLASSIC_ALGOS_GPU_PORTS"] = name
    tg = timeit(fn)
    os.environ.pop("CLASSIC_ALGOS_GPU_PORTS", None)
    return tc, tg


def report(label, tc, tg):
    print(f"{label:<48} cpu {tc * 1e3:9.1f} ms   gpu {tg * 1e3:9.1f} ms   x{tc / tg:6.1f}", flush=True)


def bench_gmm():
    g = load("gmm-ema", "gmm")
    rng = np.random.default_rng(0)
    for cov, k, d, b in (("diag", 64, 64, 8192), ("diag", 32, 64, 256), ("full", 32, 64, 8192), ("full", 64, 64, 2048), ("full", 32, 64, 256)):
        x = rng.normal(size=(b, d))
        p = g.init_gmm_parameters(x, k, cov_type=cov, eps_cov=1e-3, rng=np.random.default_rng(0))

        def step():
            gm, _ = g.compute_responsibilities(x, p.weights, p.means, p.covars, cov_type=cov, eps_cov=1e-3)
            g.compute_batch_stats(x, gm, cov_type=cov)
        report(f"gmm {cov} K={k} d={d} B={b} (E-step+stats)", *both("gmm", step))


def bench_dictlearn():
    dl = load("dictlearn-scaling", "dictlearn")
    rng = np.random.default_rng(0)
    for k, d, b, it in ((128, 64, 4096, 100), (192, 144, 4096, 100), (128, 64, 256, 50), (128, 64, 1024, 50)):
        D = rng.normal(size=(k, d)); D /= np.linalg.norm(D, axis=1, keepdims=True)
        x = rng.normal(size=(b, d))
        report(f"fista K={k} d={d} B={b} iters<={it}",
               *both("dictlearn", lambda: dl.fista_sparse_code(x, D, 0.1, max_iter=it, tol=1e-5)))


def bench_lda():
    lda = load("lda-scaling", "lda_v2")
    for name, K in (("20ng", 20), ("nytimes", 100)):
        from core.bench import data_root
        z = np.load(data_root() / "processed" / f"lda_{name}.npz")
        c = lda.Corpus(z["indptr"], z["ids"], z["cts"], int(z["V"]))
        lam = lda.init_lambda(K, c.V, 0) + lda.ETA
        eeb = lda.exp_elog_beta(lam)
        for nd in (256, 4096):
            docs = np.random.default_rng(0).choice(c.n, nd, replace=False)
            report(f"lda e-step {name} K={K} docs={nd}",
                   *both("lda", lambda: lda.e_step(docs, c.indptr, c.ids, c.cts, eeb, 1.0 / K, 50, 1e-3, True)))


def bench_hmm():
    hmm = load("hmm-ema", "hmm_v2")
    rng = np.random.default_rng(0)
    for n, L, S, V in ((4096, 80, 12, 5000), (5000, 128, 12, 64), (1024, 80, 12, 5000), (64, 80, 12, 5000)):
        seqs = -np.ones((n, L), np.int64)
        for i in range(n):
            T = int(rng.integers(L // 4, L + 1))
            seqs[i, :T] = rng.integers(0, V, T)
        lp, A, B = hmm.random_init(S, V, 0)
        report(f"hmm fwd-bwd n={n} L={L} S={S} V={V}", *both("hmm", lambda: hmm._fb_batch(seqs, lp, A, B)))
        report(f"hmm loglik  n={n} L={L} S={S} V={V}", *both("hmm", lambda: hmm._ll_batch(seqs, lp, A, B)))


def bench_als():
    from core.bench import data_root
    als = load("als-ema", "als_v2")
    z = np.load(data_root() / "processed" / "ml10m_split.npz")
    u, i, r = z["u"], z["i"], z["r"] - z["r"].mean()
    nu, ni = int(u.max()) + 1, int(i.max()) + 1
    Ru, Ri = als.CSR(u, i, r, nu), als.CSR(i, u, r, ni)
    for f in (16, 32):
        U, V = als.init_factors(nu, ni, f, 0)
        report(f"als solve all users ml10m f={f} ({nu})",
               *both("als", lambda: als.solve_rows(np.arange(nu), Ru.indptr, Ru.indices, Ru.data, V, 0.05)))
        users = np.random.default_rng(0).choice(nu, 2048, replace=False)
        report(f"als batch 2048 users: solve + item stats f={f}", *both("als", lambda: (
            als.solve_rows(users, Ru.indptr, Ru.indices, Ru.data, V, 0.05),
            als.item_batch_stats(users, U[users], Ru.indptr, Ru.indices, Ru.data, ni))))
        cnt, b, A = als.item_batch_stats(users, U[users], Ru.indptr, Ru.indices, Ru.data, ni)
        report(f"als solve_from_stats {ni} items f={f}", *both("als", lambda: als.solve_from_stats(cnt, b, A, 0.05, V)))


BENCH = {"gmm": bench_gmm, "dictlearn": bench_dictlearn, "lda": bench_lda, "hmm": bench_hmm, "als": bench_als}

if __name__ == "__main__":
    print(f"threads: OMP={os.environ.get('OMP_NUM_THREADS')} NUMBA={os.environ.get('NUMBA_NUM_THREADS')}")
    for n in sys.argv[1:] or list(BENCH):
        BENCH[n]()
