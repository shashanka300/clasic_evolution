"""T10 adaptive-k, v2 protocol. JSONL only; verdicts from scripts/evaluate.py.

Ours  ``ema_xmeans``: e-means EMA stream (start k=2) where, every SPLIT_EVERY steps, each cluster with
       >= MIN_PTS buffered recent points is tested with the X-means local BIC (Pelleg & Moore 2000,
       spherical Gaussians): split into 2 (2-means on its buffered points) if BIC(2) > BIC(1); and the
       nearest centroid pair is merged if BIC(merged) > BIC(separate). Children enter the EMA stats
       with half the parent's mass (donor-split, as revival does). Total budget: 3 epochs.
Baselines: ``xmeans`` (full-batch X-means; Improve-Params = k-means warm-started from split centres), ``gmeans`` (Hamerly & Elkan 2003,
       Anderson-Darling at alpha=1e-4 on the 1-D projection), ``silhouette_sweep`` (Faiss k-means for
       k in a FIXED grid, pick max silhouette on a 10k subsample).
Data: synthetic Gaussian blobs with known k (3 configs). Metrics: k_selected, |k - k_true|, ARI,
test MSE. test_loss = |k - k_true| (lower is better). NOTE: DP-means and the real-embedding
downstream-IVF evaluation from the plan are not run (documented gap).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from scipy import stats as sps
from sklearn.metrics import adjusted_rand_score, silhouette_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.emeans import assign_nearest, eval_kmeans_mse  # noqa: E402
from core.harness import make_run, read_runs, results_path, write_run  # noqa: E402
from core.revival import revive_dead_units  # noqa: E402
from core.stats import EMAStats  # noqa: E402

TRACK = "adaptive-k"
RUNS = results_path(TRACK)
CONFIGS = {"blobs_k8_d16": (8, 16, 2.5), "blobs_k25_d32": (25, 32, 2.0), "blobs_k50_d32": (50, 32, 1.6),
           "aniso_k30_d32": (30, 32, 1.2)}  # anisotropic per-dimension scales U(0.3, 1.5), overlapping
SPLIT_EVERY, MIN_PTS, MAX_K = 10, 64, 200


def blobs(k, d, sep, seed=0, n=100_000, aniso=False):
    rng = np.random.default_rng(seed)
    C = rng.normal(0, sep, (k, d))
    w = rng.dirichlet(np.full(k, 4.0))
    sd = rng.uniform(0.6, 1.0, k)[:, None] * np.ones((k, d))
    if aniso:
        sd = rng.uniform(0.3, 1.5, (k, d))

    def sample(m):
        z = rng.choice(k, m, p=w)
        return (C[z] + sd[z] * rng.standard_normal((m, d))).astype(np.float32), z
    xtr, ytr = sample(n)
    xte, yte = sample(10_000)
    return xtr, ytr, xte, yte


def bic_spherical(x, labels, centers):
    """X-means BIC (Pelleg & Moore 2000, eq. 2-3): pooled spherical variance."""
    n, d = x.shape
    k = len(centers)
    if n <= k:
        return -np.inf
    var = max(((x - centers[labels]) ** 2).sum() / (d * (n - k)), 1e-12)
    nj = np.bincount(labels, minlength=k)
    nj = nj[nj > 0]
    # plug-in MLE log-likelihood: mixing term + Gaussian term with pooled spherical variance
    ll = float((nj * np.log(nj / n)).sum()) - n * d / 2 * math.log(2 * math.pi * var) - d * (n - k) / 2
    p = (k - 1) + k * d + 1
    return ll - p / 2 * math.log(n)


def two_means(x, seed):
    km = faiss.Kmeans(x.shape[1], 2, niter=10, seed=seed)
    km.train(x)
    return km.centroids, km.index.search(x, 1)[1][:, 0]


def split_better(x, seed):
    c1 = x.mean(0, keepdims=True)
    b1 = bic_spherical(x, np.zeros(len(x), int), c1)
    c2, l2 = two_means(x, seed)
    return bic_spherical(x, l2, c2) > b1, c2


def ema_xmeans(x, seed, epochs=3.0, B=1024, alpha0=0.5):
    rng = np.random.default_rng(seed)
    n, d = x.shape
    C = x[rng.choice(n, 2, replace=False)].astype(np.float64)
    st = EMAStats.from_params(C)
    steps = int(epochs * (n // B))
    buf_x = []
    for t in range(steps):
        xb = x[rng.integers(0, n, B)]
        lab, _ = assign_nearest(xb, C.astype(np.float32))
        k = len(C)
        cnt = np.bincount(lab, minlength=k).astype(float)
        sums = np.zeros((k, d))
        np.add.at(sums, lab, xb.astype(float))
        a = alpha0 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / steps)))
        st.counts = (1 - a) * st.counts + a * cnt
        st.sums = (1 - a) * st.sums + a * sums
        if t % 5 == 4:
            revive_dead_units(st, rng=rng)
        C = st.means(np.float64)
        buf_x.append(xb)
        buf_x = buf_x[-SPLIT_EVERY:]
        if t % SPLIT_EVERY == SPLIT_EVERY - 1 and t < 0.8 * steps:  # structure changes stop for the final 20%
            xx = np.concatenate(buf_x)
            ll, _ = assign_nearest(xx, C.astype(np.float32))
            new_C, new_cnt, new_sum = [], [], []
            for j in range(len(C)):
                pts = xx[ll == j]
                if len(pts) >= MIN_PTS and len(C) + len(new_C) < MAX_K:
                    better, c2 = split_better(pts, seed + t + j)
                    if better:
                        half = st.counts[j] / 2
                        st.counts[j], st.sums[j] = half, half * c2[0]
                        new_C.append(c2[1]); new_cnt.append(half); new_sum.append(half * c2[1])
            if new_C:
                st.counts = np.concatenate([st.counts, new_cnt])
                st.sums = np.concatenate([st.sums, np.array(new_sum)])
            C = st.means(np.float64)
            # merge test on the closest pair
            if len(C) > 2:
                ll, _ = assign_nearest(xx, C.astype(np.float32))
                D = ((C[:, None] - C[None]) ** 2).sum(-1) + np.eye(len(C)) * 1e18
                i, j = np.unravel_index(np.argmin(D), D.shape)
                m = (ll == i) | (ll == j)
                pts = xx[m]
                if len(pts) >= MIN_PTS:
                    sep_lab = (ll[m] == j).astype(int)
                    b_sep = bic_spherical(pts, sep_lab, np.stack([C[i], C[j]]))
                    b_mer = bic_spherical(pts, np.zeros(len(pts), int), pts.mean(0, keepdims=True))
                    if b_mer > b_sep:
                        st.counts[i] += st.counts[j]
                        st.sums[i] += st.sums[j]
                        keep = np.arange(len(C)) != j
                        st.counts, st.sums = st.counts[keep], st.sums[keep]
                        C = st.means(np.float64)
    return C.astype(np.float32)


def kmeans(x, k, seed, niter=20, init=None):
    km = faiss.Kmeans(x.shape[1], k, niter=niter, seed=seed, max_points_per_centroid=10 ** 7)
    km.train(x, init_centroids=None if init is None else np.ascontiguousarray(init, np.float32))
    return km.centroids


def xmeans(x, seed, k0=2, kmax=MAX_K):
    C = kmeans(x, k0, seed)
    while len(C) < kmax:
        lab = assign_nearest(x, C)[0]
        newC, changed = [], False
        for j in range(len(C)):
            pts = x[lab == j]
            if len(pts) >= MIN_PTS:
                better, c2 = split_better(pts, seed + j)
                if better:
                    newC.extend(c2); changed = True
                    continue
            newC.append(C[j])
        if not changed:
            break
        C = kmeans(x, len(newC), seed, init=np.array(newC))  # Improve-Params warm-started from split centres
    return C


def gmeans(x, seed, kmax=MAX_K):
    C = kmeans(x, 1, seed)
    while len(C) < kmax:
        lab = assign_nearest(x, C)[0]
        newC, changed = [], False
        for j in range(len(C)):
            pts = x[lab == j]
            if len(pts) >= 2 * MIN_PTS:
                c2, l2 = two_means(pts, seed + j)
                v = c2[0] - c2[1]
                proj = pts @ v / max(np.dot(v, v), 1e-12)
                ad = sps.anderson((proj - proj.mean()) / max(proj.std(), 1e-12))
                n = len(proj)
                a2 = ad.statistic * (1 + 4 / n - 25 / n ** 2)
                if a2 > 1.8692:  # Hamerly & Elkan (2003): alpha = 1e-4 critical value for the adjusted A*^2
                    newC.extend(c2); changed = True
                    continue
            newC.append(C[j])
        if not changed:
            break
        C = kmeans(x, len(newC), seed, init=np.array(newC))
    return C


def silhouette_sweep(x, seed, k_true):
    rng = np.random.default_rng(seed)
    sub = x[rng.choice(len(x), 10_000, replace=False)]
    grid = [2, 4, 8, 12, 16, 25, 32, 50, 64, 100]  # fixed: must not depend on k_true
    best = max(grid, key=lambda k: silhouette_score(sub, assign_nearest(sub, kmeans(x, k, seed))[0]))
    return kmeans(x, best, seed), grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    have = {(r["method"], r["extra"]["group"], r["seed"]) for r in read_runs(RUNS)}
    for name, (kt, d, sep) in CONFIGS.items():
        xtr, ytr, xte, yte = blobs(kt, d, sep, aniso=name.startswith("aniso"))
        for s in range(args.seeds):
            for method in ("ema_xmeans", "xmeans", "gmeans", "silhouette_sweep"):
                if (method, name, s) in have:
                    continue
                t0 = time.perf_counter()
                extra = {}
                if method == "ema_xmeans":
                    C = ema_xmeans(xtr, s)
                elif method == "xmeans":
                    C = xmeans(xtr, s)
                elif method == "gmeans":
                    C = gmeans(xtr, s)
                else:
                    C, grid = silhouette_sweep(xtr, s, kt)
                    extra["k_grid"] = grid
                w = time.perf_counter() - t0
                ks = len(C)
                ari = adjusted_rand_score(yte, assign_nearest(xte, C)[0])
                write_run(RUNS, make_run(track=TRACK, method=method, dataset=name, n=len(xtr), d=d, k_or_units=kt, epochs=3.0,
                          batch=1024, alpha0=0.5 if method == "ema_xmeans" else None, seed=s, train_loss=None,
                          test_loss=float(abs(ks - kt)), wall_s=w, role="ours" if method == "ema_xmeans" else "baseline",
                          alpha0_source="fixed" if method == "ema_xmeans" else "na", group=name, metric="abs_k_error",
                          k_selected=ks, k_true=kt, ari=ari, test_mse=eval_kmeans_mse(xte, C), **extra))
                print(f"[{name} s={s}] {method:18s} k={ks} (true {kt}) ARI={ari:.3f} ({w:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
