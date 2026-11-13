"""T03 balanced-ivf, v2 protocol. JSONL only; verdicts from scripts/evaluate.py.

All indexes are real ``faiss.IndexIVFFlat`` objects, filled with ``invlists.add_entries`` so the
list assignment can be controlled. Latency = Faiss-internal per-query time
(``indexIVF_stats.quantization_time + search_time``), one query per call, one OMP thread: no
Python in the timed region. We also log scanned codes per query (``ndis``), which is
hardware-independent.

Methods (same nlist, same database, same queries; 3 seeds):
  faiss_ivf            Faiss k-means, 25 iterations on 256*nlist sampled points (Faiss IVF default sampling); nearest add
  emeans_ivf           e-means (law alpha0/B, T=2) on the full database; nearest add           <- kill comparator
  balanced_ivf         same EMA engine, mini-batch assignment = Sinkhorn (3 iters, eps=0.05 on cost
                       normalised by the batch median distance); nearest add                     <- pre-registered method
  balanced_ivf_priced  balanced_ivf centroids + size-penalised add: per-list prices from 50 rounds of
                       capacity price updates over the top-32 candidate lists (vector-DB style)
Latency runs must be scheduled on an otherwise idle machine (queue them alone).
Kill criterion (frozen): p99 latency at recall@10 = 0.9 improves < 10% for balanced_ivf vs emeans_ivf.
NOTE: Malinen & Franti balanced k-means (Hungarian, O(n^3)) is infeasible at 1M points; not run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.bench import load_dinov2_cifar100, load_sift1m, load_text_embeddings  # noqa: E402
from core.emeans import EMeans, ToyKMeansPlugin, assign_nearest  # noqa: E402
from core.harness import make_run, read_runs, results_path, write_run  # noqa: E402
from core.plugin import run_ema_engine  # noqa: E402
from core.scaling import emeans_alpha0, emeans_batch_size  # noqa: E402
from core.schedule import CosineSchedule  # noqa: E402
from core.stats import BatchStats  # noqa: E402

TRACK = "balanced-ivf"
RUNS = results_path(TRACK)
NPROBES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def sinkhorn_labels(x, C, eps=0.05, iters=3):
    xt = torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(DEV)
    ct = torch.from_numpy(np.ascontiguousarray(C, np.float32)).to(DEV)
    M = (xt * xt).sum(1, keepdim=True) + (ct * ct).sum(1)[None] - 2 * xt @ ct.T
    M = M.clamp_min(0)
    M = M / M.median().clamp_min(1e-12)
    cost = -M / eps
    B, K = cost.shape
    beta = torch.zeros(K, device=DEV)
    for _ in range(iters):
        alpha = -torch.logsumexp(cost + beta[None], 1) - np.log(B)
        beta = -torch.logsumexp(cost + alpha[:, None], 0) - np.log(K)
    return (cost + beta[None]).argmax(1).cpu().numpy()


class SinkhornKMeansPlugin(ToyKMeansPlugin):
    def assign(self, batch):
        labels = sinkhorn_labels(batch, self._centroids)
        k = self.k
        counts = np.bincount(labels, minlength=k).astype(np.float64)
        sums = np.zeros((k, batch.shape[1]))
        np.add.at(sums, labels, batch.astype(np.float64))
        return BatchStats(counts=counts, sums=sums)


def train(method, x, nlist, seed, T=2.0):
    n, d = x.shape
    if method == "faiss_ivf":
        km = faiss.Kmeans(d, nlist, niter=25, seed=seed, max_points_per_centroid=256)
        km.train(x)
        return km.centroids
    if method == "emeans_ivf":
        return EMeans(k=nlist, epochs=T, seed=seed).fit(x).centroids_
    B = emeans_batch_size(n, nlist, T)
    a0 = emeans_alpha0(n, nlist, T, B)
    plug = SinkhornKMeansPlugin(k=nlist, seed=seed)
    steps = max(int(round(T * (n // B))), 1)
    res = run_ema_engine(plug, x, epochs=T, batch_size=B, schedule=CosineSchedule(a0, steps, eps=0.1), seed=seed)
    return res.params


def nearest_labels(x, C):
    return assign_nearest(x, C)[0]


def priced_labels(x, C, rounds=50, cand=32, lr0=0.02):
    """Size-penalised add. Step lr0/sqrt(1+r) fixed after a pilot on DINOv2 seed 0 (lr0=0.5 diverged)."""
    n = len(x)
    nl = len(C)
    q = faiss.IndexFlatL2(C.shape[1])
    q.add(np.ascontiguousarray(C, np.float32))
    D, I = q.search(x, cand)
    D = D / np.median(D[:, 0])
    price = np.zeros(nl)
    target = n / nl
    for _ in range(rounds):
        lab = I[np.arange(n), np.argmin(D + price[I], 1)]
        cnt = np.bincount(lab, minlength=nl)
        price += lr0 / np.sqrt(1 + _) * np.log(np.maximum(cnt, 1) / target)
    return I[np.arange(n), np.argmin(D + price[I], 1)]


def build_index(x, C, labels):
    d = x.shape[1]
    quant = faiss.IndexFlatL2(d)
    quant.add(np.ascontiguousarray(C, np.float32))
    idx = faiss.IndexIVFFlat(quant, d, len(C))
    idx.is_trained = True
    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], np.arange(len(C) + 1))
    il = idx.invlists
    for j in range(len(C)):
        ids = order[bounds[j]:bounds[j + 1]].astype(np.int64)
        if len(ids):
            codes = np.ascontiguousarray(x[ids]).view(np.uint8)
            il.add_entries(j, len(ids), faiss.swig_ptr(ids), faiss.swig_ptr(codes))
    idx.ntotal = len(x)
    return idx


def measure(idx, queries, gt, nq_lat=2000):
    faiss.omp_set_num_threads(1)
    st = faiss.cvar.indexIVF_stats
    out = []
    for nprobe in NPROBES:
        if nprobe > idx.nlist:
            continue
        idx.nprobe = nprobe
        _, I = idx.search(queries, 10)
        rec = float(np.mean([len(set(I[i]) & set(gt[i][:10])) / 10 for i in range(len(queries))]))
        lat, nd = [], []
        for i in range(min(nq_lat, len(queries))):
            best = np.inf
            for _ in range(3):  # min of 3 repeats per query suppresses scheduler noise
                st.reset()
                idx.search(queries[i:i + 1], 10)
                best = min(best, (st.quantization_time + st.search_time) * 1000.0)  # ms -> us
            lat.append(best)
            nd.append(st.ndis)
        out.append({"nprobe": nprobe, "recall10": rec, "p50_us": float(np.percentile(lat, 50)),
                    "p99_us": float(np.percentile(lat, 99)), "mean_us": float(np.mean(lat)),
                    "ndis_p50": float(np.percentile(nd, 50)), "ndis_p99": float(np.percentile(nd, 99))})
    faiss.omp_set_num_threads(8)
    return out


def load(name):
    if name == "sift1m":
        s = load_sift1m(val_size=0)
        return np.ascontiguousarray(s.train, np.float32), np.ascontiguousarray(s.test, np.float32), np.asarray(s.groundtruth)
    if name == "agnews_minilm":
        s = load_text_embeddings(dataset_name="ag_news", max_train=100_000, max_test=7_600, val_size=0)
    else:
        s = load_dinov2_cifar100(val_size=0)
    xb, xq = np.ascontiguousarray(s.train, np.float32), np.ascontiguousarray(s.test, np.float32)
    flat = faiss.IndexFlatL2(xb.shape[1])
    flat.add(xb)
    return xb, xq, flat.search(xq, 10)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="sift1m:1024,sift1m:4096,agnews_minilm:256,agnews_minilm:1024,dinov2:256")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    have = {(r["method"], r["extra"]["group"], r["seed"]) for r in read_runs(RUNS)}
    cache = {}
    for spec in args.configs.split(","):
        name, nl = spec.split(":")
        nl = int(nl)
        if name not in cache:
            cache = {name: load(name)}
        xb, xq, gt = cache[name]
        group = f"{name}|nlist={nl}"
        for s in range(args.seeds):
            cents = {}
            for method in ("faiss_ivf", "emeans_ivf", "balanced_ivf", "balanced_ivf_priced"):
                if (method, group, s) in have:
                    continue
                t0 = time.perf_counter()
                tm = "balanced_ivf" if method == "balanced_ivf_priced" else method
                if tm not in cents:
                    t1 = time.perf_counter()
                    cents[tm] = (train(tm, xb, nl, s), time.perf_counter() - t1)
                C, train_s = cents[tm]
                labels = priced_labels(xb, C) if method == "balanced_ivf_priced" else nearest_labels(xb, C)
                sizes = np.bincount(labels, minlength=nl)
                imb = float((sizes.astype(np.float64) ** 2).sum() * nl / len(xb) ** 2)
                mse = float(((xb - C[labels]) ** 2).sum(1).mean())
                idx = build_index(xb, C, labels)
                lat = measure(idx, xq, gt)
                role = "ours" if method.startswith("balanced") else "baseline"
                write_run(RUNS, make_run(track=TRACK, method=method, dataset=name, n=len(xb), d=xb.shape[1], k_or_units=nl,
                          epochs=2.0 if method != "faiss_ivf" else 25.0, batch=0, alpha0=None, seed=s, train_loss=mse,
                          test_loss=mse, wall_s=time.perf_counter() - t0, role=role,
                          alpha0_source="law" if method != "faiss_ivf" else "na", group=group, metric="db_mse",
                          latency=lat, imbalance=imb, max_list=int(sizes.max()), train_s=train_s,
                          timing="faiss indexIVF_stats per query, 1 OMP thread, 2000 queries"))
                r09 = next((p for p in lat if p["recall10"] >= 0.9), None)
                print(f"[{group} s={s}] {method:20s} imb={imb:.3f} mse={mse:.4g} max_list={sizes.max()} "
                      f"first R>=0.9: nprobe={r09 and r09['nprobe']} p99={r09 and round(r09['p99_us'])}us", flush=True)


if __name__ == "__main__":
    main()
