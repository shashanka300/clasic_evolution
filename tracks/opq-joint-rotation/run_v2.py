"""T02 opq-joint-rotation, v2 protocol. JSONL only; verdicts from scripts/evaluate.py.

Equal passes: every EMA method gets exactly T passes over the training set.
  emeans_pq               [ablation]  e-means PQ codebooks, R = I, T passes
  emeans_pq_posthoc_opq   [baseline]  (T - 0.5) passes PQ, one Procrustes refit of R on 50k points,
                                      0.5 pass codebook refit in the new frame   <- kill comparator
  joint_rotation          [ours]      EMA codebooks + EMA cross-covariance H, Procrustes every m=10 batches, T passes
  joint_rotation_m1/m100  [ablation]  refresh period 1 / 100
  faiss_pq / faiss_opq    [baseline]  Faiss defaults on the same training set (not pass-matched; library reference)
Evaluation: test reconstruction MSE (sum over dims), and ADC recall@10 by loading the codebooks into a
real ``faiss.IndexPQ`` over the full database (rotated vectors), queries rotated with the same R.
Codebook alpha0 / B from the e-means law for k=256 per subspace (alpha0_source = law).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.bench import load_dinov2_cifar100, load_sift1m, load_text_embeddings  # noqa: E402
from core.harness import make_run, read_runs, results_path, write_run  # noqa: E402
from core.scaling import emeans_alpha0, emeans_batch_size  # noqa: E402
from core.schedule import CosineSchedule  # noqa: E402
from opq import OPQQuantizer, solve_orthogonal_procrustes  # noqa: E402

TRACK = "opq-joint-rotation"
RUNS = results_path(TRACK)
BUDGETS = (1.0, 2.0, 4.0, 8.0)
KSUB = 256


def load(name):
    if name == "sift1m":
        s = load_sift1m(val_size=0)
        base = np.ascontiguousarray(s.train, np.float32)
        q = np.ascontiguousarray(s.test[:2000], np.float32)
        gt = np.asarray(s.groundtruth[:2000])
        tr = base[np.random.default_rng(0).permutation(len(base))[:200_000]]
        test = base[np.random.default_rng(1).permutation(len(base))[:20_000]]
        return tr, test, base, q, gt
    s = (load_text_embeddings(dataset_name="ag_news", max_train=100_000, max_test=7_600, val_size=0)
         if name == "agnews_minilm" else load_dinov2_cifar100(val_size=0))
    base = np.ascontiguousarray(s.train, np.float32)
    p = np.random.default_rng(0).permutation(len(base))
    tr, test = base[p[10_000:]], base[p[:10_000]]
    q = np.ascontiguousarray(s.test[:2000], np.float32)
    flat = faiss.IndexFlatL2(base.shape[1])
    flat.add(base)
    return tr, test, base, q, flat.search(q, 10)[1]


def _schedule(n, T, B):
    a0 = emeans_alpha0(n, KSUB, T, B)
    return CosineSchedule(a0, max(int(round(T * (n // B))), 1), eps=0.1), a0


def fit(method, x, M, T, seed):
    n, d = x.shape
    B = emeans_batch_size(n, KSUB, T)
    if method in ("emeans_pq", "joint_rotation", "joint_rotation_m1", "joint_rotation_m100"):
        m = {"joint_rotation": 10, "joint_rotation_m1": 1, "joint_rotation_m100": 100}.get(method, 10)
        q = OPQQuantizer(d, M, KSUB, alpha_r=-1.0, refresh_m=m, seed=seed)  # alpha_r<0: rotation EMA tied to codebook step
        sched, a0 = _schedule(n, T, B)
        q.fit(x, epochs=T, batch_size=B, schedule=sched, update_rotation=(method != "emeans_pq"), seed=seed)
        return q.R, q.codebooks, a0, B
    if method == "emeans_pq_posthoc_opq":
        q = OPQQuantizer(d, M, KSUB, alpha_r=0.0, refresh_m=10 ** 9, seed=seed)
        T1 = T - 0.5
        sched, a0 = _schedule(n, T1, B)
        q.fit(x, epochs=T1, batch_size=B, schedule=sched, update_rotation=False, seed=seed)
        xs = x[:50_000]
        yhat = _decode(q.codebooks, _encode(xs @ q.R.T, q.codebooks))
        q.R = solve_orthogonal_procrustes((xs.T @ yhat).astype(np.float64) / len(xs)).astype(np.float32)
        sched2, _ = _schedule(n, 0.5, B)
        q.fit(x, epochs=0.5, batch_size=B, schedule=sched2, update_rotation=False, seed=seed + 1)
        return q.R, q.codebooks, a0, B
    raise ValueError(method)


def _encode(y, cb):
    M, K, ds = cb.shape
    codes = np.empty((len(y), M), np.int64)
    for m in range(M):
        idx = faiss.IndexFlatL2(ds)
        idx.add(np.ascontiguousarray(cb[m]))
        codes[:, m] = idx.search(np.ascontiguousarray(y[:, m * ds:(m + 1) * ds]), 1)[1][:, 0]
    return codes


def _decode(cb, codes):
    return np.concatenate([cb[m][codes[:, m]] for m in range(cb.shape[0])], 1)


def pq_index(R, cb):
    M, K, ds = cb.shape
    d = M * ds
    pq = faiss.IndexPQ(d, M, 8)
    faiss.copy_array_to_vector(np.ascontiguousarray(cb, np.float32).ravel(), pq.pq.centroids)
    pq.is_trained = True
    return pq


def evaluate(R, cb, test, base, q, gt):
    y = test @ R.T
    mse = float(((y - _decode(cb, _encode(y, cb))) ** 2).sum(1).mean())
    idx = pq_index(R, cb)
    idx.add(np.ascontiguousarray(base @ R.T, np.float32))
    _, I = idx.search(np.ascontiguousarray(q @ R.T, np.float32), 10)
    rec = float(np.mean([len(set(I[i]) & set(gt[i][:10])) / 10 for i in range(len(q))]))
    return mse, rec


def faiss_baseline(kind, x, M, test, base, q, gt, seed):
    d = x.shape[1]
    idx = faiss.index_factory(d, f"OPQ{M},PQ{M}" if kind == "faiss_opq" else f"PQ{M}")
    faiss.ParameterSpace().set_index_parameter(idx, "verbose", 0)
    t0 = time.perf_counter()
    idx.train(x)
    w = time.perf_counter() - t0
    rec_x = idx.sa_decode(idx.sa_encode(test))
    mse = float(((test - rec_x) ** 2).sum(1).mean())
    idx.add(base)
    _, I = idx.search(q, 10)
    rec = float(np.mean([len(set(I[i]) & set(gt[i][:10])) / 10 for i in range(len(q))]))
    return mse, rec, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="sift1m:8,sift1m:16,agnews_minilm:8,agnews_minilm:16,dinov2:8,dinov2:16")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    faiss.omp_set_num_threads(6)
    have = {(r["method"], r["extra"]["group"], r["seed"]) for r in read_runs(RUNS)}
    cache = {}
    for spec in args.configs.split(","):
        name, M = spec.split(":")
        M = int(M)
        if name not in cache:
            cache = {name: load(name)}
        tr, test, base, q, gt = cache[name]
        n, d = tr.shape
        group = f"{name}|M={M}"
        for s in range(args.seeds):
            for method in ("emeans_pq", "emeans_pq_posthoc_opq", "joint_rotation", "joint_rotation_m1", "joint_rotation_m100"):
                if (method, group, s) in have:
                    continue
                curve, a0, B = [], None, None
                for T in BUDGETS:
                    t0 = time.perf_counter()
                    R, cb, a0, B = fit(method, tr, M, T, s)
                    w = time.perf_counter() - t0
                    mse, rec = evaluate(R, cb, test, base, q, gt)
                    orth = float(np.linalg.norm(R.T @ R - np.eye(d)))
                    curve.append({"epoch": T, "test_loss": mse, "wall_s": w, "recall10": rec, "orth_err": orth, "alpha0": a0, "B": B})
                    print(f"[{group} s={s}] {method:22s} T={T} mse={mse:.5g} R@10={rec:.4f} ({w:.0f}s)", flush=True)
                role = {"joint_rotation": "ours", "emeans_pq_posthoc_opq": "baseline"}.get(method, "ablation")
                write_run(RUNS, make_run(track=TRACK, method=method, dataset=name, n=n, d=d, k_or_units=M, epochs=BUDGETS[-1],
                          batch=B, alpha0=a0, seed=s, train_loss=None, test_loss=curve[-1]["test_loss"], wall_s=curve[-1]["wall_s"],
                          role=role, alpha0_source="law", group=group, metric="test_recon_mse", curve=curve,
                          curve_kind="budget", recall10=curve[-1]["recall10"], ksub=KSUB))
            for kind in ("faiss_pq", "faiss_opq"):
                if (kind, group, s) in have:
                    continue
                mse, rec, w = faiss_baseline(kind, tr, M, test, base, q, gt, s)
                write_run(RUNS, make_run(track=TRACK, method=kind, dataset=name, n=n, d=d, k_or_units=M, epochs=25, batch=n,
                          alpha0=None, seed=s, train_loss=None, test_loss=mse, wall_s=w, role="baseline", alpha0_source="na",
                          group=group, metric="test_recon_mse", recall10=rec, note="Faiss index_factory defaults (deterministic: same result every seed)"))
                print(f"[{group} s={s}] {kind:22s} mse={mse:.5g} R@10={rec:.4f} ({w:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
