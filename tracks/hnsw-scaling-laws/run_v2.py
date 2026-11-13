"""T11 hnsw-scaling-laws, v2 protocol. JSONL only; verdicts from scripts/evaluate.py.

Grid (every dataset): M in {8,12,16,24,32,48} x efC in {50,100,200,400}; for every built index,
efSearch in EFS_GRID. QPS = single-thread batch query throughput over 2000 queries (hnswlib),
build with 8 threads. Run on an idle machine (queue behind scripts/wait_idle.sh).

For target recall r in {0.9, 0.95, 0.99} the grid oracle = max-QPS (M, efC, efS) with recall@10 >= r.
Law: log M*, log efC*, log efS* each = c + a log n + b log d + g log LID + h log(-log(1-r)), fitted
on CALIBRATION datasets only (OLS over the oracle points). LID = Levina-Bickel MLE (k=20).
Held-out datasets (never used in the fit): agnews_minilm, glove100_300k.

Records on held-out datasets (extra.held_out = True), one per (method, target):
  law                   predicted (M, efC, efS)
  default_hnswlib       M=16, efC=200, efS=10 (hnswlib defaults)
  default_faiss         M=32, efC=40, efS=16 (Faiss IndexHNSWFlat defaults), built with hnswlib
  default_hnswlib_efs   hnswlib default build + the smallest efS in the grid reaching r (oracle-tuned ef)
  grid_oracle           best grid point (frontier); law's QPS / oracle QPS = distance to front
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import faiss
import h5py
import hnswlib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.bench import load_dinov2_cifar100, load_sift1m, load_text_embeddings  # noqa: E402
from core.harness import make_run, read_runs, results_path, write_run  # noqa: E402

TRACK = "hnsw-scaling-laws"
RUNS = results_path(TRACK)
GRIDF = results_path(TRACK, "grid.jsonl")
MS, EFCS = (8, 12, 16, 24, 32, 48), (50, 100, 200, 400)
EFS_GRID = (10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
TARGETS = (0.9, 0.95, 0.99)
CALIB = ("synth_n50k_d32", "synth_n200k_d32", "synth_n50k_d128", "synth_n200k_d128", "sift_100k", "sift_300k", "dinov2_50k")
HELD = ("agnews_minilm", "glove100_300k")
NQ = 2000


def _gt(xb, xq):
    idx = faiss.IndexFlatL2(xb.shape[1])
    idx.add(xb)
    return idx.search(xq, 10)[1]


def load(name):
    rng = np.random.default_rng(0)
    if name.startswith("synth"):
        n = int(name.split("_n")[1].split("k")[0]) * 1000
        d = int(name.split("_d")[1])
        C = rng.normal(0, 1, (100, d)).astype(np.float32)
        def s(m):
            return (C[rng.integers(0, 100, m)] + 0.5 * rng.standard_normal((m, d))).astype(np.float32)
        xb, xq = s(n), s(NQ)
    elif name.startswith("sift"):
        sp = load_sift1m(val_size=0)
        n = int(name.split("_")[1][:-1]) * 1000
        xb = np.ascontiguousarray(sp.train[rng.permutation(1_000_000)[:n]], np.float32)
        xq = np.ascontiguousarray(sp.test[:NQ], np.float32)
    elif name == "dinov2_50k":
        sp = load_dinov2_cifar100(val_size=0)
        xb, xq = np.ascontiguousarray(sp.train, np.float32), np.ascontiguousarray(sp.test[:NQ], np.float32)
    elif name == "agnews_minilm":
        sp = load_text_embeddings(dataset_name="ag_news", max_train=100_000, max_test=7_600, val_size=0)
        xb, xq = np.ascontiguousarray(sp.train, np.float32), np.ascontiguousarray(sp.test[:NQ], np.float32)
    elif name == "glove100_300k":
        with h5py.File(ROOT / "data" / "raw" / "glove-100-angular.hdf5") as f:
            xb = np.asarray(f["train"][:300_000], np.float32)
            xq = np.asarray(f["test"][:NQ], np.float32)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True) + 1e-12
        xq /= np.linalg.norm(xq, axis=1, keepdims=True) + 1e-12
    else:
        raise ValueError(name)
    return xb, xq, _gt(xb, xq)


def lid_mle(xb, k=20, m=2000):
    rng = np.random.default_rng(0)
    q = xb[rng.choice(len(xb), m, replace=False)]
    idx = faiss.IndexFlatL2(xb.shape[1])
    idx.add(xb)
    D, _ = idx.search(q, k + 1)
    r = np.sqrt(np.maximum(D[:, 1:], 1e-20))
    return float(np.mean(1.0 / np.mean(np.log(r[:, -1:] / r[:, :-1]), axis=1)))


def build(xb, M, efc):
    idx = hnswlib.Index(space="l2", dim=xb.shape[1])
    idx.init_index(max_elements=len(xb), ef_construction=efc, M=M, random_seed=0)
    idx.set_num_threads(8)
    t0 = time.perf_counter()
    idx.add_items(xb, np.arange(len(xb)))
    return idx, time.perf_counter() - t0


def query(idx, xq, gt, efs):
    idx.set_num_threads(1)
    idx.set_ef(max(efs, 10))
    best = np.inf
    for _ in range(2):
        t0 = time.perf_counter()
        lab, _ = idx.knn_query(xq, k=10)
        best = min(best, time.perf_counter() - t0)
    rec = float(np.mean([len(set(lab[i]) & set(gt[i])) / 10 for i in range(len(xq))]))
    return rec, len(xq) / best


def grid(name, xb, xq, gt, lid):
    done = {(r["dataset"], r["M"], r["efc"], r["efs"]) for r in _read_grid()}
    for M, efc in itertools.product(MS, EFCS):
        if all((name, M, efc, e) in done for e in EFS_GRID):
            continue
        idx, bt = build(xb, M, efc)
        with open(GRIDF, "a") as f:
            for efs in EFS_GRID:
                rec, qps = query(idx, xq, gt, efs)
                f.write(json.dumps({"dataset": name, "n": len(xb), "d": xb.shape[1], "lid": lid, "M": M, "efc": efc,
                                    "efs": efs, "recall10": rec, "qps": qps, "build_s": bt}) + "\n")
        print(f"[grid] {name} M={M} efC={efc} build={bt:.1f}s", flush=True)


def _read_grid():
    if not GRIDF.exists():
        return []
    return [json.loads(l) for l in open(GRIDF) if l.strip()]


def oracle(rows, r):
    ok = [x for x in rows if x["recall10"] >= r]
    return max(ok, key=lambda x: x["qps"]) if ok else None


def fit_law(rows_all):
    feats, ys = [], {"M": [], "efc": [], "efs": []}
    for ds in CALIB:
        rows = [x for x in rows_all if x["dataset"] == ds]
        if not rows:
            continue
        for r in TARGETS:
            o = oracle(rows, r)
            if o is None:
                continue
            feats.append([1.0, math.log(o["n"]), math.log(o["d"]), math.log(o["lid"]), math.log(-math.log(1 - r))])
            for k in ys:
                ys[k].append(math.log(o[k]))
    X = np.array(feats)
    coef = {k: np.linalg.lstsq(X, np.array(v), rcond=None)[0] for k, v in ys.items()}
    return coef, len(feats)


def predict(coef, n, d, lid, r):
    f = np.array([1.0, math.log(n), math.log(d), math.log(lid), math.log(-math.log(1 - r))])
    M = int(np.clip(round(math.exp(coef["M"] @ f)), 4, 96))
    efc = int(np.clip(round(math.exp(coef["efc"] @ f)), 16, 800))
    efs = int(np.clip(round(math.exp(coef["efs"] @ f)), 10, 1024))
    return M, efc, efs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("all", "grid", "eval"))
    args = ap.parse_args()
    lids = {}
    if args.stage in ("all", "grid"):
        for name in CALIB + HELD:
            xb, xq, gt = load(name)
            lids[name] = lid_mle(xb)
            grid(name, xb, xq, gt, lids[name])
    if args.stage in ("all", "eval"):
        rows_all = _read_grid()
        coef, npts = fit_law(rows_all)
        have = {(r["method"], r["extra"]["group"]) for r in read_runs(RUNS)}
        for name in HELD:
            xb, xq, gt = load(name)
            lid = next(x["lid"] for x in rows_all if x["dataset"] == name)
            rows = [x for x in rows_all if x["dataset"] == name]
            cache = {}
            for r in TARGETS:
                group = f"{name}|R={r}"
                o = oracle(rows, r)
                M, efc, efs = predict(coef, len(xb), xb.shape[1], lid, r)
                cands = {"law": (M, efc, efs), "default_hnswlib": (16, 200, 10), "default_faiss": (32, 40, 16)}
                dflt = [x for x in rows if x["M"] == 16 and x["efc"] == 200 and x["recall10"] >= r]
                if dflt:
                    e = min(dflt, key=lambda x: x["efs"])
                    cands["default_hnswlib_efs"] = (16, 200, e["efs"])
                if o:
                    cands["grid_oracle"] = (o["M"], o["efc"], o["efs"])
                for meth, (M_, efc_, efs_) in cands.items():
                    if (meth, group) in have:
                        continue
                    if (M_, efc_) not in cache:
                        cache[(M_, efc_)] = build(xb, M_, efc_)
                    idx, bt = cache[(M_, efc_)]
                    rec, qps = query(idx, xq, gt, efs_)
                    frac = qps / o["qps"] if (o and rec >= r) else None
                    write_run(RUNS, make_run(track=TRACK, method=meth, dataset=name, n=len(xb), d=xb.shape[1], k_or_units=M_,
                              epochs=0, batch=0, alpha0=None, seed=0, train_loss=None, test_loss=float(1 - rec), wall_s=bt,
                              role="ours" if meth == "law" else "baseline", alpha0_source="law" if meth == "law" else "na",
                              group=group, metric="1-recall10", held_out=True, target_recall=r, recall10=rec, qps=qps,
                              build_s=bt, M=M_, efc=efc_, efs=efs_, lid=lid, qps_frac_of_oracle=frac,
                              law_coef={k: v.tolist() for k, v in coef.items()}, law_points=npts))
                    print(f"[{group}] {meth:20s} M={M_} efC={efc_} efS={efs_} R@10={rec:.4f} QPS={qps:.0f} "
                          f"frac_oracle={frac if frac is None else round(frac, 3)}", flush=True)


if __name__ == "__main__":
    main()
