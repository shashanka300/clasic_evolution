"""T13 drift-balanced-ivf, v2: a live IVF index under streaming inserts + distribution drift.

Combines T03 v2 (median-normalised Sinkhorn assignment) and T08 v2 (alarm-triggered rewarm, real
AG News streams). Each of 300 steps inserts B=512 new vectors (from the current distribution) into a
growing IVF (nlist=256); every EVAL_EVERY steps we measure, on 1000 queries from the current
distribution against exact ground truth over the CURRENT database:
  recall@10 at nprobe=8, p99 scanned codes per query (ndis, hardware-independent latency proxy),
  list imbalance, and cumulative compute (vector-centroid distance evaluations, incl. re-indexing).

Methods:
  static            e-means (T=2) on the first 20 steps; centroids frozen; inserts to nearest list
  live_rewarm       T08 rewarm EMA on the insert stream; every 50 steps re-assign the DB to current centroids
  live_rewarm_bal   live_rewarm with Sinkhorn mini-batch assignment (T03)                       (ours)
  periodic_rebuild  every 50 steps: e-means (T=2) on the whole current DB + full re-assignment
No kill criterion in the plan: descriptive. test_loss = 1 - time-averaged recall@10.
Scenarios: abrupt, gradual (encoder_upgrade is excluded: old DB vectors would need re-embedding).
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.emeans import EMeans, assign_nearest  # noqa: E402
from core.harness import make_run, read_runs, results_path, write_run  # noqa: E402
from core.revival import revive_dead_units  # noqa: E402
from core.stats import EMAStats  # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DRIFT = _load(ROOT / "tracks" / "drift-rewarm" / "run_v2.py", "t08_drift_v2")
BIVF = _load(ROOT / "tracks" / "balanced-ivf" / "run_v2.py", "t03_bivf_v2")

TRACK = "drift-balanced-ivf"
RUNS = results_path(TRACK)
NL, B, STEPS, EVAL_EVERY, REINDEX, NPROBE = 256, 512, 300, 25, 50, 8


class LiveIVF:
    def __init__(self, mode, x0, seed):
        self.mode, self.rng = mode, np.random.default_rng(seed)
        self.C = EMeans(k=NL, epochs=2.0, seed=seed).fit(x0).centroids_.astype(np.float64)
        self.stats = EMAStats.from_params(self.C, n0=1.0)
        self.ph = DRIFT.PageHinkleyRel()
        self.t, self.restart_at, self.alarms = 0, -10 ** 9, []
        self.db, self.lab = [], []
        self.compute = 0

    def alpha(self):
        rel = self.t - self.restart_at
        return DRIFT.cosine(rel, DRIFT.WARM) if rel < DRIFT.WARM else DRIFT.ALPHA0 * DRIFT.EPS

    def insert(self, x):
        lab, d2 = assign_nearest(x, self.C.astype(np.float32))
        self.compute += len(x) * NL
        self.db.append(x)
        self.lab.append(lab)
        if self.mode.startswith("live"):
            if self.ph.update(float(d2.mean())):
                self.restart_at = self.t
                self.alarms.append(self.t)
            tl = BIVF.sinkhorn_labels(x, self.C.astype(np.float32)) if self.mode == "live_rewarm_bal" else lab
            cnt = np.bincount(tl, minlength=NL).astype(float)
            sums = np.zeros((NL, x.shape[1]))
            np.add.at(sums, tl, x.astype(float))
            a = self.alpha()
            self.stats.counts = (1 - a) * self.stats.counts + a * cnt
            self.stats.sums = (1 - a) * self.stats.sums + a * sums
            if self.t % 5 == 4:
                revive_dead_units(self.stats, rng=self.rng)
            self.C = self.stats.means(np.float64)
        if self.t % REINDEX == REINDEX - 1 and self.mode != "static":
            X = np.concatenate(self.db)
            if self.mode == "periodic_rebuild":
                self.C = EMeans(k=NL, epochs=2.0, seed=int(self.rng.integers(1 << 30))).fit(X).centroids_.astype(np.float64)
                self.stats = EMAStats.from_params(self.C, n0=1.0)
                self.compute += 2 * len(X) * NL
            self.db, self.lab = [X], [assign_nearest(X, self.C.astype(np.float32))[0]]
            self.compute += len(X) * NL
        self.t += 1

    def evaluate(self, q):
        X = np.concatenate(self.db)
        lab = np.concatenate(self.lab)
        flat = faiss.IndexFlatL2(X.shape[1])
        flat.add(X)
        gt = flat.search(q, 10)[1]
        idx = BIVF.build_index(X, self.C.astype(np.float32), lab)
        idx.nprobe = NPROBE
        _, I = idx.search(q, 10)
        rec = float(np.mean([len(set(I[i]) & set(gt[i])) / 10 for i in range(len(q))]))
        sizes = np.bincount(lab, minlength=NL)
        qlab = faiss.IndexFlatL2(X.shape[1])
        qlab.add(self.C.astype(np.float32))
        probes = qlab.search(q, NPROBE)[1]
        ndis = sizes[probes].sum(1)
        return {"recall10": rec, "ndis_p99": float(np.percentile(ndis, 99)), "ndis_p50": float(np.percentile(ndis, 50)),
                "imbalance": float((sizes.astype(float) ** 2).sum() * NL / len(X) ** 2), "db": int(len(X))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="abrupt,gradual")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    S = DRIFT.Streams()
    have = {(r["method"], r["extra"]["group"], r["seed"]) for r in read_runs(RUNS)}
    for scen in args.scenarios.split(","):
        for seed in range(args.seeds):
            for mode in ("static", "live_rewarm", "live_rewarm_bal", "periodic_rebuild"):
                if (mode, scen, seed) in have:
                    continue
                rng = np.random.default_rng(seed)
                x0 = np.concatenate([S.sample(scen, t, B, rng, "train") for t in range(20)])
                live = LiveIVF(mode, x0, seed)
                live.db, live.lab = [x0], [assign_nearest(x0, live.C.astype(np.float32))[0]]
                curve = []
                t0 = time.perf_counter()
                for t in range(20, STEPS):
                    live.t = t
                    live.insert(S.sample(scen, t, B, rng, "train"))
                    if t % EVAL_EVERY == EVAL_EVERY - 1:
                        q = S.sample(scen, t, 1000, np.random.default_rng(30_000 + t), "test")
                        curve.append(dict(epoch=t + 1, compute=live.compute, **live.evaluate(q)))
                w = time.perf_counter() - t0
                rec = float(np.mean([c["recall10"] for c in curve]))
                write_run(RUNS, make_run(track=TRACK, method=mode, dataset=f"agnews_{scen}", n=curve[-1]["db"], d=384,
                          k_or_units=NL, epochs=1.0, batch=B, alpha0=DRIFT.ALPHA0, seed=seed, train_loss=None,
                          test_loss=1.0 - rec, wall_s=w, role="ours" if mode == "live_rewarm_bal" else "baseline",
                          alpha0_source="fixed", group=scen, metric="1-time_avg_recall10@nprobe8", stream_curve=curve,
                          alarms=live.alarms, total_compute=live.compute,
                          mean_ndis_p99=float(np.mean([c["ndis_p99"] for c in curve])),
                          mean_imbalance=float(np.mean([c["imbalance"] for c in curve]))))
                print(f"[{scen} s={seed}] {mode:18s} recall={rec:.4f} ndis_p99={np.mean([c['ndis_p99'] for c in curve]):.0f} "
                      f"imb={np.mean([c['imbalance'] for c in curve]):.3f} compute={live.compute:.3g} alarms={live.alarms[:6]}",
                      flush=True)


if __name__ == "__main__":
    main()
