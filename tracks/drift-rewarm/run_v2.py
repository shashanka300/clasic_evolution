"""T08 drift-rewarm, v2 protocol. JSONL only; verdicts from scripts/evaluate.py.

Streams are built from REAL AG News sentence embeddings (4 classes, 384-d, unit norm):
  gradual          class mixture moves linearly from class 0 to class 3 over the stream
  abrupt           classes {0,1} for the first half, then classes {2,3} appear (mix 50/50 of all 4)
  seasonal         alternates {0,1} <-> {2,3} every 60 steps
  encoder_upgrade  all classes; MiniLM-L6 embeddings for the first half, BGE-small for the same kind
                   of texts afterwards (a real encoder change, not a synthetic rotation)
Train batches come from the train pool, evaluation windows from the disjoint test pool, both drawn
from the *current* distribution. 300 steps x B=512, k=64. Eval every 5 steps on 2000 fresh points.

test_loss = time-average of window MSE over all eval steps (lower is better); regret = mean
(window MSE - oracle MSE), oracle = e-means (T=3) fitted on 20k fresh points of the current distribution.

Methods (all share the first 30-step cosine warm-up from random-sample init):
  frozen_emeans      after warm-up alpha stays at eps*alpha0 forever (e-means anneals once)
  rewarm             frozen_emeans + Page-Hinkley on relative batch loss; alarm -> 30-step cosine
                     restart to alpha0, then hold at eps*alpha0 until the next alarm   (ours)
  const_alpha_<a>    VQ-VAE style constant alpha, a in CONST_GRID; best a per scenario chosen with
                     oracle access to the same stream (favours the baseline)          (kill comparator)
  spark_decay_<g>    Spark streaming k-means forgetfulness (per-cluster count-weighted, decay g)
  periodic_retrain   every 50 steps: e-means (T=2) on the last 50 batches (25.6k points); compute charged
Detector constants were calibrated once on a synthetic Gaussian stream (not on these streams).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.bench import load_text_embeddings  # noqa: E402
from core.emeans import EMeans, assign_nearest, eval_kmeans_mse  # noqa: E402
from core.harness import make_run, read_runs, results_path, write_run  # noqa: E402
from core.revival import revive_dead_units  # noqa: E402
from core.stats import EMAStats  # noqa: E402

TRACK = "drift-rewarm"
RUNS = results_path(TRACK)
K, B, STEPS, EVAL_EVERY, WARM = 64, 512, 300, 5, 30
ALPHA0, EPS = 0.5, 0.05
CONST_GRID = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4)
SPARK_GRID = (0.5, 0.8, 0.9, 0.95, 0.99)
PH_DELTA, PH_LAMBDA, PH_EMA = 0.005, 0.2, 0.05  # calibrated on synthetic Gaussian streams only: 0 false alarms, delay 1 (abrupt 2/16) / ~13 (gradual)


class PageHinkleyRel:
    """Page-Hinkley on z_t = loss_t / running_mean - 1 (scale-free)."""

    def __init__(self, delta=PH_DELTA, lam=PH_LAMBDA, ema=PH_EMA):
        self.delta, self.lam, self.ema = delta, lam, ema
        self.reset()

    def reset(self):
        self.mean, self.cum, self.mn, self.n = None, 0.0, 0.0, 0

    def update(self, loss):
        self.n += 1
        if self.mean is None:
            self.mean = loss
            return False
        z = loss / self.mean - 1.0
        self.mean = (1 - self.ema) * self.mean + self.ema * loss
        self.cum += z - self.delta
        self.mn = min(self.mn, self.cum)
        if self.cum - self.mn > self.lam:
            self.reset()
            return True
        return False


def cosine(t, period):
    return ALPHA0 * (EPS + (1 - EPS) * 0.5 * (1 + math.cos(math.pi * min(t, period) / period)))


class Streams:
    def __init__(self):
        mini = load_text_embeddings(dataset_name="ag_news", max_train=100_000, max_test=7_600, val_size=0)
        bge = load_text_embeddings(dataset_name="ag_news", model_name="BAAI/bge-small-en-v1.5", max_train=100_000,
                                   max_test=7_600, val_size=0)
        from datasets import load_dataset
        ds = load_dataset("fancyzhx/ag_news")
        ytr = np.array(ds["train"]["label"][:100_000])
        yte = np.array(ds["test"]["label"][:7_600])
        self.enc = {"mini": (np.asarray(mini.train, np.float32), np.asarray(mini.test, np.float32)),
                    "bge": (np.asarray(bge.train, np.float32), np.asarray(bge.test, np.float32))}
        self.pools = {c: (np.flatnonzero(ytr == c), np.flatnonzero(yte == c)) for c in range(4)}

    def mix(self, scen, t):
        """(class weights, encoder) at step t."""
        f = t / STEPS
        if scen == "gradual":
            w = np.array([max(0, 1 - 3 * f), max(0, 1 - abs(3 * f - 1)), max(0, 1 - abs(3 * f - 2)), max(0, 3 * f - 2)])
            return w / w.sum(), "mini"
        if scen == "abrupt":
            return (np.array([.5, .5, 0, 0]) if f < 0.5 else np.full(4, .25)), "mini"
        if scen == "seasonal":
            return (np.array([.5, .5, 0, 0]) if (t // 60) % 2 == 0 else np.array([0, 0, .5, .5])), "mini"
        if scen == "encoder_upgrade":
            return np.full(4, .25), ("mini" if f < 0.5 else "bge")
        raise ValueError(scen)

    def sample(self, scen, t, m, rng, split):
        w, enc = self.mix(scen, t)
        cls = rng.choice(4, size=m, p=w)
        si = 0 if split == "train" else 1
        idx = np.concatenate([rng.choice(self.pools[c][si], size=(cls == c).sum()) for c in range(4)])
        return self.enc[enc][si][idx]


class StreamKMeans:
    def __init__(self, x0, mode, param=None, seed=0):
        rng = np.random.default_rng(seed)
        self.C = x0[rng.choice(len(x0), K, replace=False)].astype(np.float64)
        self.stats = EMAStats.from_params(self.C)
        self.mode, self.param, self.rng = mode, param, rng
        self.t, self.restart_at, self.alarms = 0, 0, []
        self.ph = PageHinkleyRel()
        self.spark_n = np.zeros(K)
        self.buffer = []
        self.points_processed = 0

    def alpha(self):
        if self.mode.startswith("const"):
            return self.param
        rel = self.t - self.restart_at
        if rel < WARM:
            return cosine(rel, WARM)
        return ALPHA0 * EPS

    def step(self, x):
        lab, d2 = assign_nearest(x, self.C.astype(np.float32))
        self.points_processed += len(x)
        loss = float(d2.mean())
        if self.mode == "rewarm" and self.t >= WARM and self.ph.update(loss):
            self.restart_at = self.t
            self.alarms.append(self.t)
        cnt = np.bincount(lab, minlength=K).astype(np.float64)
        sums = np.zeros((K, x.shape[1]))
        np.add.at(sums, lab, x.astype(np.float64))
        if self.mode.startswith("spark"):
            g = self.param
            self.spark_n = self.spark_n * g
            tot = self.spark_n + cnt
            has = cnt > 0
            self.C[has] = (self.C[has] * self.spark_n[has, None] + sums[has]) / tot[has, None]
            self.spark_n = tot
        else:
            a = self.alpha()
            self.stats.counts = (1 - a) * self.stats.counts + a * cnt
            self.stats.sums = (1 - a) * self.stats.sums + a * sums
            if self.t % 5 == 4:
                revive_dead_units(self.stats, rng=self.rng)
            self.C = self.stats.means(np.float64)
        if self.mode == "periodic_retrain":
            self.buffer.append(x)
            self.buffer = self.buffer[-50:]
            if self.t % 50 == 49:
                xb = np.concatenate(self.buffer)
                em = EMeans(k=K, epochs=2.0, seed=int(self.rng.integers(1 << 30))).fit(xb)
                self.C = em.centroids_.astype(np.float64)
                self.stats = EMAStats.from_params(self.C, n0=1.0)
                self.points_processed += 2 * len(xb)
        self.t += 1


def run_stream(S, scen, mode, param, seed, oracle_cache):
    rng = np.random.default_rng(seed)
    x0 = S.sample(scen, 0, 4096, rng, "train")
    m = StreamKMeans(x0, mode, param, seed)
    curve, regrets = [], []
    t0 = time.perf_counter()
    for t in range(STEPS):
        m.step(S.sample(scen, t, B, rng, "train"))
        if t % EVAL_EVERY == EVAL_EVERY - 1:
            erng = np.random.default_rng(10_000 + t)  # same eval window for every method / seed
            xw = S.sample(scen, t, 2000, erng, "test")
            mse = eval_kmeans_mse(xw, m.C.astype(np.float32))
            if t not in oracle_cache:
                xo = S.sample(scen, t, 20_000, np.random.default_rng(20_000 + t), "train")
                oracle_cache[t] = eval_kmeans_mse(xw, EMeans(k=K, epochs=3.0, seed=0).fit(xo).centroids_)
            curve.append({"epoch": t + 1, "test_loss": mse, "oracle": oracle_cache[t]})
            regrets.append(mse - oracle_cache[t])
    return curve, float(np.mean(regrets)), m, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="gradual,abrupt,seasonal,encoder_upgrade")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    S = Streams()
    have = {(r["method"], r["extra"]["group"], r["seed"]) for r in read_runs(RUNS)}
    methods = ([("frozen_emeans", "frozen", None), ("rewarm", "rewarm", None), ("periodic_retrain", "periodic_retrain", None)]
               + [(f"const_alpha_{a}", "const", a) for a in CONST_GRID] + [(f"spark_decay_{g}", "spark", g) for g in SPARK_GRID])
    for scen in args.scenarios.split(","):
        oracle = {}
        for seed in range(args.seeds):
            for name, mode, param in methods:
                if (name, scen, seed) in have:
                    continue
                curve, regret, m, w = run_stream(S, scen, mode, param, seed, oracle)
                tl = float(np.mean([c["test_loss"] for c in curve]))
                role = "ours" if name == "rewarm" else "baseline"
                write_run(RUNS, make_run(track=TRACK, method=name, dataset=f"agnews_{scen}", n=STEPS * B, d=384, k_or_units=K,
                          epochs=1.0, batch=B, alpha0=param if mode == "const" else (None if mode == "spark" else ALPHA0),
                          seed=seed, train_loss=None, test_loss=tl, wall_s=w, role=role,
                          alpha0_source="swept" if mode in ("const", "spark") else "fixed", group=scen,
                          metric="time_avg_window_mse", stream_curve=curve, regret=regret, alarms=m.alarms,
                          points_processed=m.points_processed, spark_decay=param if mode == "spark" else None))
                print(f"[{scen} s={seed}] {name:22s} avgMSE={tl:.5f} regret={regret:.5f} alarms={m.alarms[:8]} "
                      f"pts={m.points_processed}", flush=True)


if __name__ == "__main__":
    main()
