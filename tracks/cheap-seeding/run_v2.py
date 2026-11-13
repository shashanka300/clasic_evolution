"""T09 cheap-seeding, v2 protocol: reuses the original budget-charged experiment
(``run_bench.run_single_experiment``: seeding cost is charged against the epoch budget in distance
evaluations) and writes v2 records. Verdicts from scripts/evaluate.py.

Groups = dataset | k | T (total budget incl. seeding); paired by seed against ``random``.
v2 adds k = 1024 on SIFT1M / DINOv2: the e-means paper's seeding remark concerns large k,
which the original k in {10, 50, 100} grid never reached.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

TRACK_DIR = Path(__file__).resolve().parent
ROOT = TRACK_DIR.parents[1]
sys.path.insert(0, str(TRACK_DIR))
sys.path.insert(0, str(ROOT))

from core.bench import load_dinov2_cifar100, load_sift1m  # noqa: E402
from core.harness import make_run, read_runs, results_path, write_run  # noqa: E402
from run_bench import generate_synthetic_blobs, run_single_experiment  # noqa: E402

TRACK = "cheap-seeding"
RUNS = results_path(TRACK)
METHODS = ["random", "kmeanspp", "greedy_kmeanspp", "kmeans_parallel", "afkmc2", "subsampled_kmeanspp", "split_as_you_go"]
KS = {"synthetic": (10, 50, 100), "sift1m": (10, 50, 100, 1024), "dinov2": (10, 50, 100, 1024)}
EXPENSIVE = {"greedy_kmeanspp", "kmeans_parallel"}


def load(name):
    if name == "synthetic":
        return generate_synthetic_blobs(n_train=100_000, n_test=10_000, d=32, k_true=50, seed=42)
    s = load_sift1m(val_size=10_000) if name == "sift1m" else load_dinov2_cifar100()
    return np.asarray(s.train, np.float32), np.asarray(s.test, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="synthetic,dinov2,sift1m")
    ap.add_argument("--epochs", default="1,2,3")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    have = {(r["method"], r["extra"]["group"], r["seed"]) for r in read_runs(RUNS)}
    for name in args.datasets.split(","):
        xtr, xte = load(name)
        for k in KS[name]:
            for T in (float(e) for e in args.epochs.split(",")):
                group = f"{name}|k={k}|T={T:g}"
                for s in range(args.seeds):
                    for m in METHODS:
                        if (m, group, s) in have:
                            continue
                        if m in EXPENSIVE and k > 100 or (name == "sift1m" and m in EXPENSIVE and k > 50):
                            continue
                        t0 = time.perf_counter()
                        rec = run_single_experiment(m, xtr, xte, name, k, T, seed=s, device="auto")
                        ex = rec["extra"]
                        write_run(RUNS, make_run(track=TRACK, method=m, dataset=name, n=rec["n"], d=rec["d"], k_or_units=k,
                                  epochs=T, batch=rec["batch"], alpha0=rec["alpha0"], seed=s, train_loss=rec["train_loss"],
                                  test_loss=rec["test_loss"], wall_s=rec["wall_s"],
                                  role="baseline" if m == "random" else "ours", alpha0_source="law", group=group,
                                  metric="test_mse_budget_charged",
                                  **{key: ex[key] for key in ("seeding_cost_epochs", "clustering_epochs", "dist_evals_seed",
                                                              "dist_evals_clust", "dist_evals_total", "dist_evals_budget",
                                                              "wall_s_seed", "wall_s_clust", "revivals")}))
                        print(f"[{group} s={s}] {m:20s} test={rec['test_loss']:.5g} seed_cost={ex['seeding_cost_epochs']:.3f}ep "
                              f"({time.perf_counter() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
