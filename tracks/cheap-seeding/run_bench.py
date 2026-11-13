"""Benchmark runner for Track T09: cheap-seeding on Apple Silicon M1 Pro.

Produces frozen 15-key JSONL records adhering to core.bench schema:
{track, method, dataset, n, d, k_or_units, epochs, batch, alpha0, seed, train_loss, test_loss, wall_s, gpu_s, extra}
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np

# Ensure root and tracks/cheap-seeding are on sys.path
TRACK_DIR = Path(__file__).resolve().parent
ROOT_DIR = TRACK_DIR.parent.parent
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from accounting import compute_budget_plan, compute_loss_ratios, BudgetPlan
from core.bench import Timer, data_root, load_dinov2_cifar100, load_sift1m, write_result
from core.emeans import eval_kmeans_mse
from core.scaling import emeans_alpha0, emeans_batch_size
from plugin import fit_budgeted_seeded_emeans, fit_budgeted_split_as_you_go
from seeding import (
    seed_afkmc2,
    seed_greedy_kmeanspp,
    seed_kmeans_parallel,
    seed_kmeanspp,
    seed_random,
    seed_subsampled_kmeanspp,
)

# Cap OpenMP threads per Mac M1 Pro execution rules
try:
    import faiss
    faiss.omp_set_num_threads(8)
except ImportError:
    pass


def generate_synthetic_blobs(
    n_train: int = 100_000,
    n_test: int = 10_000,
    d: int = 32,
    k_true: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate controlled synthetic Gaussian mixture blobs."""
    rng = np.random.default_rng(seed)
    # Generate true cluster centers with separation
    centers = rng.uniform(-10.0, 10.0, size=(k_true, d)).astype(np.float32)

    train_labels = rng.integers(0, k_true, size=n_train)
    test_labels = rng.integers(0, k_true, size=n_test)

    std = 1.0 / np.sqrt(d)
    x_train = centers[train_labels] + rng.normal(0.0, std, size=(n_train, d)).astype(np.float32)
    x_test = centers[test_labels] + rng.normal(0.0, std, size=(n_test, d)).astype(np.float32)

    return x_train, x_test


def run_single_experiment(
    method: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    dataset_name: str,
    k: int,
    total_epochs: float,
    seed: int = 0,
    batch_size: int | None = None,
    alpha0: float | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    """Execute one benchmark run under strict epoch budget accounting."""
    n, d = x_train.shape
    b = batch_size if batch_size is not None else emeans_batch_size(n, k, total_epochs)
    b = min(max(int(b), 1), n)
    a0 = float(alpha0) if alpha0 is not None else emeans_alpha0(n, k, total_epochs, b)

    rng = np.random.default_rng(seed)

    # 1. Seeding Phase
    t_seed_start = time.perf_counter()

    if method == "random":
        c_init, dist_evals_seed = seed_random(x_train, k, rng)
    elif method in ("kmeanspp", "kmeans_plusplus"):
        c_init, dist_evals_seed = seed_kmeanspp(x_train, k, rng)
    elif method == "greedy_kmeanspp":
        c_init, dist_evals_seed = seed_greedy_kmeanspp(x_train, k, rng)
    elif method in ("kmeans_parallel", "kmeans||"):
        c_init, dist_evals_seed = seed_kmeans_parallel(x_train, k, rng, rounds=3)
    elif method in ("afkmc2", "afk_mc2"):
        c_init, dist_evals_seed = seed_afkmc2(x_train, k, rng)
    elif method in ("subsampled_kmeanspp", "single_batch_kmeanspp"):
        c_init, dist_evals_seed = seed_subsampled_kmeanspp(x_train, k, rng, sample_ratio=0.05, max_sample=8192)
    elif method == "split_as_you_go":
        c_init = None
        dist_evals_seed = 0
    else:
        raise ValueError(f"Unknown seeding method: {method}")

    wall_s_seed = time.perf_counter() - t_seed_start

    # 2. Budget Accounting
    plan: BudgetPlan = compute_budget_plan(
        n=n,
        k=k,
        total_epochs=total_epochs,
        batch_size=b,
        dist_evals_seed=dist_evals_seed,
    )

    # 3. Clustering Phase
    t_clust_start = time.perf_counter()

    if method == "split_as_you_go":
        final_centroids, steps_run, revivals, dist_evals_clust = fit_budgeted_split_as_you_go(
            x=x_train,
            k=k,
            budget_plan=plan,
            alpha0=a0,
            device=device,
            seed=seed,
        )
    else:
        assert c_init is not None
        final_centroids, steps_run, revivals = fit_budgeted_seeded_emeans(
            x=x_train,
            initial_centroids=c_init,
            budget_plan=plan,
            alpha0=a0,
            device=device,
            seed=seed,
        )
        dist_evals_clust = steps_run * b * k

    wall_s_clust = time.perf_counter() - t_clust_start
    total_wall_s = wall_s_seed + wall_s_clust

    # 4. Evaluation (train & test MSE)
    # Evaluate test loss on held-out test split
    test_loss = float(eval_kmeans_mse(x_test, final_centroids, device=device))

    # Evaluate train loss (on a subset of up to 50k points if n > 50k for speed)
    if n > 50_000:
        idx_eval = np.linspace(0, n - 1, 50_000, dtype=np.int64)
        train_loss = float(eval_kmeans_mse(x_train[idx_eval], final_centroids, device=device))
    else:
        train_loss = float(eval_kmeans_mse(x_train, final_centroids, device=device))

    extra: dict[str, Any] = {
        "device": device if device in ("cpu", "mps") else "cpu",
        "mps_s": 0.0,
        "cpu_threads": 8,
        "hw": "m1_pro_32gb",
        "seeding_method": method,
        "seeding_cost_epochs": plan.seeding_cost_epochs,
        "clustering_epochs": plan.clustering_epochs,
        "dist_evals_seed": int(dist_evals_seed),
        "dist_evals_clust": int(dist_evals_clust),
        "dist_evals_total": int(dist_evals_seed + dist_evals_clust),
        "dist_evals_budget": plan.dist_evals_budget,
        "wall_s_seed": round(float(wall_s_seed), 6),
        "wall_s_clust": round(float(wall_s_clust), 6),
        "clustering_steps": int(steps_run),
        "revivals": int(revivals),
    }

    record: dict[str, Any] = {
        "track": "cheap-seeding",
        "method": method,
        "dataset": dataset_name,
        "n": int(n),
        "d": int(d),
        "k_or_units": int(k),
        "epochs": float(total_epochs),
        "batch": int(b),
        "alpha0": round(float(a0), 6),
        "seed": int(seed),
        "train_loss": round(float(train_loss), 6),
        "test_loss": round(float(test_loss), 6),
        "wall_s": round(float(total_wall_s), 6),
        "gpu_s": 0.0,
        "extra": extra,
    }

    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Track T09 Cheap Seeding Benchmarks")
    parser.add_argument("--output", type=str, default="results/cheap-seeding/benchmarks.jsonl")
    parser.add_argument("--datasets", nargs="+", default=["synthetic", "sift1m", "dinov2"])
    parser.add_argument("--methods", nargs="+", default=["random", "kmeanspp", "greedy_kmeanspp", "kmeans_parallel", "afkmc2", "subsampled_kmeanspp", "split_as_you_go"])
    parser.add_argument("--k-values", nargs="+", type=int, default=[10, 50, 100])
    parser.add_argument("--epochs", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting T09 Cheap Seeding Benchmarks")
    print(f"Datasets: {args.datasets}")
    print(f"Methods: {args.methods}")
    print(f"k values: {args.k_values}")
    print(f"Epochs: {args.epochs}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {out_path}")

    # Cache for random baseline test losses to compute ratio_vs_random
    random_losses: dict[tuple[str, int, float, int], float] = {}

    for dname in args.datasets:
        print(f"\n================ Loading Dataset: {dname} ================")
        if dname == "synthetic":
            # N = 100k, d = 32
            x_train, x_test = generate_synthetic_blobs(n_train=100_000, n_test=10_000, d=32, k_true=50, seed=42)
            d_val = 32
        elif dname == "sift1m":
            split = load_sift1m(val_size=10_000)
            x_train = np.asarray(split.train, dtype=np.float32)
            x_test = np.asarray(split.test, dtype=np.float32)
            d_val = 128
        elif dname in ("dinov2", "dinov2_cifar100"):
            split = load_dinov2_cifar100()
            x_train = np.asarray(split.train, dtype=np.float32)
            x_test = np.asarray(split.test, dtype=np.float32)
            d_val = 384
        else:
            raise ValueError(f"Unknown dataset: {dname}")

        print(f"Dataset {dname}: train shape {x_train.shape}, test shape {x_test.shape}")

        for k in args.k_values:
            for ep in args.epochs:
                # First run random baseline across seeds so baseline is recorded
                methods_order = ["random"] + [m for m in args.methods if m != "random"]

                for seed in args.seeds:
                    for method in methods_order:
                        # Skip heavy baselines (greedy_kmeanspp, kmeans_parallel) for k=100 on SIFT1M to respect runtime
                        if dname == "sift1m" and method in ("greedy_kmeanspp", "kmeans_parallel") and k > 50:
                            continue

                        print(f"[{dname}] k={k} T={ep} seed={seed} method={method} ... ", end="", flush=True)
                        t0 = time.perf_counter()
                        record = run_single_experiment(
                            method=method,
                            x_train=x_train,
                            x_test=x_test,
                            dataset_name=dname,
                            k=k,
                            total_epochs=ep,
                            seed=seed,
                            device=args.device,
                        )
                        elapsed = time.perf_counter() - t0

                        # Record random test loss
                        key = (dname, k, ep, seed)
                        if method == "random":
                            random_losses[key] = record["test_loss"]
                            record["extra"]["ratio_vs_random"] = 1.0
                        else:
                            rand_loss = random_losses.get(key)
                            if rand_loss is not None:
                                ratios = compute_loss_ratios(record["test_loss"], rand_loss)
                                record["extra"].update(ratios)

                        write_result(out_path, record)
                        print(f"done ({elapsed:.2f}s) -> test_loss: {record['test_loss']:.4f}, wall_s: {record['wall_s']:.2f}s, steps: {record['extra']['clustering_steps']}")

    print(f"\nAll benchmark runs complete. Results written to {out_path}")


if __name__ == "__main__":
    main()
