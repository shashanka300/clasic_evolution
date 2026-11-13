"""Benchmark runner for Track T03: balanced-ivf on Apple Silicon.

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

# Ensure root and tracks/balanced-ivf are on sys.path
TRACK_DIR = Path(__file__).resolve().parent
ROOT_DIR = TRACK_DIR.parent.parent
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Cap OpenMP threads to single thread for deterministic query latency profiling
try:
    import faiss
    faiss.omp_set_num_threads(1)
except ImportError:
    faiss = None

from core.bench import Timer, data_root, load_sift1m, load_text_embeddings, write_result
from core.schedule import CosineSchedule
from balanced_ivf import (
    BalancedIVFIndex,
    BalancedKMeansIVFIndex,
    EMeansIVFIndex,
    FaissIVFBenchmark,
    compute_imbalance_factor,
    generate_skewed_data,
    pairwise_sq_dists,
)


def compute_groundtruth_top10(queries: np.ndarray, database: np.ndarray) -> np.ndarray:
    """Compute exact Euclidean top-10 nearest neighbors in batches."""
    num_q = queries.shape[0]
    num_db = database.shape[0]
    gt = np.zeros((num_q, 10), dtype=np.int32)
    q_norms = np.sum(queries**2, axis=1, keepdims=True)
    db_norms = np.sum(database**2, axis=1)

    batch_size = 250
    for q_start in range(0, num_q, batch_size):
        q_end = min(q_start + batch_size, num_q)
        dots = queries[q_start:q_end] @ database.T
        dists = q_norms[q_start:q_end] + db_norms[None, :] - 2.0 * dots
        gt[q_start:q_end] = np.argsort(dists, axis=1)[:, :10]

    return gt


def run_benchmark_trial(
    method_name: str,
    dataset_name: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    queries: np.ndarray,
    groundtruth: np.ndarray,
    d: int,
    nlist: int,
    epochs: float = 1.0,
    batch_size: int = 1024,
    alpha0: float = 0.1,
    epsilon: float = 0.05,
    num_iters: int = 3,
    nprobe_eval: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute one benchmark trial and produce a frozen 15-key record."""
    n_train = int(x_train.shape[0])
    b_size = min(max(int(batch_size), 1), n_train)

    t_start = time.perf_counter()
    imbalance = 1.0
    p50_lat = 0.0
    p99_lat = 0.0
    recall = 0.0
    probe_results: list[dict[str, Any]] = []

    if method_name == "balanced_ivf":
        index = BalancedIVFIndex(
            d=d,
            nlist=nlist,
            epsilon=epsilon,
            num_iters=num_iters,
            seed=seed,
        )
        index.fit(x_train, epochs=epochs, batch_size=b_size, alpha0=alpha0, seed=seed)
        index.build(x_train, assign_mode="nearest")
        wall_s = time.perf_counter() - t_start

        imbalance = index.imbalance_factor()
        # Evaluate query routing
        probes = [1, 2, 4, 8, 16, 32]
        probes = [p for p in probes if p <= nlist]
        if nprobe_eval not in probes and nprobe_eval <= nlist:
            probes.append(nprobe_eval)
            probes.sort()
        probe_results = index.evaluate_queries(queries, groundtruth, nprobe_list=probes)

        # Primary metrics at nprobe_eval
        chosen_probe = next((r for r in probe_results if r["nprobe"] == nprobe_eval), probe_results[-1])
        recall = chosen_probe["recall_at_10"]
        p50_lat = chosen_probe["p50_latency_ms"]
        p99_lat = chosen_probe["p99_latency_ms"]

        # MSE distortion
        train_sample = x_train[: min(n_train, 10_000)]
        test_sample = x_test[: min(len(x_test), 5_000)]
        train_loss = float(np.mean(np.min(pairwise_sq_dists(train_sample, index.centroids), axis=1)))
        test_loss = float(np.mean(np.min(pairwise_sq_dists(test_sample, index.centroids), axis=1)))

    elif method_name == "emeans_ivf":
        index = EMeansIVFIndex(
            d=d,
            nlist=nlist,
            seed=seed,
        )
        index.fit(x_train, epochs=epochs, batch_size=b_size, alpha0=alpha0, seed=seed)
        index.build(x_train, assign_mode="nearest")
        wall_s = time.perf_counter() - t_start

        imbalance = index.imbalance_factor()
        probes = [1, 2, 4, 8, 16, 32]
        probes = [p for p in probes if p <= nlist]
        if nprobe_eval not in probes and nprobe_eval <= nlist:
            probes.append(nprobe_eval)
            probes.sort()
        probe_results = index.evaluate_queries(queries, groundtruth, nprobe_list=probes)

        chosen_probe = next((r for r in probe_results if r["nprobe"] == nprobe_eval), probe_results[-1])
        recall = chosen_probe["recall_at_10"]
        p50_lat = chosen_probe["p50_latency_ms"]
        p99_lat = chosen_probe["p99_latency_ms"]

        train_sample = x_train[: min(n_train, 10_000)]
        test_sample = x_test[: min(len(x_test), 5_000)]
        train_loss = float(np.mean(np.min(pairwise_sq_dists(train_sample, index.centroids), axis=1)))
        test_loss = float(np.mean(np.min(pairwise_sq_dists(test_sample, index.centroids), axis=1)))

    elif method_name == "balanced_kmeans":
        index = BalancedKMeansIVFIndex(
            d=d,
            nlist=nlist,
            penalty_lambda=0.01,
            seed=seed,
        )
        index.fit(x_train, epochs=epochs, batch_size=b_size, alpha0=alpha0, seed=seed)
        index.build(x_train, assign_mode="nearest")
        wall_s = time.perf_counter() - t_start

        imbalance = index.imbalance_factor()
        probes = [1, 2, 4, 8, 16, 32]
        probes = [p for p in probes if p <= nlist]
        if nprobe_eval not in probes and nprobe_eval <= nlist:
            probes.append(nprobe_eval)
            probes.sort()
        probe_results = index.evaluate_queries(queries, groundtruth, nprobe_list=probes)

        chosen_probe = next((r for r in probe_results if r["nprobe"] == nprobe_eval), probe_results[-1])
        recall = chosen_probe["recall_at_10"]
        p50_lat = chosen_probe["p50_latency_ms"]
        p99_lat = chosen_probe["p99_latency_ms"]

        train_sample = x_train[: min(n_train, 10_000)]
        test_sample = x_test[: min(len(x_test), 5_000)]
        train_loss = float(np.mean(np.min(pairwise_sq_dists(train_sample, index.centroids), axis=1)))
        test_loss = float(np.mean(np.min(pairwise_sq_dists(test_sample, index.centroids), axis=1)))

    elif method_name == "faiss_ivf":
        bench = FaissIVFBenchmark(d=d, nlist=nlist, seed=seed)
        bench.fit_and_build(x_train, x_train)
        wall_s = time.perf_counter() - t_start

        imbalance = bench.imbalance_factor()
        probes = [1, 2, 4, 8, 16, 32]
        probes = [p for p in probes if p <= nlist]
        if nprobe_eval not in probes and nprobe_eval <= nlist:
            probes.append(nprobe_eval)
            probes.sort()
        probe_results = bench.evaluate_queries(queries, groundtruth, nprobe_list=probes)

        chosen_probe = next((r for r in probe_results if r["nprobe"] == nprobe_eval), probe_results[-1])
        recall = chosen_probe["recall_at_10"]
        p50_lat = chosen_probe["p50_latency_ms"]
        p99_lat = chosen_probe["p99_latency_ms"]

        # Approximate MSE via coarse quantizer centroids
        q_centroids = bench.index.quantizer.reconstruct_n(0, nlist)
        train_sample = x_train[: min(n_train, 10_000)]
        test_sample = x_test[: min(len(x_test), 5_000)]
        train_loss = float(np.mean(np.min(pairwise_sq_dists(train_sample, q_centroids), axis=1)))
        test_loss = float(np.mean(np.min(pairwise_sq_dists(test_sample, q_centroids), axis=1)))

    else:
        raise ValueError(f"Unknown method {method_name}")

    record = {
        "track": "balanced-ivf",
        "method": method_name,
        "dataset": dataset_name,
        "n": n_train,
        "d": d,
        "k_or_units": nlist,
        "epochs": float(epochs),
        "batch": b_size,
        "alpha0": float(alpha0),
        "seed": int(seed),
        "train_loss": float(train_loss),
        "test_loss": float(test_loss),
        "wall_s": float(wall_s),
        "gpu_s": 0.0,
        "extra": {
            "imbalance_factor": float(imbalance),
            "p50_latency_ms": float(p50_lat),
            "p99_latency_ms": float(p99_lat),
            "recall_at_10": float(recall),
            "nprobe": int(nprobe_eval),
            "epsilon": float(epsilon),
            "num_iters": int(num_iters),
            "all_probes": probe_results,
        },
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Track T03 balanced-ivf Benchmark Runner")
    parser.add_argument("--smoke", action="store_true", help="Run rapid smoke test")
    parser.add_argument("--seeds", type=int, default=1, help="Number of random seeds")
    parser.add_argument("--epochs", type=float, default=1.0, help="Number of training epochs")
    parser.add_argument("--datasets", nargs="+", default=["synthetic_skewed"], help="Datasets to evaluate")
    parser.add_argument("--methods", nargs="+", default=["balanced_ivf", "emeans_ivf"], help="Methods to benchmark")
    parser.add_argument("--nlist", type=int, default=64, help="Number of IVF inverted lists (K)")
    parser.add_argument("--nprobe", type=int, default=8, help="Query nprobe setting for summary record")
    parser.add_argument("--batch-size", type=int, default=1024, help="Mini-batch size")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Sinkhorn entropy regularization")
    parser.add_argument("--num-iters", type=int, default=3, help="Number of Sinkhorn iterations")
    parser.add_argument("--sweep-knobs", action="store_true", help="Run ablation sweep over Sinkhorn iters and epsilon")
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(ROOT_DIR / "results" / "balanced-ivf" / "benchmarks.jsonl"),
        help="Path to output JSONL file",
    )
    args = parser.parse_args()

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        print("[SMOKE] Executing rapid smoke test for balanced-ivf...")
        x_tr, x_te, q, gt = generate_skewed_data(
            n_train=2000,
            n_test=500,
            d=16,
            k_clusters=8,
            skew_factor=2.0,
            seed=42,
        )
        methods = ["balanced_ivf", "emeans_ivf"]
        for m in methods:
            rec = run_benchmark_trial(
                method_name=m,
                dataset_name="synthetic_skewed_smoke",
                x_train=x_tr,
                x_test=x_te,
                queries=q[:200],
                groundtruth=gt[:200],
                d=16,
                nlist=8,
                epochs=1.0,
                batch_size=256,
                alpha0=0.1,
                epsilon=args.epsilon,
                num_iters=args.num_iters,
                nprobe_eval=4,
                seed=42,
            )
            write_result(out_path, rec)
            print(
                f"  [{m}] Imbalance={rec['extra']['imbalance_factor']:.3f}, "
                f"Recall@10={rec['extra']['recall_at_10']:.3f}, "
                f"p99={rec['extra']['p99_latency_ms']:.3f}ms"
            )
        print("[SMOKE] Smoke test completed successfully.")
        return

    # Full benchmark execution
    datasets = args.datasets
    methods = args.methods

    for d_name in datasets:
        print(f"\n================ Loading Dataset: {d_name} ================")
        if d_name == "synthetic_skewed":
            d_dim = 32
            nlist_val = args.nlist
            x_tr, x_te, q, gt = generate_skewed_data(
                n_train=25_000,
                n_test=2_500,
                d=d_dim,
                k_clusters=nlist_val,
                skew_factor=2.2,
                seed=42,
            )
            eval_q = q[:1000]
            eval_gt = gt[:1000]

        elif d_name == "sift1m":
            split = load_sift1m()
            x_tr = np.asarray(split.train[:50_000], dtype=np.float32)
            x_te = np.asarray(split.test[:5_000], dtype=np.float32)
            eval_q = x_te[:1000]
            d_dim = x_tr.shape[1]
            nlist_val = min(args.nlist, 256)
            eval_gt = compute_groundtruth_top10(eval_q, x_tr)

        elif d_name in ("ag_news", "text_embeddings"):
            try:
                split = load_text_embeddings(dataset_name="ag_news", max_train=50_000)
                x_tr = np.asarray(split.train[:40_000], dtype=np.float32)
                x_te = np.asarray(split.test[:5_000], dtype=np.float32)
                eval_q = x_te[:1000]
                d_dim = x_tr.shape[1]
                nlist_val = min(args.nlist, 256)
                eval_gt = compute_groundtruth_top10(eval_q, x_tr)
            except Exception as e:
                print(f"Skipping text_embeddings due to load failure: {e}")
                continue
        else:
            print(f"Unknown dataset {d_name}, skipping.")
            continue

        for seed in range(42, 42 + args.seeds):
            for m in methods:
                print(f"--> Running {m} on {d_name} (seed={seed}, nlist={nlist_val})...")
                rec = run_benchmark_trial(
                    method_name=m,
                    dataset_name=d_name,
                    x_train=x_tr,
                    x_test=x_te,
                    queries=eval_q,
                    groundtruth=eval_gt,
                    d=d_dim,
                    nlist=nlist_val,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    alpha0=0.1,
                    epsilon=args.epsilon,
                    num_iters=args.num_iters,
                    nprobe_eval=args.nprobe,
                    seed=seed,
                )
                write_result(out_path, rec)
                print(
                    f"    Done {m}: Imbalance={rec['extra']['imbalance_factor']:.3f}, "
                    f"Recall@10={rec['extra']['recall_at_10']:.3f}, "
                    f"p50={rec['extra']['p50_latency_ms']:.3f}ms, "
                    f"p99={rec['extra']['p99_latency_ms']:.3f}ms, "
                    f"Wall={rec['wall_s']:.2f}s"
                )

        if args.sweep_knobs and d_name == "synthetic_skewed":
            print("\n--- Running Knob Sweep (Sinkhorn iterations and epsilon) on synthetic_skewed ---")
            for iters in [1, 3, 10]:
                for eps in [0.01, 0.05, 0.20]:
                    print(f"--> Sweep: num_iters={iters}, epsilon={eps}...")
                    rec = run_benchmark_trial(
                        method_name="balanced_ivf",
                        dataset_name="synthetic_skewed",
                        x_train=x_tr,
                        x_test=x_te,
                        queries=eval_q,
                        groundtruth=eval_gt,
                        d=d_dim,
                        nlist=nlist_val,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        alpha0=0.1,
                        epsilon=eps,
                        num_iters=iters,
                        nprobe_eval=args.nprobe,
                        seed=42,
                    )
                    write_result(out_path, rec)
                    print(
                        f"    [iters={iters}, eps={eps}] Imbalance={rec['extra']['imbalance_factor']:.3f}, "
                        f"Recall@10={rec['extra']['recall_at_10']:.3f}, "
                        f"p99={rec['extra']['p99_latency_ms']:.3f}ms"
                    )


if __name__ == "__main__":
    main()
