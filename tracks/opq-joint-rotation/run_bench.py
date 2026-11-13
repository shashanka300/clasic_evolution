"""Benchmark runner for Track T02: opq-joint-rotation on Apple Silicon M1 Pro.

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

# Ensure root and tracks/opq-joint-rotation are on sys.path
TRACK_DIR = Path(__file__).resolve().parent
ROOT_DIR = TRACK_DIR.parent.parent
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Cap OpenMP threads per Mac M1 Pro execution rules
try:
    import faiss
    faiss.omp_set_num_threads(8)
except ImportError:
    faiss = None

from core.bench import Timer, data_root, load_dinov2_cifar100, load_sift1m, write_result
from core.schedule import CosineSchedule
from opq import (
    FaissOPQBenchmark,
    OPQQuantizer,
    PostHocOPQ,
    StandardPQ,
    generate_correlated_blobs,
)


def compute_groundtruth_top10(queries: np.ndarray, database: np.ndarray) -> np.ndarray:
    """Compute exact Euclidean top-10 nearest neighbors in batches."""
    num_q = queries.shape[0]
    num_db = database.shape[0]
    gt = np.zeros((num_q, 10), dtype=np.int32)
    q_norms = np.sum(queries**2, axis=1, keepdims=True)
    db_norms = np.sum(database**2, axis=1)

    batch_size = 200
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
    M: int,
    k_sub: int = 256,
    epochs: float = 1.0,
    batch_size: int = 1024,
    alpha0: float = 0.1,
    alpha_r: float = 0.05,
    refresh_m: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute one benchmark trial and produce frozen 15-key record."""
    n_train = int(x_train.shape[0])
    d_val = int(d)
    b_size = min(max(int(batch_size), 1), n_train)
    steps_per_epoch = max(n_train // b_size, 1)
    total_steps = max(int(round(epochs * steps_per_epoch)), 1)
    schedule = CosineSchedule(alpha0=alpha0, total_steps=total_steps, eps=0.1)

    t0 = time.perf_counter()
    wall_s = 0.0
    gpu_s = 0.0
    rot_det = 1.0
    rot_err = 0.0
    recall = 0.0

    if method_name == "standard_pq":
        model = StandardPQ(d=d, M=M, k_sub=k_sub, seed=seed)
        t_start = time.perf_counter()
        model.fit(x_train, epochs=epochs, batch_size=b_size, schedule=schedule, seed=seed)
        wall_s = time.perf_counter() - t_start

        train_sample = x_train[: min(n_train, 10_000)]
        train_loss = float(model.reconstruction_mse(train_sample))
        test_loss = float(model.reconstruction_mse(x_test))

        test_codes = model.encode(x_test)
        recall = float(model.evaluate_recall(queries, test_codes, groundtruth, top_k=10))

    elif method_name == "post_hoc_opq":
        model = PostHocOPQ(d=d, M=M, k_sub=k_sub, refit_epochs=0.5, seed=seed)
        t_start = time.perf_counter()
        model.fit(x_train, epochs=epochs, batch_size=b_size, schedule=schedule, seed=seed)
        wall_s = time.perf_counter() - t_start

        train_sample = x_train[: min(n_train, 10_000)]
        train_loss = float(model.reconstruction_mse(train_sample))
        test_loss = float(model.reconstruction_mse(x_test))

        test_codes = model.encode(x_test)
        recall = float(model.evaluate_recall(queries, test_codes, groundtruth, top_k=10))

        R = model.quantizer.R
        rot_det = float(np.linalg.det(R))
        rot_err = float(np.linalg.norm(R.T @ R - np.eye(d, dtype=np.float32)))

    elif method_name.startswith("opq_joint"):
        model = OPQQuantizer(
            d=d,
            M=M,
            k_sub=k_sub,
            alpha_r=alpha_r,
            refresh_m=refresh_m,
            seed=seed,
        )
        t_start = time.perf_counter()
        model.fit(
            x_train,
            epochs=epochs,
            batch_size=b_size,
            schedule=schedule,
            update_rotation=True,
            seed=seed,
        )
        wall_s = time.perf_counter() - t_start

        train_sample = x_train[: min(n_train, 10_000)]
        train_loss = float(model.reconstruction_mse(train_sample))
        test_loss = float(model.reconstruction_mse(x_test))

        test_codes = model.encode(x_test)
        recall = float(model.evaluate_recall(queries, test_codes, groundtruth, top_k=10))

        R = model.R
        rot_det = float(np.linalg.det(R))
        rot_err = float(np.linalg.norm(R.T @ R - np.eye(d, dtype=np.float32)))

    elif method_name == "faiss_opq":
        if faiss is None:
            raise RuntimeError("Faiss is not installed or available.")
        fb = FaissOPQBenchmark(d=d, M=M, nbits=8)
        t_start = time.perf_counter()
        fb.fit(x_train)
        wall_s = time.perf_counter() - t_start

        train_sample = x_train[: min(n_train, 10_000)]
        train_loss = float(fb.reconstruction_mse(train_sample))
        test_loss = float(fb.reconstruction_mse(x_test))
        recall = float(fb.evaluate_recall(queries, x_test, groundtruth, top_k=10))

        a_mat = faiss.vector_to_array(fb.opq.A).reshape(d, d)
        rot_det = float(np.linalg.det(a_mat))
        rot_err = float(np.linalg.norm(a_mat.T @ a_mat - np.eye(d)))

    else:
        raise ValueError(f"Unknown method {method_name}")

    record = {
        "track": "opq-joint-rotation",
        "method": method_name,
        "dataset": dataset_name,
        "n": n_train,
        "d": d_val,
        "k_or_units": int(M),
        "epochs": float(epochs),
        "batch": int(b_size),
        "alpha0": float(alpha0),
        "seed": int(seed),
        "train_loss": round(train_loss, 6),
        "test_loss": round(test_loss, 6),
        "wall_s": round(wall_s, 4),
        "gpu_s": round(gpu_s, 4),
        "extra": {
            "recall_at_10": round(recall, 4),
            "m_subspaces": int(M),
            "k_sub": int(k_sub),
            "d_s": int(d // M),
            "refresh_m": int(refresh_m),
            "alpha_r": float(alpha_r),
            "rotation_det": round(rot_det, 4),
            "rotation_orthogonality_error": round(rot_err, 6),
        },
    }

    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="T02 opq-joint-rotation benchmark runner")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke execution")
    parser.add_argument("--seeds", type=str, default="1", help="Comma-separated random seeds or count")
    parser.add_argument("--epochs", type=float, default=1.0, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=1024, help="Mini-batch size")
    parser.add_argument("--datasets", type=str, default="synthetic,sift1m,dinov2", help="Datasets to evaluate")
    parser.add_argument(
        "--methods",
        type=str,
        default="standard_pq,post_hoc_opq,opq_joint_m1,opq_joint_m10,opq_joint_m100,faiss_opq",
        help="Methods to benchmark",
    )
    parser.add_argument("--output", type=str, default="results/opq-joint-rotation/benchmarks.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    # Parse seeds
    if "," in args.seeds:
        seed_list = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seed_count = int(args.seeds)
        seed_list = [42 + i * 101 for i in range(seed_count)]

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    output_path = ROOT_DIR / args.output if not Path(args.output).is_absolute() else Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting T02 opq-joint-rotation Benchmark (smoke={args.smoke}) ===")
    print(f"Datasets: {datasets}")
    print(f"Methods: {methods}")
    print(f"Seeds: {seed_list}")
    print(f"Output: {output_path}")

    records_written = 0

    for dname in datasets:
        print(f"\n--- Loading dataset: {dname} ---")
        if dname == "synthetic":
            d = 32 if not args.smoke else 16
            m_values = [4, 8] if not args.smoke else [4]
            k_sub = 64 if args.smoke else 128
            n_train = 5000 if args.smoke else 30_000
            n_test = 500 if args.smoke else 3000

            x_train, x_test, queries, gt = generate_correlated_blobs(
                n_train=n_train,
                n_test=n_test,
                d=d,
                k_clusters=16,
                correlation=0.85,
                seed=seed_list[0],
            )
            queries = queries[: 50 if args.smoke else 200]
            gt = gt[: 50 if args.smoke else 200]

        elif dname == "sift1m":
            split = load_sift1m()
            d = 128
            m_values = [8, 16] if not args.smoke else [8]
            k_sub = 256
            train_max = 5000 if args.smoke else 50_000
            test_max = 1000 if args.smoke else 5000
            query_max = 50 if args.smoke else 200

            x_train = np.asarray(split.train[:train_max], dtype=np.float32)
            x_test = np.asarray(split.test[:test_max], dtype=np.float32)
            queries = x_test[:query_max].copy()
            # Compute exact ground truth against x_test for consistency
            gt = compute_groundtruth_top10(queries, x_test)

        elif dname == "dinov2":
            split = load_dinov2_cifar100()
            d = 384
            m_values = [16, 32] if not args.smoke else [16]
            k_sub = 256
            train_max = 5000 if args.smoke else 45_000
            test_max = 1000 if args.smoke else 5000
            query_max = 50 if args.smoke else 200

            x_train = np.asarray(split.train[:train_max], dtype=np.float32)
            x_test = np.asarray(split.test[:test_max], dtype=np.float32)
            queries = x_test[:query_max].copy()
            gt = compute_groundtruth_top10(queries, x_test)

        else:
            print(f"Unknown dataset {dname}, skipping.")
            continue

        for M in m_values:
            for method in methods:
                if method == "faiss_opq" and faiss is None:
                    print("Faiss not installed, skipping faiss_opq.")
                    continue

                refresh_m = 10
                if method == "opq_joint_m1":
                    refresh_m = 1
                elif method == "opq_joint_m10":
                    refresh_m = 10
                elif method == "opq_joint_m100":
                    refresh_m = 100

                for seed in seed_list:
                    print(
                        f"Running {method:16s} | {dname:10s} | d={d:3d} M={M:2d} | seed={seed:3d}...",
                        end="",
                        flush=True,
                    )
                    try:
                        record = run_benchmark_trial(
                            method_name=method,
                            dataset_name=dname,
                            x_train=x_train,
                            x_test=x_test,
                            queries=queries,
                            groundtruth=gt,
                            d=d,
                            M=M,
                            k_sub=k_sub,
                            epochs=args.epochs,
                            batch_size=args.batch_size if not args.smoke else 256,
                            refresh_m=refresh_m,
                            seed=seed,
                        )
                        write_result(output_path, record)
                        records_written += 1
                        print(
                            f" done! MSE={record['test_loss']:.4f} Recall@10={record['extra']['recall_at_10']:.4f} ({record['wall_s']:.2f}s)"
                        )
                    except Exception as e:
                        print(f" FAILED ({e})")
                        if args.smoke:
                            raise

    print(f"\n=== Benchmark Complete. Emitted {records_written} records to {output_path} ===")


if __name__ == "__main__":
    main()
