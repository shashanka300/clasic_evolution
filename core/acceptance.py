"""T00 acceptance (v2): does our e-means reproduce "beats Lloyd@25 in ~1.4-2.6 epochs"?

Lloyd baseline = Faiss k-means, run as 25 chained single-iteration calls (identical to one 25-iteration
run, since Faiss Lloyd is deterministic given the centroids) so we can log the full trajectory.
e-means = independent runs per epoch budget T with the paper's alpha0 / batch laws.

Two comparison groups per (dataset, k): ``|train`` (training MSE, the paper's headline objective,
e-means with last-epoch-mean finalization) and ``|test`` (held-out MSE).

    python -m core.acceptance                     # full: SIFT1M k=1024,4096; AG News k=1024; DINOv2 k=512
    python -m core.acceptance --quick             # SIFT1M k=1024 only
"""

from __future__ import annotations

import argparse
import time

import faiss
import numpy as np

from core.bench import load_dinov2_cifar100, load_sift1m, load_text_embeddings
from core.emeans import EMeans, eval_kmeans_mse
from core.harness import done_hashes, make_run, results_path, write_run

TRACK = "core"
BUDGETS = (1.0, 1.5, 2.0, 2.5, 3.0, 5.0)


def lloyd_curve(train, test, k, seed, iters=25, device="auto"):
    faiss.omp_set_num_threads(10)
    n, d = train.shape
    rng = np.random.default_rng(seed)
    cent = np.ascontiguousarray(train[rng.choice(n, k, replace=False)], dtype=np.float32)
    pts, wall = [], 0.0
    for it in range(1, iters + 1):
        km = faiss.Kmeans(d, k, niter=1, nredo=1, seed=seed + it, verbose=False, max_points_per_centroid=n)
        t0 = time.perf_counter()
        km.train(train, init_centroids=cent)
        wall += time.perf_counter() - t0
        cent = np.ascontiguousarray(km.centroids, dtype=np.float32)
        pts.append({"epoch": it, "train": eval_kmeans_mse(train, cent, device=device),
                    "test": eval_kmeans_mse(test, cent, device=device), "wall_s": wall})
        print(f"  lloyd it={it:2d} train={pts[-1]['train']:.2f} test={pts[-1]['test']:.2f} wall={wall:.1f}s", flush=True)
    return pts


def emeans_curve(train, test, k, seed, device):
    pts = []
    meta = {}
    for T in BUDGETS:
        em = EMeans(k=k, epochs=T, device=device, seed=seed)
        t0 = time.perf_counter()
        em.fit(train)
        w = time.perf_counter() - t0
        meta = {"batch": em.batch_size, "alpha0": em.alpha0}
        pts.append({"epoch": T, "train": eval_kmeans_mse(train, em.centroids_, device=device),
                    "test": eval_kmeans_mse(test, em.centroids_, device=device), "wall_s": w,
                    "batch": em.batch_size, "alpha0": em.alpha0})
        print(f"  e-means T={T} B={em.batch_size} a0={em.alpha0:.3f} train={pts[-1]['train']:.2f} "
              f"test={pts[-1]['test']:.2f} wall={w:.1f}s", flush=True)
    return pts, meta


def emit(path, dsname, train, k, seed, method, pts, role, reference, alpha0, a0src, batch):
    for split in ("train", "test"):
        curve = [{"epoch": p["epoch"], "test_loss": p[split], "wall_s": p["wall_s"]} for p in pts]
        write_run(path, make_run(
            track=TRACK, method=method, dataset=dsname, n=train.shape[0], d=train.shape[1], k_or_units=k,
            epochs=pts[-1]["epoch"], batch=batch, alpha0=alpha0, seed=seed,
            train_loss=pts[-1]["train"], test_loss=pts[-1][split], wall_s=pts[-1]["wall_s"],
            role=role, alpha0_source=a0src, group=f"{dsname}|k={k}|{split}", metric=f"{split}_mse",
            reference=reference, curve=curve,
            curve_kind="trajectory" if method == "lloyd_faiss" else "budget",
            config={"split": split, "budgets": list(BUDGETS)},
            points=pts if method != "lloyd_faiss" else None,
        ))


def run(dsname, split, k, seeds, device, path):
    train = np.ascontiguousarray(split.train, dtype=np.float32)
    test = np.ascontiguousarray(split.test, dtype=np.float32)
    print(f"\n=== {dsname} n={train.shape[0]:,} d={train.shape[1]} k={k} ===", flush=True)
    for s in seeds:
        print(f" seed {s}", flush=True)
        pts = lloyd_curve(train, test, k, s, device=device)
        emit(path, dsname, train, k, s, "lloyd_faiss", pts, "baseline", True, None, "na", train.shape[0])
        pts, meta = emeans_curve(train, test, k, s, device)
        emit(path, dsname, train, k, s, "emeans", pts, "ours", False, meta["alpha0"], "law", meta["batch"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = tuple(range(args.seeds))
    path = results_path(TRACK)

    sift = load_sift1m(val_size=0)
    run("sift1m", sift, 1024, seeds, args.device, path)
    if args.quick:
        return
    run("sift1m", sift, 4096, seeds, args.device, path)
    run("agnews_minilm", load_text_embeddings(dataset_name="ag_news", max_train=100_000, max_test=7_600, val_size=0),
        1024, seeds, args.device, path)
    run("dinov2s_cifar100", load_dinov2_cifar100(val_size=0), 512, seeds, args.device, path)


if __name__ == "__main__":
    main()
