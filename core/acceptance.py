"""Phase 0 acceptance test: reproduce e-means vs full-pass Lloyd@25 on SIFT1M + embeddings."""

from __future__ import annotations

import argparse
import numpy as np
import faiss

from core.bench import DatasetSplit, Timer, load_dinov2_cifar100, load_sift1m, load_text_embeddings
from core.emeans import EMeans, ToyKMeansPlugin, eval_kmeans_mse
from core.plugin import run_ema_engine
from core.schedule import CosineSchedule


def run_lloyd_baseline(train: np.ndarray, k: int, niter: int = 25, seed: int = 0) -> tuple[np.ndarray, float]:
    """Run full-pass Faiss Lloyd k-means for ``niter`` iterations."""
    faiss.omp_set_num_threads(8)
    n, d = train.shape
    km = faiss.Kmeans(
        d=d,
        k=k,
        niter=niter,
        nredo=1,
        seed=seed,
        verbose=False,
        max_points_per_centroid=max(n, 1),
    )
    with Timer(use_mps=False) as t:
        km.train(np.ascontiguousarray(train, dtype=np.float32))
    return km.centroids, t.wall_s


def evaluate_dataset(
    split: DatasetSplit,
    k: int,
    epoch_grid: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0),
    seeds: tuple[int, ...] = (0, 1, 2),
    device: str = "auto",
) -> dict[str, object]:
    """Benchmark e-means across ``epoch_grid`` against full-pass Lloyd@25."""
    train = np.ascontiguousarray(split.train, dtype=np.float32)
    test = np.ascontiguousarray(split.test, dtype=np.float32)

    lloyd_test_losses: list[float] = []
    lloyd_times: list[float] = []
    for s in seeds:
        c_lloyd, w_s = run_lloyd_baseline(train, k=k, niter=25, seed=s)
        lloyd_test_losses.append(eval_kmeans_mse(test, c_lloyd, device=device))
        lloyd_times.append(w_s)
    lloyd_mean = float(np.mean(lloyd_test_losses))
    lloyd_wall = float(np.mean(lloyd_times))

    print(f"\n=== {split.name} (n={train.shape[0]:,}, d={train.shape[1]}, k={k:,}) ===")
    print(f"Lloyd@25  test_mse={lloyd_mean:.6f} ± {np.std(lloyd_test_losses):.6f}  wall={lloyd_wall:.2f}s")

    first_beat: float | None = None
    epoch_results: dict[float, dict[str, float]] = {}
    for ep in epoch_grid:
        losses: list[float] = []
        walls: list[float] = []
        for s in seeds:
            em = EMeans(k=k, epochs=ep, device=device, seed=s)
            with Timer(use_mps=(device == "mps")) as t:
                em.fit(train)
            losses.append(eval_kmeans_mse(test, em.centroids_, device=device))
            walls.append(t.wall_s)
        mean_l = float(np.mean(losses))
        std_l = float(np.std(losses))
        mean_w = float(np.mean(walls))
        ratio = mean_l / lloyd_mean
        beat = mean_l <= lloyd_mean
        if beat and first_beat is None:
            first_beat = ep
        epoch_results[ep] = {"test_mse": mean_l, "std": std_l, "wall_s": mean_w, "ratio": ratio}
        print(
            f"e-means@{ep:<3} test_mse={mean_l:.6f} ± {std_l:.6f}  "
            f"ratio={ratio:.4f}x  wall={mean_w:.2f}s  {'[BEATS LLOYD@25]' if beat else ''}"
        )

    # Verify ToyKMeansPlugin equivalence on seed 0, epochs=2.0
    em0 = EMeans(k=k, epochs=2.0, device=device, seed=0).fit(train)
    assert em0.batch_size is not None and em0.alpha0 is not None
    steps = max(int(round(2.0 * (train.shape[0] // em0.batch_size))), 1)
    plug = ToyKMeansPlugin(k=k, device=device, seed=0)
    res = run_ema_engine(
        plug,
        train,
        epochs=2.0,
        batch_size=em0.batch_size,
        schedule=CosineSchedule(em0.alpha0, steps, eps=0.1),
        seed=0,
    )
    max_diff = float(np.max(np.abs(em0.centroids_ - res.params)))
    print(f"ToyKMeansPlugin max centroid diff vs EMeans: {max_diff:.2e}")
    assert max_diff == 0.0, f"Expected exact match, got {max_diff}"

    return {
        "dataset": split.name,
        "lloyd25_mse": lloyd_mean,
        "lloyd25_wall_s": lloyd_wall,
        "first_beat_epoch": first_beat,
        "epochs": epoch_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run T00 e-means acceptance benchmark.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--skip-dino", action="store_true")
    args = parser.parse_args()

    seeds = tuple(range(args.seeds))
    faiss.omp_set_num_threads(8)

    sift = load_sift1m(val_size=0, normalize=False)
    r_sift = evaluate_dataset(sift, k=4096, seeds=seeds, device=args.device)

    ag = load_text_embeddings(dataset_name="ag_news", max_train=100_000, max_test=7_600, val_size=0)
    r_ag = evaluate_dataset(ag, k=1024, seeds=seeds, device=args.device)

    results = [r_sift, r_ag]
    if not args.skip_dino:
        dino = load_dinov2_cifar100(val_size=0)
        r_dino = evaluate_dataset(dino, k=512, seeds=seeds, device=args.device)
        results.append(r_dino)

    print("\n=== ACCEPTANCE SUMMARY ===")
    all_ok = True
    for r in results:
        fb = r["first_beat_epoch"]
        ok = fb is not None and fb <= 3.0
        all_ok = all_ok and ok
        print(f"  {r['dataset']:28s} first_beat_epoch={fb}  status={'PASS' if ok else 'FAIL'}")

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

