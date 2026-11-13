"""Shared alpha0-sweep protocol for every EMA-of-statistics track (Step 2).

Protocol (identical for all tracks so T12 can pool them):

* grid: ``ALPHA_GRID`` = 8 log-spaced values in [0.01, 1.0];
* each (config, alpha0, seed) run logs val / test / train loss at the end of an epoch budget T;
* alpha0* is chosen on **validation** loss only (test is never used for selection);
* alpha0* is refined by a parabola in log(alpha0) through the best grid point and its neighbours,
  bootstrapped over seeds for a 95% CI, and reported with its "flat region": grid values whose mean
  val loss is within ``max(0.5% * |L*|, SE(L*))`` of the best;
* the headline "ours" curve for a group uses, at each T, the val-selected alpha0 and reports test loss.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Callable, Iterable

import numpy as np

from core.harness import done_hashes, make_run, read_runs, write_run, config_hash

ALPHA_GRID = tuple(float(a) for a in np.round(np.logspace(-2, 0, 8), 4))  # 0.01 .. 1.0
FLAT_REL_TOL = 0.005


def sweep_group(dataset: str, n: int, k: int, batch: int, tag: str = "") -> str:
    return f"{dataset}|n={n}|k={k}|B={batch}{('|' + tag) if tag else ''}"


def run_sweep(
    *,
    track: str,
    method: str,
    dataset: str,
    data: dict[str, np.ndarray],
    k: int,
    d: int,
    fit_eval: Callable[[np.ndarray, int, float, float, int, int], dict[str, float]],
    configs: Iterable[dict[str, Any]],
    seeds: Iterable[int],
    path,
    metric: str,
    alphas: Iterable[float] = ALPHA_GRID,
    tag: str = "",
    extra_config: dict[str, Any] | None = None,
) -> None:
    """Run (and resume) a sweep. ``fit_eval(x_train, k, alpha0, T, B, seed) -> {val, test, train, ...}``.

    ``data`` must contain ``train`` (the full pool; the first ``n`` rows of a fixed permutation are
    used for a config with that ``n``), ``val`` and ``test``.
    """
    done = done_hashes(path)
    seeds = list(seeds)
    for cfg in configs:
        n, B, T = int(cfg["n"]), int(cfg["B"]), float(cfg["T"])
        x_tr = data["train"][:n]
        for a in alphas:
            base_cfg = dict(extra_config or {}, sweep=True, tag=tag)
            probe = dict(base_cfg, method=method, dataset=dataset, n=n, k=k, epochs=T, batch=B, alpha0=float(a))
            h = config_hash(probe)
            for s in seeds:
                if (h, s) in done:
                    continue
                t0 = time.perf_counter()
                out = fit_eval(x_tr, k, float(a), T, B, s)
                w = time.perf_counter() - t0
                extra = {key: v for key, v in out.items() if key not in ("val", "test", "train")}
                write_run(path, make_run(
                    track=track, method=method, dataset=dataset, n=n, d=d, k_or_units=k, epochs=T, batch=B,
                    alpha0=float(a), seed=s, train_loss=out.get("train"), test_loss=out["test"], wall_s=w,
                    role="ours", alpha0_source="fixed", group=sweep_group(dataset, n, k, B, tag), metric=metric,
                    config=base_cfg, val_loss=out["val"], sweep=True, **extra,
                ))
                print(f"  [{track}] {dataset} n={n} k={k} B={B} T={T} a0={a:.4f} s={s} "
                      f"val={out['val']:.5g} test={out['test']:.5g} ({w:.1f}s)", flush=True)


def _parabolic_log_min(la: np.ndarray, lv: np.ndarray, i: int) -> float:
    if i == 0 or i == len(la) - 1:
        return float(la[i])
    x, y = la[i - 1 : i + 2], lv[i - 1 : i + 2]
    denom = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2])
    a = (x[2] * (y[1] - y[0]) + x[1] * (y[0] - y[2]) + x[0] * (y[2] - y[1])) / denom
    b = (x[2] ** 2 * (y[0] - y[1]) + x[1] ** 2 * (y[2] - y[0]) + x[0] ** 2 * (y[1] - y[2])) / denom
    if a <= 0:
        return float(la[i])
    return float(np.clip(-b / (2 * a), x[0], x[2]))


def select_alpha(rows: list[dict[str, Any]], n_boot: int = 1000, rng_seed: int = 0) -> dict[str, Any]:
    """alpha0* summary for one (group, T) cell of a sweep."""
    by_a: dict[float, dict[int, float]] = defaultdict(dict)
    test_by_a: dict[float, dict[int, float]] = defaultdict(dict)
    for r in rows:
        by_a[r["alpha0"]][r["seed"]] = r["extra"]["val_loss"]
        test_by_a[r["alpha0"]][r["seed"]] = r["test_loss"]
    alphas = sorted(a for a in by_a if all(v is not None and math.isfinite(v) for v in by_a[a].values()))
    seeds = sorted(set.intersection(*(set(by_a[a]) for a in alphas)))
    mat = np.array([[by_a[a][s] for s in seeds] for a in alphas])  # (A, S)
    mean = mat.mean(axis=1)
    i = int(np.argmin(mean))
    la = np.log(np.array(alphas))
    best_log = _parabolic_log_min(la, mean, i)
    se = float(mat[i].std(ddof=1) / math.sqrt(len(seeds))) if len(seeds) > 1 else 0.0
    tol = max(FLAT_REL_TOL * abs(mean[i]), se)
    flat = [a for a, m in zip(alphas, mean) if m <= mean[i] + tol]
    rng = np.random.default_rng(rng_seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(seeds), len(seeds))
        mb = mat[:, idx].mean(axis=1)
        j = int(np.argmin(mb))
        boots.append(_parabolic_log_min(la, mb, j))
    lo, hi = np.exp(np.percentile(boots, [2.5, 97.5]))
    return {
        "alpha_star": float(math.exp(best_log)),
        "alpha_star_grid": float(alphas[i]),
        "ci95": (float(lo), float(hi)),
        "flat_range": (float(min(flat)), float(max(flat))),
        "at_grid_edge": i in (0, len(alphas) - 1),
        "val_curve": {float(a): float(m) for a, m in zip(alphas, mean)},
        "test_at_star": {s: test_by_a[alphas[i]][s] for s in seeds if s in test_by_a[alphas[i]]},
        "seeds": seeds,
    }


def interp_loss(val_curve: dict[float, float], alpha: float) -> float:
    """Mean val loss at ``alpha`` by linear interpolation in log(alpha); clamps to the grid ends."""
    xs = sorted(val_curve)
    la = math.log(min(max(alpha, xs[0]), xs[-1]))
    lx = [math.log(x) for x in xs]
    return float(np.interp(la, lx, [val_curve[x] for x in xs]))


def sweep_cells(path, method: str | None = None) -> dict[tuple, dict[str, Any]]:
    """{(group, T): select_alpha(...)} over all sweep rows in ``path``."""
    cells: dict[tuple, list] = defaultdict(list)
    for r in read_runs(path):
        if r["extra"].get("sweep") and (method is None or r["method"] == method):
            cells[(r["extra"]["group"], r["epochs"], r["method"])].append(r)
    return {key: select_alpha(rows) for key, rows in cells.items()}


def emit_swept_curves(*, sweep_path, runs_path, track, method, out_method, group_of, metric, extra_fn=None) -> None:
    """Write one headline record per seed: test loss at the val-selected alpha0 for every budget T.

    ``group_of(sweep_group) -> comparison group`` maps a sweep group onto the group holding the
    reference baseline (None = skip).
    """
    cells = sweep_cells(sweep_path, method)
    rows_by = defaultdict(list)
    for r in read_runs(sweep_path):
        if r["method"] == method and r["extra"].get("sweep"):
            rows_by[(r["extra"]["group"], r["epochs"], r["alpha0"], r["seed"])].append(r)
    by_group: dict[str, dict[float, dict]] = defaultdict(dict)
    for (g, T, m), cell in cells.items():
        if group_of(g) is not None:
            by_group[g][T] = cell
    existing = {(r["method"], r["extra"]["group"], r["seed"]) for r in read_runs(runs_path)}
    for g, per_T in by_group.items():
        cg = group_of(g)
        seeds = sorted(set.intersection(*(set(c["seeds"]) for c in per_T.values())))
        for s in seeds:
            if (out_method, cg, s) in existing:
                continue
            curve, alphas, last = [], {}, None
            for T in sorted(per_T):
                a = per_T[T]["alpha_star_grid"]
                (r,) = rows_by[(g, T, a, s)][:1]
                curve.append({"epoch": T, "test_loss": r["test_loss"], "wall_s": r["wall_s"], "alpha0": a})
                alphas[T] = a
                last = r
            write_run(runs_path, make_run(
                track=track, method=out_method, dataset=last["dataset"], n=last["n"], d=last["d"],
                k_or_units=last["k_or_units"], epochs=last["epochs"], batch=last["batch"], alpha0=alphas[max(alphas)],
                seed=s, train_loss=last["train_loss"], test_loss=last["test_loss"], wall_s=last["wall_s"],
                role="ours", alpha0_source="swept", group=cg, metric=metric, curve=curve, curve_kind="budget",
                alpha0_by_T=alphas, selected_on="val", **(extra_fn(last) if extra_fn else {}),
            ))
