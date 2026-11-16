#!/usr/bin/env python3
"""Isolated timing of the *swept* EMA method: for each comparison group, re-run the fit at the val-selected alpha0 of
every budget T (the selection ``emit_swept_curves`` uses, read from the main results' sweep.jsonl) and record its
test loss and wall-clock. Written to $CLASSIC_ALGOS_RESULTS_DIR/<track>/runs.jsonl as method ``<ema>_swept``, next to
the reference and kmlaw records of the same timing pass, so scripts/evaluate.py can report wall-to-match for the
method the verdicts credit. Run alone (phase C2 of results/jobs_5090.txt).

  CLASSIC_ALGOS_RESULTS_DIR=results/_timing python scripts/timing_swept.py gmm-ema [--seeds 3]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.harness import make_run, read_runs, write_run  # noqa: E402
from core.sweep import sweep_cells, sweep_group  # noqa: E402

from core.budgets import active  # noqa: E402

# protocol v3 budgets, restricted by CLASSIC_ALGOS_BUDGETS: the v3 timing pass times T <= 5 only (PAPER_LOG §2)
BUDGETS = active()
MAIN = ROOT / "results"  # the queue's sweep rows (alpha0 selection); timing records go to the env results dir


def load_runner(track):
    tdir = ROOT / "tracks" / track
    sys.path.insert(0, str(tdir))
    spec = importlib.util.spec_from_file_location(f"{track.replace('-', '_')}_run_v2", tdir / "run_v2.py")
    m = importlib.util.module_from_spec(spec)
    sys.argv = [str(tdir / "run_v2.py")]
    spec.loader.exec_module(m)
    return m


def out_path(track):
    root = Path(os.environ.get("CLASSIC_ALGOS_RESULTS_DIR", ""))
    if not root or root.resolve() == MAIN.resolve():
        sys.exit("set CLASSIC_ALGOS_RESULTS_DIR to the timing directory (never the main results)")
    return root / track / "runs.jsonl"


def alphas_for(track, method, sgroup):
    cells = sweep_cells(MAIN / track / "sweep.jsonl", method)
    return {T: c["alpha_star_grid"] for (g, T, _m), c in cells.items() if g == sgroup and T in BUDGETS}


def emit(path, *, track, method, dataset, n, d, k, B, s, curve, group, metric, alphas):
    write_run(path, make_run(track=track, method=method, dataset=dataset, n=n, d=d, k_or_units=k, epochs=curve[-1]["epoch"],
                             batch=B, alpha0=curve[-1]["alpha0"], seed=s, train_loss=None, test_loss=curve[-1]["test_loss"],
                             wall_s=curve[-1]["wall_s"], role="ours", alpha0_source="swept", group=group, metric=metric,
                             curve=curve, curve_kind="budget", alpha0_by_T=alphas, selected_on="val", timing_pass=True))


def trackkit_track(track, datasets, ema, head_b, seeds):
    rv = load_runner(track)
    path = out_path(track)
    have = {(r["method"], r["extra"]["group"], r["seed"]) for r in read_runs(path)}
    for name in datasets:
        p = rv.make_problem(name)
        if isinstance(p, tuple):  # hmm-ema returns (problem, sanity)
            p = p[0]
        B = head_b[name] if isinstance(head_b, dict) else head_b
        alphas = alphas_for(track, ema, sweep_group(p.dataset, p.n, p.k, B, p.tag))
        if len(alphas) < len(BUDGETS):
            print(f"[timing] {track} {name}: sweep incomplete ({sorted(alphas)}), skipped", flush=True)
            continue
        # untimed warm-up on a small slice: numba kernels compile / load from cache outside the timed fits
        p.fit(alphas[1.0], 1.0, B, 12345, n=min(p.n, 8 * B))
        for s in seeds:
            if (f"{ema}_swept", p.group, s) in have:
                continue
            curve = []
            for T in BUDGETS:
                t0 = time.perf_counter()
                m = p.fit(alphas[T], T, B, s)
                w = time.perf_counter() - t0
                curve.append({"epoch": T, "test_loss": p.loss(m, "test"), "wall_s": w, "alpha0": alphas[T]})
            emit(path, track=track, method=f"{ema}_swept", dataset=p.dataset, n=p.n, d=p.d, k=p.k, B=B, s=s, curve=curve,
                 group=p.group, metric=p.metric, alphas=alphas)
            print(f"[timing] {p.group} s={s} {ema}_swept walls={[round(c['wall_s'], 2) for c in curve]}", flush=True)


def gmm_track(seeds):
    rv = load_runner("gmm-ema")
    path = out_path("gmm-ema")
    have = {(r["method"], r["extra"]["group"], r["seed"]) for r in read_runs(path)}
    for name in ("synthetic", "sift1m_pca64", "dinov2_pca64", "agnews_pca64"):
        data, _ = rv.load(name)
        x = data["train"]
        n, d = x.shape
        k = 32 if name == "synthetic" else rv.HEAD_K
        for cov in ("diag", "full"):
            group = f"{name}|k={k}|{cov}"
            alphas = alphas_for("gmm-ema", f"gmm_ema_{cov}", sweep_group(name, n, k, rv.HEAD_B, cov))
            if len(alphas) < len(BUDGETS):
                print(f"[timing] gmm {group}: sweep incomplete ({sorted(alphas)}), skipped", flush=True)
                continue
            rv.gmm_ema(x[:8 * rv.HEAD_B], k, cov, alphas[1.0], 1.0, rv.HEAD_B, 12345)  # untimed warm-up
            for s in seeds:
                if ("gmm_ema_swept", group, s) in have:
                    continue
                curve = []
                for T in BUDGETS:
                    t0 = time.perf_counter()
                    prm, _ = rv.gmm_ema(x, k, cov, alphas[T], T, rv.HEAD_B, s)
                    w = time.perf_counter() - t0
                    curve.append({"epoch": T, "test_loss": rv.evaluate_gmm_nll(data["test"], prm), "wall_s": w,
                                  "alpha0": alphas[T]})
                emit(path, track="gmm-ema", method="gmm_ema_swept", dataset=name, n=n, d=d, k=k, B=rv.HEAD_B, s=s,
                     curve=curve, group=group, metric="test_nll", alphas=alphas)
                print(f"[timing] {group} s={s} gmm_ema_swept walls={[round(c['wall_s'], 2) for c in curve]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track", choices=("gmm-ema", "dictlearn-scaling", "lda-scaling", "als-ema", "hmm-ema"))
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    if args.track == "gmm-ema":
        return gmm_track(seeds)
    spec = {"dictlearn-scaling": (("cifar_p8", "nmf_20ng", "cifar_p12"), "dl_ema", 256),
            "lda-scaling": (("20ng", "enron", "nytimes"), "lda_ema", 256),
            "als-ema": (("ml1m", "ml10m"), "als_ema", {"ml1m": 256, "ml10m": 2048}),
            "hmm-ema": (("synthetic", "brown_pos", "uci_har_vq"), "hmm_ema", 256)}[args.track]
    trackkit_track(args.track, *spec, seeds)


if __name__ == "__main__":
    main()
