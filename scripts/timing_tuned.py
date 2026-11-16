#!/usr/bin/env python3
"""Isolated timing of the tuned prior-art baselines (stepwise / online EM, with and without revival), added
2026-11-17 to answer "wall-clock for the tuned baselines" (PAPER_LOG §7). For each comparison group it reads the
val-selected (tau0, kappa) per budget T from the main results' ``<m>_tuned`` / ``<m>_rev_tuned`` records, re-runs
those fits alone and records test loss and wall-clock in $CLASSIC_ALGOS_RESULTS_DIR/<track>/runs.jsonl under the same
method name, next to the reference / kmlaw / swept records of the v3 timing pass, so scripts/evaluate.py reports
wall-to-match for them in the same way. Same conditions as timing_swept.py: CPU only, 12 threads, run alone,
3 seeds, CLASSIC_ALGOS_BUDGETS restricting T to <= 5.

  CLASSIC_ALGOS_RESULTS_DIR=results/_timing CLASSIC_ALGOS_DEVICE=cpu CLASSIC_ALGOS_BUDGETS=0.25,0.5,1,2,3,5 \
      python scripts/timing_tuned.py hmm-ema
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from core.budgets import active  # noqa: E402
from core.harness import make_run, read_runs, write_run  # noqa: E402
from core.schedule import InverseTimeSchedule  # noqa: E402
from timing_swept import load_runner, out_path  # noqa: E402

BUDGETS = active()
MAIN = ROOT / "results"
SPEC = {"dictlearn-scaling": (("cifar_p8", "nmf_20ng", "cifar_p12"), "stepwise_dl", True),
        "lda-scaling": (("20ng", "enron", "nytimes"), "online_lda", True),
        "als-ema": (("ml1m", "ml10m"), "stepwise_als", False),
        "hmm-ema": (("synthetic", "brown_pos", "uci_har_vq"), "stepwise_em", True)}


def selections(track, method, group):
    """(tau0, kappa) per T of the main run's val-selected curve (identical across seeds)."""
    recs = [r for r in read_runs(MAIN / track / "runs.jsonl") if r["method"] == method and r["extra"]["group"] == group]
    if not recs:
        return None
    return {float(c["epoch"]): (c["tau0"], c["kappa"]) for c in recs[0]["extra"]["curve"]}


def emit(path, *, track, method, dataset, n, d, k, B, s, curve, group, metric):
    write_run(path, make_run(track=track, method=method, dataset=dataset, n=n, d=d, k_or_units=k,
                             epochs=curve[-1]["epoch"], batch=B, alpha0=None, seed=s, train_loss=None,
                             test_loss=curve[-1]["test_loss"], wall_s=curve[-1]["wall_s"], role="baseline",
                             alpha0_source="swept", group=group, metric=metric, curve=curve, curve_kind="budget",
                             selected_on="val", timing_pass=True))


def run_group(path, have, track, method, group, seeds, fit, loss, meta, warm):
    sel = selections(track, method, group)
    if not sel or any(T not in sel for T in BUDGETS):
        print(f"[timing] {track} {group} {method}: no selection, skipped", flush=True)
        return
    warm(sel[1.0])
    for s in seeds:
        if (method, group, s) in have:
            continue
        curve = []
        for T in BUDGETS:
            t0, kp = sel[T]
            tic = time.perf_counter()
            m = fit(T, s, InverseTimeSchedule(1.0, tau0=t0, kappa=kp))
            w = time.perf_counter() - tic
            curve.append({"epoch": T, "test_loss": loss(m), "wall_s": w, "tau0": t0, "kappa": kp})
        emit(path, method=method, group=group, s=s, curve=curve, **meta)
        print(f"[timing] {group} s={s} {method} walls={[round(c['wall_s'], 2) for c in curve]}", flush=True)


def trackkit_track(track, seeds):
    datasets, base, has_rev = SPEC[track]
    rv = load_runner(track)
    path = out_path(track)
    have = {(r["method"], r["extra"]["group"], r["seed"]) for r in read_runs(path)}
    for name in datasets:
        p = rv.make_problem(name)
        if isinstance(p, tuple):
            p = p[0]
        B = rv.HEAD_B[name] if isinstance(rv.HEAD_B, dict) else rv.HEAD_B
        for rev in ((False, True) if has_rev else (False,)):
            method = f"{base}{'_rev' if rev else ''}_tuned"
            meta = dict(track=track, dataset=p.dataset, n=p.n, d=p.d, k=p.k, B=B, metric=p.metric)
            run_group(path, have, track, method, p.group, seeds,
                      fit=lambda T, s, sch, rev=rev: p.fit(1.0, T, B, s, schedule=sch, revival=rev),
                      loss=lambda m: p.loss(m, "test"), meta=meta,
                      warm=lambda tk, rev=rev: p.fit(1.0, 1.0, B, 12345, n=min(p.n, 8 * B), revival=rev,
                                                     schedule=InverseTimeSchedule(1.0, tau0=tk[0], kappa=tk[1])))


def gmm_track(seeds):
    rv = load_runner("gmm-ema")
    path = out_path("gmm-ema")
    have = {(r["method"], r["extra"]["group"], r["seed"]) for r in read_runs(path)}
    B = rv.HEAD_B
    for name in ("synthetic", "sift1m_pca64", "dinov2_pca64", "agnews_pca64"):
        data, _ = rv.load(name)
        x = data["train"]
        n, d = x.shape
        k = 32 if name == "synthetic" else rv.HEAD_K
        for cov in ("diag", "full"):
            group = f"{name}|k={k}|{cov}"
            for rev in (False, True):
                method = f"online_em{'_rev' if rev else ''}_tuned"
                meta = dict(track="gmm-ema", dataset=name, n=n, d=d, k=k, B=B, metric="test_nll")
                run_group(path, have, "gmm-ema", method, group, seeds,
                          fit=lambda T, s, sch, rev=rev, cov=cov: rv.gmm_ema(x, k, cov, 1.0, T, B, s, schedule=sch,
                                                                             revival=rev)[0],
                          loss=lambda prm: rv.evaluate_gmm_nll(data["test"], prm), meta=meta,
                          warm=lambda tk, rev=rev, cov=cov: rv.gmm_ema(
                              x[:8 * B], k, cov, 1.0, 1.0, B, 12345, revival=rev,
                              schedule=InverseTimeSchedule(1.0, tau0=tk[0], kappa=tk[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track", choices=("gmm-ema", *SPEC))
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    return gmm_track(seeds) if a.track == "gmm-ema" else trackkit_track(a.track, seeds)


if __name__ == "__main__":
    main()
