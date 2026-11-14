"""Generic v2 stages shared by the EMA-of-statistics tracks (T04-T07; T01 predates it).

A track supplies a :class:`Problem`; the stages produce, per comparison group:

  reference  <classic>_full_batch   25-iteration trajectory from the shared init
  baseline   <prior-art>_tuned      e.g. stepwise / online EM, hyperparameters chosen on val per T
  ours       <track>_ema_swept      cosine EMA + revival, alpha0 chosen on val per T
  ours       <track>_ema_kmlaw      alpha0, B from the e-means k-means law, untuned
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from core.budgets import BUDGETS, LONG_BUDGETS, active, anchor_rows, long_alphas, long_stepwise
from core.harness import config_hash, done_hashes, make_run, read_runs, write_run
from core.scaling import emeans_alpha0, emeans_batch_size
from core.schedule import InverseTimeSchedule
from core.sweep import emit_swept_curves, run_sweep, sweep_group

STEPWISE_GRID = tuple((t0, kp) for t0 in (2.0, 10.0, 100.0) for kp in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0))


@dataclass
class Problem:
    track: str
    dataset: str
    k: int
    d: int
    n: int
    metric: str
    # fit(alpha0, T, B, seed, schedule=None, revival=True, n=None) -> model
    fit: Callable[..., Any]
    # loss(model, split_name) -> float (lower is better)
    loss: Callable[[Any, str], float]
    # reference(seed) -> (points[{epoch,test_loss,val,wall_s}], model)
    reference: Callable[[int], tuple[list[dict], Any]]
    extras: Callable[[Any], dict] = lambda m: {}
    tag: str = ""
    config: dict = field(default_factory=dict)

    @property
    def group(self) -> str:
        return f"{self.dataset}|k={self.k}{('|' + self.tag) if self.tag else ''}"


def stage_reference(p: Problem, runs, seeds, ref_method: str, sanity: Callable[[int], tuple[float, float]] | None = None,
                    sanity_method: str | None = None):
    have = {(r["method"], r["seed"]) for r in read_runs(runs) if r["extra"]["group"] == p.group}
    for s in seeds:
        if (ref_method, s) not in have:
            pts, model = p.reference(s)
            write_run(runs, make_run(track=p.track, method=ref_method, dataset=p.dataset, n=p.n, d=p.d, k_or_units=p.k,
                      epochs=25, batch=p.n, alpha0=None, seed=s, train_loss=None, test_loss=pts[-1]["test_loss"],
                      wall_s=pts[-1]["wall_s"], role="baseline", alpha0_source="na", group=p.group, metric=p.metric,
                      reference=True, curve=pts, curve_kind="trajectory", config=p.config, **p.extras(model)))
            print(f"[ref] {p.group} s={s} {ref_method}@25 test={pts[-1]['test_loss']:.5g} ({pts[-1]['wall_s']:.0f}s)", flush=True)
        if sanity is not None and (sanity_method, s) not in have:
            loss, w = sanity(s)
            write_run(runs, make_run(track=p.track, method=sanity_method, dataset=p.dataset, n=p.n, d=p.d, k_or_units=p.k,
                      epochs=25, batch=p.n, alpha0=None, seed=s, train_loss=None, test_loss=loss, wall_s=w,
                      role="baseline", alpha0_source="na", group=p.group, metric=p.metric, config=p.config,
                      note="library sanity check of the reference; final point only"))


def stage_sweep(p: Problem, sweep_path, seeds, method: str, configs: list[dict], k: int | None = None, alphas=None):
    def fit_eval(x, kk, a0, T, B, s):
        m = p.fit(a0, T, B, s, n=len(x), k=kk)
        return {"val": p.loss(m, "val"), "test": p.loss(m, "test"), **p.extras(m)}

    run_sweep(track=p.track, method=method, dataset=p.dataset, data={"train": np.arange(p.n)}, k=k or p.k, d=p.d,
              fit_eval=fit_eval, configs=configs, seeds=seeds, path=sweep_path, metric=p.metric, tag=p.tag,
              extra_config=p.config, **({"alphas": alphas} if alphas is not None else {}))


def stage_head_sweep(p: Problem, sweep_path, seeds, method: str, head_B: int, budgets=BUDGETS):
    """Head-group sweep: full alpha0 grid at the short budgets, frozen restricted grid at the long ones."""
    budgets = active(budgets)
    short = [T for T in budgets if T not in LONG_BUDGETS]
    if short:
        stage_sweep(p, sweep_path, seeds, method, [{"n": p.n, "B": head_B, "T": T} for T in short])
    long_ = [T for T in budgets if T in LONG_BUDGETS]
    if long_:
        al = long_alphas(sweep_path, method, sweep_group(p.dataset, p.n, p.k, head_B, p.tag))
        stage_sweep(p, sweep_path, seeds, method, [{"n": p.n, "B": head_B, "T": T} for T in long_], alphas=al)


def stage_stepwise(p: Problem, runs, sweep_path, seeds, B: int, method: str, grid=STEPWISE_GRID):
    """Robbins-Monro / stepwise-EM step (tau0 + t)^-kappa, (tau0, kappa) chosen on val per T. No revival."""
    done = done_hashes(sweep_path)
    sg = f"stepwise|{p.group}"
    for T in active(BUDGETS):
        tgrid = long_stepwise(sweep_path, sg, anchor_rows(sweep_path, sg, "sweep_stepwise")) if T in LONG_BUDGETS else grid
        for t0, kp in tgrid:
            cfg = dict(p.config, method=method, dataset=p.dataset, T=T, B=B, tau0=t0, kappa=kp, n=p.n)
            h = config_hash(cfg)
            for s in seeds:
                if (h, s) in done:
                    continue
                tic = time.perf_counter()
                m = p.fit(1.0, T, B, s, schedule=InverseTimeSchedule(1.0, tau0=t0, kappa=kp), revival=False)
                w = time.perf_counter() - tic
                rec = make_run(track=p.track, method=method, dataset=p.dataset, n=p.n, d=p.d, k_or_units=p.k, epochs=T,
                               batch=B, alpha0=None, seed=s, train_loss=None, test_loss=p.loss(m, "test"), wall_s=w,
                               role="baseline", alpha0_source="na", group=sg, metric=p.metric, config=cfg,
                               val_loss=p.loss(m, "val"), tau0=t0, kappa=kp, sweep_stepwise=True)
                rec["extra"]["config_hash"] = h
                write_run(sweep_path, rec)
    if os.environ.get("CLASSIC_ALGOS_STEPWISE_CELLS_ONLY") or os.environ.get("CLASSIC_ALGOS_BUDGETS"):
        # per-seed worker of a split online stage: the (tau0, kappa) choice below averages val over ALL seeds, so it
        # is left to one final full-seed run of this stage once every worker's cells are in
        return
    out = f"{method}_tuned"
    if any(r["method"] == out and r["extra"]["group"] == p.group for r in read_runs(runs)):
        return
    rows = [r for r in read_runs(sweep_path) if r["extra"].get("sweep_stepwise") and r["extra"]["group"] == sg]
    for s in seeds:
        curve = []
        for T in BUDGETS:
            cands: dict[tuple, list] = {}
            for r in rows:
                if r["epochs"] == T:
                    cands.setdefault((r["extra"]["tau0"], r["extra"]["kappa"]), []).append(r)
            best = min(cands, key=lambda key: np.mean([r["extra"]["val_loss"] for r in cands[key]]))
            r = next(r for r in cands[best] if r["seed"] == s)
            curve.append({"epoch": T, "test_loss": r["test_loss"], "wall_s": r["wall_s"], "tau0": best[0], "kappa": best[1]})
        write_run(runs, make_run(track=p.track, method=out, dataset=p.dataset, n=p.n, d=p.d, k_or_units=p.k,
                  epochs=BUDGETS[-1], batch=B, alpha0=None, seed=s, train_loss=None, test_loss=curve[-1]["test_loss"],
                  wall_s=curve[-1]["wall_s"], role="baseline", alpha0_source="swept", group=p.group, metric=p.metric,
                  curve=curve, curve_kind="budget", selected_on="val", config=p.config))


def stage_kmlaw(p: Problem, runs, seeds, method: str, n_for_law: int | None = None):
    if any(r["method"] == method and r["extra"]["group"] == p.group for r in read_runs(runs)):
        return
    nl = n_for_law or p.n
    for s in seeds:
        curve = []
        for T in BUDGETS:
            B = emeans_batch_size(nl, p.k, T)
            a0 = emeans_alpha0(nl, p.k, T, B)
            tic = time.perf_counter()
            m = p.fit(a0, T, B, s)
            curve.append({"epoch": T, "test_loss": p.loss(m, "test"), "wall_s": time.perf_counter() - tic, "alpha0": a0, "B": B})
        write_run(runs, make_run(track=p.track, method=method, dataset=p.dataset, n=p.n, d=p.d, k_or_units=p.k,
                  epochs=BUDGETS[-1], batch=curve[-1]["B"], alpha0=curve[-1]["alpha0"], seed=s, train_loss=None,
                  test_loss=curve[-1]["test_loss"], wall_s=curve[-1]["wall_s"], role="ours", alpha0_source="law",
                  group=p.group, metric=p.metric, curve=curve, curve_kind="budget", config=p.config))


def stage_emit(p: Problem, runs, sweep_path, method: str, out_method: str, head_B: int):
    target = sweep_group(p.dataset, p.n, p.k, head_B, p.tag)
    emit_swept_curves(sweep_path=sweep_path, runs_path=runs, track=p.track, method=method, out_method=out_method,
                      group_of=lambda g: p.group if g == target else None, metric=p.metric)


def run_all(p: Problem, runs, sweep_path, seeds, *, ref_method, ema_method, stepwise_method, head_B,
            sanity=None, sanity_method=None, stage="all", law_sweep_configs=None, law_ks=()):
    if stage in ("all", "reference"):
        stage_reference(p, runs, seeds, ref_method, sanity, sanity_method)
    if stage in ("all", "sweep"):
        stage_head_sweep(p, sweep_path, seeds, ema_method, head_B)
    if stage == "plan":  # freeze the restricted long-budget grids (run once, after the T=5 cells are complete)
        sg = sweep_group(p.dataset, p.n, p.k, head_B, p.tag)
        print(f"[plan] {sg} alphas={long_alphas(sweep_path, ema_method, sg)} stepwise="
              f"{long_stepwise(sweep_path, f'stepwise|{p.group}', anchor_rows(sweep_path, f'stepwise|{p.group}', 'sweep_stepwise'))}",
              flush=True)
    if stage in ("all", "online"):
        stage_stepwise(p, runs, sweep_path, seeds, head_B, stepwise_method)
    if stage in ("all", "kmlaw"):
        stage_kmlaw(p, runs, seeds, f"{ema_method}_kmlaw")
    if stage in ("all", "emit"):
        stage_emit(p, runs, sweep_path, ema_method, f"{ema_method}_swept", head_B)
    if stage == "law-sweep" and law_sweep_configs:
        for kk in law_ks or (p.k,):
            stage_sweep(p, sweep_path, seeds, ema_method, law_sweep_configs, k=kk)
