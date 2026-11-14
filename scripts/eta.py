#!/usr/bin/env python3
"""Rough ETA per queue job from the JSONL: cells done vs. each job's grid, remaining cost from a per-job fit of
wall_s ~ c * n * T on its finished cells (newest 60 only, i.e. current contention and port settings).
Prints hours of remaining work per job; each job is one sequential process."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
ALPHAS = 8
BUDGETS = (1.0, 2.0, 3.0, 5.0)
STEP_GRID = 18


def rows(track, kind):
    p = R / track / f"{kind}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def cost_c(rs):
    v = [r["wall_s"] / (r["n"] * max(r["epochs"], 0.1)) for r in rs[-60:] if r.get("wall_s") and r.get("n")]
    return float(np.median(v)) if v else None


def law_job(track, ds, ks, cfg_fn, seeds, n_full, tag_ok=lambda r: True):
    sw = [r for r in rows(track, "sweep") if r["dataset"] == ds and tag_ok(r) and r.get("alpha0") is not None and not r["extra"].get("sweep_stepwise")]
    done = {(r["n"], r["k_or_units"], r["batch"], r["epochs"], round(r["alpha0"], 4), r["seed"]) for r in sw}
    cells = [(c["n"], k, c["B"], c["T"]) for c in cfg_fn(n_full) for k in ks]
    total = len(cells) * ALPHAS * seeds
    rem = [(n, T) for (n, k, B, T) in cells for _ in range(ALPHAS * seeds)]
    ndone = sum(1 for (n, k, B, T) in cells for a in range(ALPHAS) for s in range(seeds)
                if any(d[:4] == (n, k, B, T) for d in done)) if False else None
    # count per cell how many (alpha, seed) are done
    per = defaultdict(int)
    for d in done:
        per[d[:4]] += 1
    left_cost = sum(n * T * max(ALPHAS * seeds - per[(n, k, B, T)], 0) for (n, k, B, T) in cells)
    left = sum(max(ALPHAS * seeds - per[c], 0) for c in cells)
    law_rows = [r for r in sw if (r["n"], r["k_or_units"], r["batch"], r["epochs"]) in set(cells)]
    return total - left, total, left_cost, cost_c(law_rows) or cost_c(sw)


def all_job(track, ds, k, head_b, n_full, ref_method, seeds=3, tag_ok=lambda r: True):
    sw = [r for r in rows(track, "sweep") if r["dataset"] == ds and tag_ok(r)]
    head = {(r["epochs"], round(r["alpha0"], 4), r["seed"]) for r in sw
            if r.get("alpha0") is not None and not r["extra"].get("sweep_stepwise") and r["n"] == n_full and r["k_or_units"] == k and r["batch"] == head_b}
    step = [r for r in sw if r["extra"].get("sweep_stepwise") or r["extra"].get("sweep_oem")]
    runs = [r for r in rows(track, "runs") if r["dataset"] == ds and tag_ok(r)]
    ref_done = sum(1 for r in runs if r["method"] == ref_method)
    head_left = [T for T in BUDGETS for _ in range(ALPHAS * seeds)]
    per = defaultdict(int)
    for T, a, s in head:
        per[T] += 1
    left_cost = sum(n_full * T * (ALPHAS * seeds - per[T]) for T in BUDGETS)
    step_left = max(len(BUDGETS) * STEP_GRID * seeds - len(step), 0)
    left_cost += step_left * n_full * np.mean(BUDGETS)
    left_cost += max(seeds - ref_done, 0) * 25 * n_full
    total = len(BUDGETS) * ALPHAS * seeds + len(BUDGETS) * STEP_GRID * seeds + seeds
    done = len(head) + len(step) + min(ref_done, seeds)
    return done, total, left_cost, cost_c(sw)


def runs_job(track, group_ok, total, per_rec_s=None):
    rs = [r for r in rows(track, "runs") if group_ok(r)]
    w = [r["wall_s"] for r in rs[-30:] if r.get("wall_s")]
    s = per_rec_s or (float(np.median(w)) if w else None)
    return len(rs), total, (total - len(rs)), s


TRACK_C = {}


def fmt(job, done, total, left_cost, c, track=None):
    if c is None and track:
        c = TRACK_C.get(track)
    elif c is not None and track:
        TRACK_C.setdefault(track, c)
    hours = left_cost * c / 3600 if c else float("nan")
    print(f"{job:<16} {done:>5}/{total:<5} {'' if math.isnan(hours) else f'{hours:6.1f} h'}")
    return hours


def main():
    out = {}
    hmm_cfg = lambda n: [{"n": nn, "B": B, "T": T} for nn in (n // 4, n // 2, n) for B in (64, 512) for T in (1.0, 2.0, 5.0)]
    dl_cfg = lambda n: [{"n": nn, "B": B, "T": T} for nn in (n // 4, n) for B in (64, 512) for T in (1.0, 2.0, 5.0)]
    lda_cfg = lambda n: [{"n": nn, "B": B, "T": T} for nn in (n // 4, n) for B in (64, 512) for T in (1.0, 2.0)]
    gmm_cfg = lambda n: [{"n": nn, "B": B, "T": T} for nn in (n // 4, n // 2, n) for B in (256, 2048) for T in (1.0, 2.0, 5.0)]
    als_cfg = {"ml1m": 256, "ml10m": 2048}

    def nmax(track, ds):
        ns = [r["n"] for r in rows(track, "sweep") if r["dataset"] == ds]
        if ns:
            return max(ns)
        # not started: train size from the processed cache (dictlearn) / corpus (LDA, ~90% train)
        proc = ROOT / "data" / "processed"
        if track == "dictlearn-scaling" and (proc / f"dl_{ds}.npz").exists():
            return int(np.load(proc / f"dl_{ds}.npz")["train"].shape[0])
        if track == "lda-scaling" and (proc / f"lda_{ds}.npz").exists():
            return int(0.9 * (len(np.load(proc / f"lda_{ds}.npz")["indptr"]) - 1))
        return None

    print(f"{'job':<16} {'cells':>11}  remaining (one process)")
    # law sweeps
    n = nmax("hmm-ema", "brown_pos"); out["hmm_law"] = fmt("hmm_law", *law_job("hmm-ema", "brown_pos", (6, 12), hmm_cfg, 3, n), track="hmm-ema")
    for ds, j in (("cifar_p8", "dl_law_p8"), ("nmf_20ng", "dl_law_nmf")):
        n = nmax("dictlearn-scaling", ds)
        K = {"cifar_p8": 128, "nmf_20ng": 32}[ds]
        out[j] = fmt(j, *law_job("dictlearn-scaling", ds, (K // 2, K), dl_cfg, 3, n), track="dictlearn-scaling")
    for ds, j in (("20ng", "lda_law_20ng"), ("enron", "lda_law_enron"), ("nytimes", "lda_law_nyt")):
        n = nmax("lda-scaling", ds)
        out[j] = fmt(j, *law_job("lda-scaling", ds, (20, 50), lda_cfg, 2, n), track="lda-scaling") if n else print(f"{j:<16} not started")
    n = nmax("gmm-ema", "sift1m_pca64")
    out["gmm_law"] = fmt("gmm_law", *law_job("gmm-ema", "sift1m_pca64", (16, 64), gmm_cfg, 3, n,
                                             tag_ok=lambda r: r["extra"].get("config", {}).get("cov", "diag") == "diag")) if n else None
    for ds, j in (("ml1m", "als_law_1m"), ("ml10m", "als_law_10m")):
        n = nmax("als-ema", ds); hb = als_cfg[ds]
        out[j] = fmt(j, *law_job("als-ema", ds, (16, 32), lambda n, hb=hb: [{"n": n, "B": B, "T": T} for B in (hb // 4, hb, hb * 4)
                                                                         for T in (1.0, 2.0, 5.0)], 3, n)) if n else None
    # full stage-all jobs
    for ds, j, K in (("cifar_p8", "dl_p8", 128), ("nmf_20ng", "dl_nmf", 32), ("cifar_p12", "dl_p12", 192)):
        n = nmax("dictlearn-scaling", ds)
        out[j] = fmt(j, *all_job("dictlearn-scaling", ds, K, 256, n, "dl_full_batch"), track="dictlearn-scaling") if n else print(f"{j:<16} no rows yet")
    for ds, j in (("20ng", "lda_all_20ng"), ("enron", "lda_all_enron"), ("nytimes", "lda_all_nyt")):
        n = nmax("lda-scaling", ds)
        out[j] = fmt(j, *all_job("lda-scaling", ds, 50, 256, n, "batch_vb"), track="lda-scaling") if n else print(f"{j:<16} no rows yet")
    full = lambda r: r["extra"].get("config", {}).get("cov") == "full" or "|full" in r["extra"].get("group", "")
    for ds, j in (("synthetic", "gmm_full_syn"), ("dinov2_pca64", "gmm_full_dino"), ("agnews_pca64", "gmm_full_ag")):
        ks = [r["k_or_units"] for r in rows("gmm-ema", "sweep") if r["dataset"] == ds and full(r)]
        n = max((r["n"] for r in rows("gmm-ema", "sweep") if r["dataset"] == ds and full(r)), default=None)
        out[j] = fmt(j, *all_job("gmm-ema", ds, max(set(ks), key=ks.count), 1024, n, "em_full_batch", tag_ok=full), track="gmm-ema") if n else \
            print(f"{j:<16} no full-cov rows yet")
    # record-based jobs
    for scen, j in (("seasonal", "drift_seas"), ("encoder_upgrade", "drift_enc")):
        d, t, left, s = runs_job("drift-rewarm", lambda r, sc=scen: r["extra"]["group"] == sc, 45)
        out[j] = left * (s or 0) / 3600; print(f"{j:<16} {d:>5}/{t:<5} {out[j]:6.1f} h")
    for scen, j in (("abrupt", "divf_abr"), ("gradual", "divf_grad")):
        d, t, left, s = runs_job("drift-balanced-ivf", lambda r, sc=scen: r["extra"]["group"] == sc, 12)
        out[j] = left * (s or 0) / 3600; print(f"{j:<16} {d:>5}/{t:<5} {out[j]:6.1f} h" + ("" if s else "  (no record yet)"))
    for ds, j, ks in (("dinov2", "seed_dino", (10, 50, 100, 1024)), ("sift1m", "seed_sift", (10, 50, 100, 1024))):
        tot = sum(3 * 3 * sum(1 for m in range(7) if not ((m in (2, 3)) and (k > 100 or (ds == "sift1m" and k > 50)))) for k in ks)
        d, t, left, s = runs_job("cheap-seeding", lambda r, dd=ds: r["dataset"] == dd, tot)
        out[j] = left * (s or 0) / 3600; print(f"{j:<16} {d:>5}/{t:<5} {out[j]:6.1f} h" + ("" if s else "  (no record yet)"))
    d, t, left, s = runs_job("opq-joint-rotation", lambda r: True, 6 * 3 * 7)
    out["opq"] = left * (s or 0) / 3600; print(f"{'opq':<16} {d:>5}/{t:<5} {out['opq']:6.1f} h" + ("" if s else "  (no record yet)"))
    vals = {k: v for k, v in out.items() if isinstance(v, float) and not math.isnan(v)}
    if vals:
        k = max(vals, key=vals.get)
        print(f"\nlongest: {k} {vals[k]:.1f} h;  sum of all remaining work {sum(vals.values()):.0f} job-hours")


if __name__ == "__main__":
    main()
