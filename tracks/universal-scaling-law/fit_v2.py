"""T12 universal-scaling-law, v2: is alpha0* predicted by one shared law across EMA-of-statistics algorithms?

Input: every ``results/<track>/sweep.jsonl`` for T01, T04-T07 (cells = track x dataset x n x k x B x T).
alpha0* per cell = val-selected (core.sweep.select_alpha). Cells whose optimum is at the LOWER grid edge are
censored and excluded from fitting (counted and reported).

Models (log alpha0*):
  M0  null          c_alg
  M1  universal     c_alg + a log(n/k) + b log B + t log T            (shared exponents)
  M2  per-alg       c_alg + a_alg log(n/k) + b_alg log B + t_alg log T
  KM  e-means law   0.824 k^-0.46 B^0.90 n^-0.45 T^-0.11 (no fit; k-means law applied unchanged)

Score = val-loss penalty of the predicted alpha0: (L(a_hat) - L*) / |L*|, L interpolated in log alpha
on the cell's seed-mean val curve. NOT R^2 on log alpha0.
  LOCO  leave-one-cell-out (all models)
  LOAO  leave-one-algorithm-out (M0/M1 use the mean of the other algorithms' constants; M2 n/a)
Pre-registered decision (scripts/kill_criteria.py: t12_universal): universal law supported iff
M1 beats M0 in LOCO (median penalty lower, paired Wilcoxon p < 0.05) AND in LOAO (median lower).
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.harness import make_run, results_path, write_run  # noqa: E402
from core.scaling import emeans_alpha0  # noqa: E402
from core.sweep import interp_loss, sweep_cells  # noqa: E402

TRACK = "universal-scaling-law"
SOURCES = ("gmm-ema", "dictlearn-scaling", "lda-scaling", "als-ema", "hmm-ema")


def load_cells():
    cells = []
    for tr in SOURCES:
        p = results_path(tr, "sweep.jsonl")
        if not p.exists():
            continue
        for (g, T, m), c in sweep_cells(p).items():
            if m.startswith(("online_em", "stepwise")):
                continue
            kv = dict(part.split("=", 1) for part in g.split("|")[1:] if "=" in part)
            cells.append({"alg": tr, "group": g, "dataset": g.split("|")[0], "n": float(kv["n"]), "k": float(kv["k"]),
                          "B": float(kv["B"]), "T": float(T), "a_star": c["alpha_star"],
                          # alpha0 <= 1 by definition, so an optimum at the top edge is the constrained optimum;
                          # only the LOWER edge is censored (pre-registered 2026-11-13 before law-sweep data).
                          "edge": c["at_grid_edge"] and c["alpha_star_grid"] == min(c["val_curve"]),
                          "curve": c["val_curve"], "L_star": min(c["val_curve"].values())})
    return cells


def design(cells, algs, shared=True):
    rows = []
    for c in cells:
        onehot = [1.0 if c["alg"] == a else 0.0 for a in algs]
        f = [math.log(c["n"] / c["k"]), math.log(c["B"]), math.log(c["T"])]
        if shared:
            rows.append(onehot + f)
        else:
            rows.append(onehot + [o * v for o in onehot for v in f])
    return np.array(rows)


def fit(cells, model, algs):
    y = np.log([c["a_star"] for c in cells])
    if model == "M0":
        X = np.array([[1.0 if c["alg"] == a else 0.0 for a in algs] for c in cells])
    else:
        X = design(cells, algs, shared=(model == "M1"))
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta


def predict(beta, c, model, algs, alg_const=None):
    oh = [1.0 if c["alg"] == a else 0.0 for a in algs]
    f = [math.log(c["n"] / c["k"]), math.log(c["B"]), math.log(c["T"])]
    if model == "M0":
        x = oh
    elif model == "M1":
        x = oh + f
    else:
        x = oh + [o * v for o in oh for v in f]
    if alg_const is None:
        val = float(np.dot(beta, x))
    else:
        val = alg_const + (float(np.dot(beta[len(algs):], f)) if model == "M1" else 0.0)
    return min(max(math.exp(val), 1e-4), 1.0)


def penalty(c, a):
    return (interp_loss(c["curve"], a) - c["L_star"]) / max(abs(c["L_star"]), 1e-12)


def main():
    cells = load_cells()
    fitc = [c for c in cells if not c["edge"]]
    algs = sorted({c["alg"] for c in fitc})
    print(f"cells: {len(cells)} total, {len(cells) - len(fitc)} censored at grid edge, {len(fitc)} used; algorithms: {algs}")
    out = {}
    # LOCO
    for model in ("M0", "M1", "M2", "KM"):
        pens = []
        for i, c in enumerate(fitc):
            if model == "KM":
                a = emeans_alpha0(int(c["n"]), int(c["k"]), c["T"], int(c["B"]))
            else:
                train = fitc[:i] + fitc[i + 1:]
                if not any(t["alg"] == c["alg"] for t in train):
                    continue
                a = predict(fit(train, model, algs), c, model, algs)
            pens.append(penalty(c, a))
        out[("loco", model)] = pens
    # LOAO
    for model in ("M0", "M1"):
        pens = []
        for held in algs:
            train = [c for c in fitc if c["alg"] != held]
            others = sorted({c["alg"] for c in train})
            if not others:
                continue
            beta = fit(train, model, others)
            const = float(np.mean(beta[: len(others)]))
            for c in (c for c in fitc if c["alg"] == held):
                cc = dict(c, alg="__held__")
                pens.append(penalty(c, predict(beta, cc, model, others, alg_const=const)))
        out[("loao", model)] = pens
    # full-data fit for reporting exponents (+ bootstrap CI over cells)
    beta = fit(fitc, "M1", algs) if fitc else np.zeros(len(algs) + 3)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, len(fitc), len(fitc))
        sub = [fitc[i] for i in idx]
        if {c["alg"] for c in sub} == set(algs):
            boots.append(fit(sub, "M1", algs)[len(algs):])
    ci = np.percentile(np.array(boots), [2.5, 97.5], axis=0).T.tolist() if boots else None
    path = results_path(TRACK)
    if path.exists():
        path.unlink()  # derived track: recomputed from the sweeps every time
    for (ev, model), pens in out.items():
        if not pens:
            continue
        print(f"{ev:4s} {model}: median penalty {np.median(pens):.4%}  mean {np.mean(pens):.4%}  "
              f"<0.5%: {np.mean(np.array(pens) < 0.005):.0%}  (n={len(pens)})")
        write_run(path, make_run(track=TRACK, method=model, dataset="all_sweeps", n=len(pens), d=0, k_or_units=len(algs),
                  epochs=0, batch=0, alpha0=None, seed=0, train_loss=None, test_loss=float(np.median(pens)), wall_s=0.0,
                  role="ours" if model == "M1" else "baseline", alpha0_source="law", group=ev, metric="median_val_penalty",
                  penalties=[float(p) for p in pens], algorithms=algs, n_cells=len(cells), n_censored=len(cells) - len(fitc),
                  m1_exponents={"log(n/k)": float(beta[len(algs)]), "log B": float(beta[len(algs) + 1]),
                                "log T": float(beta[len(algs) + 2])} if model == "M1" else None,
                  m1_exponent_ci95=ci if model == "M1" else None))


if __name__ == "__main__":
    main()
