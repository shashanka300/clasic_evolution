"""T12 exploratory refit: is alpha0* one law across algorithms once size is measured per statistic cell?

Hypothesis (from the literature review, reports/Novel directions for EMA statistics.md): the EMA rate is set by how
many observations each sufficient-statistic cell receives per step, not by n/k. Per-step observations per cell:

  gmm        q = B / k                       (each point feeds one component's d coordinates)
  dictlearn  q = B * rho                     (rho = code density: each sample feeds rho*k atoms)
  lda        q = B * Lbar / (K * V)          (tokens per (topic, word) cell; Lbar = tokens per doc)
  hmm        q = B * Lbar / (K * V)          (emission cells; Lbar = tokens per sequence)
  als        q = B * (ratings/user) / items  (ratings per item statistic)
and q_eff replaces V by the effective vocabulary exp(H(unigram)) (Zipf: most cells are almost never hit).

Models for log alpha0* (scored like fit_v2: val-loss penalty of the predicted alpha, LOCO and leave-one-algorithm-out):
  KM    e-means k-means law, unchanged
  M1    c_alg + log(n/k), log B, log T          (fit_v2's universal model; needs a per-algorithm constant)
  M1g   M1 with ONE global constant
  Q1    c_alg + log q, log(n/B), log T
  Q0    Q1 with ONE global constant           <- a truly universal law: predicts an unseen algorithm with no constant
  Qe0   Q0 with q_eff
  --missing-info (2026-11-17): adds Qh1 / Qh0 = Qe1 / Qe0 plus log(1 - h), h = normalised posterior entropy of the
  latents (missing_info.py; 0 for dictionary learning and ALS). Saved to refit_coords_mi.json; the default run is
  unchanged.
Censoring: optima at the grid's lower edge (0.01) are excluded from fitting, but every cell is scored (the loss
interpolation clamps to the grid, so predicting <= 0.01 for an edge cell costs nothing).
Exploratory: not pre-registered, writes nothing to results/*.jsonl; prints a report and saves a JSON summary.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.harness import read_runs, results_path  # noqa: E402
from core.scaling import emeans_alpha0  # noqa: E402
from core.sweep import interp_loss, sweep_cells  # noqa: E402

DATA = ROOT / "data"
SOURCES = ("gmm-ema", "dictlearn-scaling", "lda-scaling", "als-ema", "hmm-ema")
OUT = ROOT / "results" / "universal-scaling-law" / "refit_coords.json"
STATS = ROOT / "results" / "universal-scaling-law" / "dataset_stats.json"


def _veff(counts):
    p = np.asarray(counts, np.float64)
    p = p[p > 0] / p.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def dataset_stats():
    if STATS.exists():
        return json.loads(STATS.read_text())
    st = {}
    for name in ("20ng", "enron", "nytimes"):
        z = np.load(DATA / "processed" / f"lda_{name}.npz")
        ip, ids, cts, V = z["indptr"], z["ids"], z["cts"], int(z["V"])
        st[f"lda-scaling/{name}"] = {"L": float(cts.sum() / (len(ip) - 1)), "V": V,
                                    "Veff": _veff(np.bincount(ids, weights=cts, minlength=V))}
    # HMM: synthetic (rebuild the emission matrix exactly as tracks/hmm-ema/run_v2.synthetic draws it)
    rng = np.random.default_rng(7)
    S, V = 12, 200
    rng.dirichlet(np.ones(S))
    for i in range(S):
        rng.dirichlet(np.ones(S) * 0.3 + 8.0 * np.eye(S)[i])
    Bm = rng.dirichlet(np.ones(V) * 0.05, size=S)
    st["hmm-ema/synthetic"] = {"L": 40.0, "V": V, "Veff": float(np.mean([_veff(b) for b in Bm]))}
    z = np.load(DATA / "processed" / "uci_har_vq.npz")
    st["hmm-ema/uci_har_vq"] = {"L": 128.0, "V": 64, "Veff": _veff(np.bincount(z["train"].ravel(), minlength=64))}
    import nltk
    nltk.data.path.insert(0, str(DATA / "raw" / "nltk_data"))
    from collections import Counter
    from nltk.corpus import brown as bc
    words = [[w.lower() for w in s] for s in bc.sents()]
    cnt = Counter(w for s in words for w in s)
    top = [c for _, c in cnt.most_common(4999)]
    unk = sum(cnt.values()) - sum(top)
    st["hmm-ema/brown_pos"] = {"L": float(np.mean([min(len(s), 80) for s in words])), "V": 5000,
                               "Veff": _veff(top + [unk])}
    STATS.write_text(json.dumps(st, indent=1))
    return st


def load_cells(st):
    cells = []
    for tr in SOURCES:
        p = results_path(tr, "sweep.jsonl")
        if not p.exists():
            continue
        rows = [r for r in read_runs(p) if r["extra"].get("sweep")]
        dens = defaultdict(list)
        als_meta = {}
        for r in rows:
            if "code_density" in r["extra"]:
                dens[r["extra"]["group"]].append(r["extra"]["code_density"])
            cfg = r["extra"].get("config", {})
            if "n_ratings_train" in cfg:
                als_meta.setdefault(r["dataset"], (cfg["n_ratings_train"], cfg["n_items"], r["n"]))
                full = als_meta[r["dataset"]]
                if r["n"] > full[2]:
                    als_meta[r["dataset"]] = (cfg["n_ratings_train"], cfg["n_items"], r["n"])
        for (g, T, m), c in sweep_cells(p).items():
            if m.startswith(("online_em", "stepwise")):
                continue
            ds = g.split("|")[0]
            kv = dict(part.split("=", 1) for part in g.split("|")[1:] if "=" in part)
            n, k, B = float(kv["n"]), float(kv["k"]), float(kv["B"])
            if tr == "gmm-ema":
                q = qe = B / k
            elif tr == "dictlearn-scaling":
                q = qe = B * float(np.mean(dens[g])) if dens[g] else B * 0.1
            elif tr == "als-ema":
                nr, ni, nu = als_meta[ds]
                q = qe = B * (nr / nu) / ni
            else:
                s = st[f"{tr}/{ds}"]
                q = B * s["L"] / (k * s["V"])
                qe = B * s["L"] / (k * s["Veff"])
            cells.append({"alg": tr, "group": g, "T": float(T), "n": n, "k": k, "B": B, "q": q, "qe": qe,
                          "a_star": c["alpha_star"], "a_grid": c["alpha_star_grid"],
                          "edge": c["at_grid_edge"] and (c["alpha_star_grid"] == min(c["val_curve"]) or
                                                   max(c["val_curve"]) < 1.0),  # restricted long-budget grids (core.budgets)
                          "top": c["at_grid_edge"] and c["alpha_star_grid"] == max(c["val_curve"]),
                          "curve": c["val_curve"], "L_star": min(c["val_curve"].values())})
    return cells


def group_key(c):
    g = c["group"].split("|")
    kv = dict(x.split("=", 1) for x in g[1:] if "=" in x)
    tag = [x for x in g[1:] if "=" not in x]
    return f"{c['alg']}|{g[0]}|n={int(float(kv['n']))}|k={int(float(kv['k']))}" + (f"|{tag[0]}" if tag else "")


def attach_h(cells):
    mi = json.loads((ROOT / "results" / "universal-scaling-law" / "missing_info.json").read_text())
    for c in cells:
        c["h"] = 0.0 if c["alg"] in ("dictlearn-scaling", "als-ema") else mi[group_key(c)]["h"]
    return cells


FEATS = {
    "M1": lambda c: [math.log(c["n"] / c["k"]), math.log(c["B"]), math.log(c["T"])],
    "Q1": lambda c: [math.log(c["q"]), math.log(c["n"] / c["B"]), math.log(c["T"])],
    "Qe1": lambda c: [math.log(c["qe"]), math.log(c["n"] / c["B"]), math.log(c["T"])],
    "Qh1": lambda c: [math.log(c["qe"]), math.log(max(1.0 - c["h"], 1e-3)), math.log(c["n"] / c["B"]), math.log(c["T"])],
}
MODELS = {"M1": ("M1", True), "M1g": ("M1", False), "Q1": ("Q1", True), "Q0": ("Q1", False),
          "Qe1": ("Qe1", True), "Qe0": ("Qe1", False)}


def xrow(c, model, algs):
    f, per_alg = MODELS[model]
    oh = [1.0 if c["alg"] == a else 0.0 for a in algs] if per_alg else [1.0]
    return oh + FEATS[f](c)


def fit(cells, model, algs):
    X = np.array([xrow(c, model, algs) for c in cells])
    y = np.log([c["a_star"] for c in cells])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def predict(beta, c, model, algs, held_const=None):
    _, per_alg = MODELS[model]
    if per_alg and held_const is not None:
        f = FEATS[MODELS[model][0]](c)
        val = held_const + float(np.dot(beta[len(algs):], f))
    else:
        val = float(np.dot(beta, xrow(c, model, algs)))
    return min(max(math.exp(val), 1e-5), 1.0)


def penalty(c, a):
    return (interp_loss(c["curve"], a) - c["L_star"]) / max(abs(c["L_star"]), 1e-12)


def main():
    global OUT
    st = dataset_stats()
    cells = load_cells(st)
    mi = "--missing-info" in sys.argv
    if mi:
        cells = attach_h(cells)
        MODELS.update({"Qh1": ("Qh1", True), "Qh0": ("Qh1", False)})
        OUT = OUT.with_name("refit_coords_mi.json")
        for a in sorted({c["alg"] for c in cells}):
            hs = [c["h"] for c in cells if c["alg"] == a]
            print(f"  h {a:18s} median {np.median(hs):.3f}  range {min(hs):.3f}-{max(hs):.3f}")
    fitc = [c for c in cells if not c["edge"]]
    algs = sorted({c["alg"] for c in cells})
    print(f"cells {len(cells)}; censored at the lower edge (alpha*<=0.01): {len(cells) - len(fitc)}")
    print("\nper algorithm: cells, censored, median alpha*, median q, median q/alpha* (= observations per cell held"
          " in the EMA at peak rate)")
    for a in algs:
        cs = [c for c in cells if c["alg"] == a]
        print(f"  {a:18s} {len(cs):4d} {sum(c['edge'] for c in cs):4d}  a*={np.median([c['a_star'] for c in cs]):.3g}"
              f"  q={np.median([c['q'] for c in cs]):.3g}  q_eff={np.median([c['qe'] for c in cs]):.3g}"
              f"  q/a*={np.median([c['q'] / c['a_star'] for c in cs]):.3g}  q_eff/a*={np.median([c['qe'] / c['a_star'] for c in cs]):.3g}")
    res = {}
    fit_algs = sorted({c["alg"] for c in fitc})
    for model in ("KM", *MODELS):
        loco, loao, held_pred = [], defaultdict(list), defaultdict(list)
        for i, c in enumerate(cells):
            if model == "KM":
                loco.append(penalty(c, emeans_alpha0(int(c["n"]), int(c["k"]), c["T"], int(c["B"]))))
                continue
            train = [t for t in fitc if t is not c]
            if MODELS[model][1] and not any(t["alg"] == c["alg"] for t in train):
                continue
            loco.append(penalty(c, predict(fit(train, model, fit_algs), c, model, fit_algs)))
        if model != "KM":
            for held in algs:
                train = [c for c in fitc if c["alg"] != held]
                others = sorted({c["alg"] for c in train})
                beta = fit(train, model, others)
                const = float(np.mean(beta[: len(others)])) if MODELS[model][1] else None
                for c in (c for c in cells if c["alg"] == held):
                    a = predict(beta, c, model, others, held_const=const)
                    loao[held].append(penalty(c, a))
                    held_pred[held].append((a, c["a_grid"]))
        else:
            for c in cells:
                a = emeans_alpha0(int(c["n"]), int(c["k"]), c["T"], int(c["B"]))
                loao[c["alg"]].append(penalty(c, a))
                held_pred[c["alg"]].append((a, c["a_grid"]))
        allo = [p for v in loao.values() for p in v]
        res[model] = {"loco_median": float(np.median(loco)), "loco_mean": float(np.mean(loco)),
                      "loao_median": float(np.median(allo)), "loao_mean": float(np.mean(allo)),
                      "loao_by_alg": {a: float(np.median(v)) for a, v in loao.items()},
                      "pred_vs_grid_by_alg": {a: [float(np.median([p for p, _ in v])), float(np.median([g for _, g in v]))]
                                              for a, v in held_pred.items()}}
    print("\nval-loss penalty of the predicted alpha0 (median over cells; 0 = as good as the swept optimum)")
    print(f"  {'model':5s} {'LOCO':>8s} {'LOAO':>8s}   LOAO by held-out algorithm   [predicted / best grid alpha, medians]")
    for model, r in res.items():
        by = "  ".join(f"{a.split('-')[0]}={r['loao_by_alg'][a]:.2%}[{r['pred_vs_grid_by_alg'][a][0]:.3g}/"
                       f"{r['pred_vs_grid_by_alg'][a][1]:.3g}]" for a in algs)
        print(f"  {model:5s} {r['loco_median']:8.3%} {r['loao_median']:8.3%}   {by}")
    for model in ("M1g", "Q0", "Qe0") + (("Qh0",) if mi else ()):
        beta = fit(fitc, model, fit_algs)
        names = {"M1": ["log(n/k)", "log B", "log T"], "Q1": ["log q", "log(n/B)", "log T"],
                 "Qe1": ["log q_eff", "log(n/B)", "log T"],
                 "Qh1": ["log q_eff", "log(1-h)", "log(n/B)", "log T"]}[MODELS[model][0]]
        X = np.array([xrow(c, model, fit_algs) for c in fitc])
        y = np.log([c["a_star"] for c in fitc])
        r2 = 1 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
        res[model]["global_fit"] = {"const": float(beta[0]), **{nm: float(b) for nm, b in zip(names, beta[1:])}, "r2_log_alpha": float(r2)}
        print(f"  global fit {model}: alpha0* = {math.exp(beta[0]):.3g} " +
              " ".join(f"* exp({b:+.3f} {nm})" for nm, b in zip(names, beta[1:])) + f"   R^2(log a*) = {r2:.3f}")
    OUT.write_text(json.dumps({"n_cells": len(cells), "n_censored": len(cells) - len(fitc), "models": res}, indent=1))
    print(f"\nsaved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
