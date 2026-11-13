"""FROZEN kill criteria, transcribed from RESEARCH_PLAN.md section 3.

Committed before any v2 re-run. Do not edit thresholds after seeing results; if a criterion turns
out to be ill-posed, add a *new* criterion next to it and report both.

Each criterion is a function ``(summary, records) -> (killed: bool | None, reason: str)``.
``summary`` is the per-group output of ``scripts/evaluate.py:summarise_track``; ``records`` are the
track's raw v2 rows. ``None`` means "not enough data to decide" (reported as INCOMPLETE, never PASS).

Conventions used by every criterion:
* ``test_loss`` is lower-is-better.
* gain = 25 / epochs_to_match, computed on the seed-mean curve against the seed-mean reference@25.
  A method that never reaches the reference within its logged curve has gain 0.
* "beats" = lower seed-mean AND paired t-test (by seed) p < 0.05.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any, Callable

import numpy as np
from scipy import stats as sps

ALPHA = 0.05
FROZEN_AT = "2026-11-13"


# ---------------------------------------------------------------- helpers

def _paired(a: dict[int, float], b: dict[int, float]) -> tuple[float, float, int]:
    """Mean(a-b) and two-sided paired-t p over shared seeds."""
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return float("nan"), float("nan"), len(seeds)
    x = np.array([a[s] for s in seeds])
    y = np.array([b[s] for s in seeds])
    diff = x - y
    if np.allclose(diff, diff[0]):
        p = 0.0 if diff[0] != 0 else 1.0
    else:
        p = float(sps.ttest_rel(x, y).pvalue)
    return float(diff.mean()), p, len(seeds)


def beats(a: dict[int, float], b: dict[int, float]) -> bool:
    d, p, n = _paired(a, b)
    return n >= 2 and d < 0 and p < ALPHA


def final_by_seed(records, method: str, group: str, key: str = "test_loss") -> dict[int, float]:
    out = {}
    for r in records:
        if r["method"] == method and r["extra"]["group"] == group and r.get(key) is not None:
            out[r["seed"]] = float(r[key])
    return out


def best_gain_by_dataset(summary) -> dict[str, float]:
    best: dict[str, float] = defaultdict(float)
    for g in summary["groups"].values():
        for m in g["methods"].values():
            if m["role"] == "ours":
                best[g["dataset"]] = max(best[g["dataset"]], m["gain"] or 0.0)
    return dict(best)


def _gain_criterion(threshold: float) -> Callable:
    def crit(summary, records):
        gains = best_gain_by_dataset(summary)
        if not gains:
            return None, "no ours-vs-reference groups logged"
        med = median(gains.values())
        detail = ", ".join(f"{d}: {g:.2f}x" for d, g in sorted(gains.items()))
        return med < threshold, f"median best gain {med:.2f}x vs required {threshold}x ({detail})"
    return crit


# ---------------------------------------------------------------- per-track

def t01_gmm(summary, records):
    """Kill if: after the sweep, best variant needs >8 epochs to match EM@25 on 2 of 3 datasets."""
    best_etm: dict[str, float] = {}
    for g in summary["groups"].values():
        for m in g["methods"].values():
            if m["role"] != "ours":
                continue
            e = m["epochs_to_match"] if m["epochs_to_match"] is not None else float("inf")
            best_etm[g["dataset"]] = min(best_etm.get(g["dataset"], float("inf")), e)
    if len(best_etm) < 3:
        return None, f"need >=3 datasets, have {len(best_etm)}"
    slow = sum(e > 8 for e in best_etm.values())
    detail = ", ".join(f"{d}: {e:.2f}" for d, e in sorted(best_etm.items()))
    return slow / len(best_etm) >= 2 / 3, f"{slow}/{len(best_etm)} datasets need >8 epochs ({detail})"


def t02_opq(summary, records):
    """Kill if: joint online rotation is no better than e-means PQ + one final OPQ refit."""
    groups = sorted({r["extra"]["group"] for r in records})
    wins, total, rows = 0, 0, []
    for g in groups:
        a = final_by_seed(records, "joint_rotation", g)
        b = final_by_seed(records, "emeans_pq_posthoc_opq", g)
        if not a or not b:
            continue
        total += 1
        w = beats(a, b)
        wins += w
        rows.append(f"{g}: {'win' if w else 'no win'}")
    if total == 0:
        return None, "missing joint_rotation / emeans_pq_posthoc_opq pairs"
    return wins / total < 0.5, f"joint rotation significantly better in {wins}/{total} groups ({'; '.join(rows)})"


def _p99_at_recall(lat: list[dict], target: float) -> float | None:
    pts = sorted(lat, key=lambda p: p["recall10"])
    for lo, hi in zip(pts, pts[1:]):
        if lo["recall10"] <= target <= hi["recall10"]:
            if hi["recall10"] == lo["recall10"]:
                return hi["p99_us"]
            f = (target - lo["recall10"]) / (hi["recall10"] - lo["recall10"])
            return float(np.exp(np.log(lo["p99_us"]) + f * (np.log(hi["p99_us"]) - np.log(lo["p99_us"]))))
    if pts and pts[0]["recall10"] >= target:
        return pts[0]["p99_us"]
    return None


def t03_balanced_ivf(summary, records):
    """Kill if: p99 latency at recall 0.9 improves by <10% over e-means IVF."""
    by = defaultdict(lambda: defaultdict(list))
    for r in records:
        lat = r["extra"].get("latency")
        if lat and r["method"] in ("emeans_ivf", "balanced_ivf"):
            v = _p99_at_recall(lat, 0.9)
            if v is not None:
                by[r["extra"]["group"]][r["method"]].append(v)
    imps = {}
    for g, m in by.items():
        if m.get("emeans_ivf") and m.get("balanced_ivf"):
            imps[g] = 1.0 - float(np.mean(m["balanced_ivf"])) / float(np.mean(m["emeans_ivf"]))
    if not imps:
        return None, "no groups with p99@recall0.9 for both methods"
    med = median(imps.values())
    detail = ", ".join(f"{g}: {v:+.1%}" for g, v in sorted(imps.items()))
    return med < 0.10, f"median p99@R0.9 improvement {med:+.1%} vs required +10% ({detail})"


def t05_lda(summary, records):
    """Kill if: the fitted law does worse than default online LDA on 2 of 3 corpora."""
    corpora = defaultdict(lambda: [0, 0])
    for g in sorted({r["extra"]["group"] for r in records}):
        a = final_by_seed(records, "cosine_law", g)
        b = final_by_seed(records, "hoffman_default", g)
        if not a or not b:
            continue
        ds = next(r["dataset"] for r in records if r["extra"]["group"] == g)
        corpora[ds][1] += 1
        corpora[ds][0] += float(np.mean(list(a.values()))) > float(np.mean(list(b.values())))
    if len(corpora) < 3:
        return None, f"need >=3 corpora with cosine_law + hoffman_default, have {len(corpora)}"
    worse = sum(w > 0 for w, _ in corpora.values())
    return worse / len(corpora) >= 2 / 3, f"law worse than default on {worse}/{len(corpora)} corpora"


def t07_hmm(summary, records):
    """Kill if: gain < 2x or worse local optima than full Baum-Welch."""
    killed_gain, why = _gain_criterion(2.0)(summary, records)
    if killed_gain is None:
        return None, why
    worse = []
    for gname, g in summary["groups"].items():
        for mname, m in g["methods"].items():
            if m["role"] == "ours" and m.get("final_vs_ref_p") is not None:
                if m["final_vs_ref_diff"] > 0 and m["final_vs_ref_p"] < ALPHA:
                    worse.append(f"{gname}/{mname}")
    return killed_gain or bool(worse), f"{why}; significantly worse final optimum: {worse or 'none'}"


def t08_drift(summary, records):
    """Kill if: no scenario beats 'constant alpha tuned per scenario'."""
    scen = sorted({r["extra"]["group"] for r in records})
    wins = []
    decided = 0
    for g in scen:
        ours = final_by_seed(records, "rewarm", g)
        consts = sorted({r["method"] for r in records if r["extra"]["group"] == g and r["method"].startswith("const_alpha_")})
        if not ours or not consts:
            continue
        decided += 1
        best = min(consts, key=lambda m: np.mean(list(final_by_seed(records, m, g).values())))
        if beats(ours, final_by_seed(records, best, g)):
            wins.append(f"{g} (vs {best})")
    if decided == 0:
        return None, "no scenario with rewarm + const_alpha grid"
    return not wins, f"rewarm beats per-scenario-tuned constant alpha in: {wins or 'none'} of {decided}"


def t09_seeding(summary, records):
    """Kill if: nothing beats random init once seeding time is charged."""
    wins = []
    for g in sorted({r["extra"]["group"] for r in records}):
        base = final_by_seed(records, "random", g)
        if not base:
            continue
        for m in sorted({r["method"] for r in records if r["extra"]["group"] == g and r["method"] != "random"}):
            if beats(final_by_seed(records, m, g), base):
                wins.append(f"{g}/{m}")
    return not wins, f"budget-charged wins over random: {len(wins)} ({', '.join(wins[:8])}{'...' if len(wins) > 8 else ''})"


def t10_adaptive_k(summary, records):
    """Kill if: the selected k is unstable across seeds (CV > 20%)."""
    ks = defaultdict(list)
    for r in records:
        if r["extra"].get("k_selected") is not None and r["extra"]["role"] == "ours":
            ks[(r["extra"]["group"], r["method"])].append(float(r["extra"]["k_selected"]))
    if not ks:
        return None, "no k_selected logged"
    cvs = {k: float(np.std(v) / max(np.mean(v), 1e-12)) for k, v in ks.items() if len(v) >= 2}
    worst = max(cvs.values())
    return worst > 0.20, f"max CV of selected k {worst:.1%} (threshold 20%)"


def t11_hnsw(summary, records):
    """Kill if: the law loses to library defaults on held-out datasets."""
    held = [r for r in records if r["extra"].get("held_out")]
    pairs = defaultdict(dict)
    for r in held:
        pairs[(r["dataset"], r["extra"]["target_recall"])][r["method"]] = r
    wins, total = 0, 0
    for key, ms in pairs.items():
        law = ms.get("law")
        defs = [v for k, v in ms.items() if k.startswith("default_")]
        if law is None or not defs:
            continue
        total += 1
        t = key[1]
        law_ok = law["extra"]["recall10"] >= t
        best_def = max((d for d in defs if d["extra"]["recall10"] >= t), key=lambda d: d["extra"]["qps"], default=None)
        wins += law_ok and (best_def is None or law["extra"]["qps"] >= best_def["extra"]["qps"])
    if total == 0:
        return None, "no held-out law/default pairs"
    return wins / total < 0.5, f"law wins {wins}/{total} held-out (dataset, target-recall) pairs"


def t00_acceptance(summary, records):
    """T00 done-criterion: e-means beats Lloyd@25 training loss within <=3 epochs on SIFT1M and >=1
    embedding set (paper: ~1.4-2.6 epochs). 'Killed' here means the reproduction FAILED."""
    ok = {}
    for gname, g in summary["groups"].items():
        if not gname.endswith("|train"):
            continue
        m = g["methods"].get("emeans")
        if m is None:
            continue
        e = m["epochs_to_match"]
        ok[gname] = e is not None and e <= 3.0
    if not any(k.startswith("sift1m") for k in ok) or not any(not k.startswith("sift1m") for k in ok):
        return None, f"need SIFT1M and an embedding set; have {sorted(ok)}"
    sift_ok = all(v for k, v in ok.items() if k.startswith("sift1m"))
    emb_ok = any(v for k, v in ok.items() if not k.startswith("sift1m"))
    return not (sift_ok and emb_ok), "; ".join(f"{k}: {'<=3 ep' if v else 'NOT <=3 ep'}" for k, v in sorted(ok.items()))


# ---------------------------------------------------------------- amendments
# Amendments never replace a frozen criterion: both are evaluated and reported side by side.
AMENDMENTS_LOG = [
    ("2026-11-13", "hmm-ema", "t07 as frozen applies 'worse local optimum' to EVERY ours method incl. the untuned "
     "k-means-law variant; the plan's intent is the method under test. Added t07_best_variant (gain and optimum "
     "judged on the best ours method per dataset). Added after seeing partial synthetic results."),
]


def t07_hmm_best_variant(summary, records):
    """Amendment: gain < 2x, or the BEST ours method per group has a significantly worse final optimum."""
    killed_gain, why = _gain_criterion(2.0)(summary, records)
    if killed_gain is None:
        return None, why
    worse = []
    for gname, g in summary["groups"].items():
        ours = {k: m for k, m in g["methods"].items() if m["role"] == "ours" and m["final_mean"] is not None}
        if not ours:
            continue
        bname, best = min(ours.items(), key=lambda kv: kv[1]["final_mean"])
        if best.get("final_vs_ref_p") is not None and best["final_vs_ref_diff"] > 0 and best["final_vs_ref_p"] < ALPHA:
            worse.append(f"{gname}/{bname}")
    return killed_gain or bool(worse), f"{why}; best variant significantly worse: {worse or 'none'}"


def t12_universal(summary, records):
    """Pre-registered 2026-11-13, before any law-sweep data existed. 'Killed' = universal law NOT supported:
    M1 must beat M0 in LOCO (median lower AND paired Wilcoxon p<0.05) and in LOAO (median lower)."""
    by = {(r["extra"]["group"], r["method"]): r["extra"]["penalties"] for r in records}
    need = [("loco", "M0"), ("loco", "M1"), ("loao", "M0"), ("loao", "M1")]
    if any(k not in by or len(by[k]) < 5 for k in need):
        return None, "need LOCO and LOAO penalties for M0 and M1 (>=5 cells each)"
    m0, m1 = np.array(by[("loco", "M0")]), np.array(by[("loco", "M1")])
    p = float(sps.wilcoxon(m1, m0, alternative="less").pvalue) if len(m0) == len(m1) and np.any(m1 != m0) else 1.0
    loco_ok = np.median(m1) < np.median(m0) and p < ALPHA
    loao_ok = np.median(by[("loao", "M1")]) < np.median(by[("loao", "M0")])
    return not (loco_ok and loao_ok), (f"LOCO median M1 {np.median(m1):.3%} vs M0 {np.median(m0):.3%} (Wilcoxon p={p:.3g}); "
                                       f"LOAO median M1 {np.median(by[('loao', 'M1')]):.3%} vs M0 {np.median(by[('loao', 'M0')]):.3%}")


SECONDARY: dict[str, list[tuple[str, Callable]]] = {
    "hmm-ema": [("[amendment 2026-11-13] gain < 2x or BEST variant has worse optimum", t07_hmm_best_variant)],
}


CRITERIA: dict[str, tuple[str, Callable]] = {
    "core": ("REPRODUCTION FAILS: e-means not <=3 epochs to Lloyd@25 train loss on SIFT1M + an embedding set", t00_acceptance),
    "gmm-ema": ("best variant >8 epochs to match EM@25 on 2 of 3 datasets", t01_gmm),
    "opq-joint-rotation": ("joint rotation no better than e-means PQ + final OPQ refit", t02_opq),
    "balanced-ivf": ("p99 latency at recall 0.9 improves <10% over e-means IVF", t03_balanced_ivf),
    "dictlearn-scaling": ("gain < 2x in epochs-to-match", _gain_criterion(2.0)),
    "lda-scaling": ("fitted law worse than default online LDA on 2 of 3 corpora", t05_lda),
    "als-ema": ("gain < 2x in epochs-to-match", _gain_criterion(2.0)),
    "hmm-ema": ("gain < 2x or worse local optima than Baum-Welch", t07_hmm),
    "drift-rewarm": ("no scenario beats constant alpha tuned per scenario", t08_drift),
    "cheap-seeding": ("nothing beats random init once seeding time is charged", t09_seeding),
    "adaptive-k": ("selected k unstable across seeds (CV > 20%)", t10_adaptive_k),
    "hnsw-scaling-laws": ("law loses to library defaults on held-out datasets", t11_hnsw),
    "universal-scaling-law": ("UNIVERSAL LAW NOT SUPPORTED: M1 fails to beat per-algorithm-constant null (LOCO+LOAO)", t12_universal),
}
