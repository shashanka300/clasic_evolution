# Statistics added in response to the methodology review (2026-11-17). Run from the repo root.
#
# 1. Multiple comparisons: the 102 paired t-tests of recipe (*_swept) vs tuned stepwise + revival (*_rev_tuned)
#    (17 groups x T in {1,2,3,5,10,25}) with Holm (FWER 5%) and Benjamini-Hochberg (FDR 5%) corrections, plus a
#    distribution-free count: cells where every seed favours the same method (exact sign-flip p = 2/32 = 0.0625
#    with 5 seeds, so no 5-seed nonparametric test can reach p < 0.05 on its own).
# 2. Reference definition: epochs to reach (a) full-batch@25 (the paper's reference) and (b) the best full-batch
#    iterate within 25 iterations (seed-mean curve minimum), for every headline method of all 19 groups.
#
# Output: results/review_stats.json and a printed summary.
import json, math, collections, sys
import numpy as np
from scipy import stats

sys.path.insert(0, "scripts")
from evaluate import _curve, _mean_curve, epochs_to_match  # noqa: E402

PAIRS = {"gmm-ema": ("gmm_ema_swept", "online_em_rev_tuned"),
         "hmm-ema": ("hmm_ema_swept", "stepwise_em_rev_tuned"),
         "lda-scaling": ("lda_ema_swept", "online_lda_rev_tuned"),
         "dictlearn-scaling": ("dl_ema_swept", "stepwise_dl_rev_tuned")}
METHODS = {"gmm-ema": ["gmm_ema_swept", "gmm_ema_kmlaw", "online_em_tuned", "online_em_rev_tuned"],
           "hmm-ema": ["hmm_ema_swept", "hmm_ema_kmlaw", "stepwise_em_tuned", "stepwise_em_rev_tuned"],
           "lda-scaling": ["lda_ema_swept", "lda_ema_kmlaw", "online_lda_tuned", "online_lda_rev_tuned"],
           "dictlearn-scaling": ["dl_ema_swept", "dl_ema_kmlaw", "stepwise_dl_tuned", "stepwise_dl_rev_tuned"],
           "als-ema": ["als_ema_swept", "als_ema_kmlaw", "stepwise_als_tuned"]}
EP = [1, 2, 3, 5, 10, 25]


def load(tr):
    recs = []
    for l in open(f"results/{tr}/runs.jsonl"):
        r = json.loads(l)
        if "group" in r["extra"]:
            recs.append(r)
    return recs


def holm(ps, alpha=0.05):
    order = np.argsort(ps)
    rej = np.zeros(len(ps), bool)
    for i, j in enumerate(order):
        if ps[j] <= alpha / (len(ps) - i):
            rej[j] = True
        else:
            break
    return rej


def bh(ps, alpha=0.05):
    order = np.argsort(ps)
    m = len(ps)
    k = 0
    for i, j in enumerate(order, 1):
        if ps[j] <= alpha * i / m:
            k = i
    rej = np.zeros(m, bool)
    rej[order[:k]] = True
    return rej


out = {"tests": [], "reference": {}}
data = {tr: load(tr) for tr in METHODS}

# ---- 1. multiple comparisons ---------------------------------------------------------------------------------------
for tr, (a, b) in PAIRS.items():
    cur = collections.defaultdict(dict)
    for r in data[tr]:
        if r["method"] in (a, b):
            cur[(r["extra"]["group"], r["method"])][r["seed"]] = _curve(r)
    for g in sorted({g for g, _ in cur}):
        for e in EP:
            A, B = cur[(g, a)], cur[(g, b)]
            seeds = [s for s in A if s in B and e in A[s] and e in B[s]]
            d = np.array([A[s][e] - B[s][e] for s in seeds])
            ref = float(np.mean([B[s][e] for s in seeds]))
            if np.all(d == 0):
                p = 1.0
            else:
                p = float(stats.ttest_1samp(d, 0.0).pvalue)
            out["tests"].append({"track": tr, "group": g, "T": e, "n": len(d), "rel_diff_pct": 100 * d.mean() / ref,
                                 "p": p, "all_seeds_same_sign": bool(np.all(d < 0) or np.all(d > 0)),
                                 "sign": int(np.sign(d.mean()))})

ps = np.array([t["p"] for t in out["tests"]])
for name, rej in (("raw", ps < 0.05), ("holm", holm(ps)), ("bh", bh(ps))):
    for t, r in zip(out["tests"], rej):
        t[name] = bool(r)
fam = {"gmm-ema": "GMM", "hmm-ema": "HMM", "dictlearn-scaling": "DL", "lda-scaling": "LDA"}
summ = collections.defaultdict(lambda: collections.Counter())
for t in out["tests"]:
    f = fam[t["track"]]
    for name in ("raw", "holm", "bh"):
        summ[(f, name)]["W" if t[name] and t["sign"] < 0 else "L" if t[name] and t["sign"] > 0 else "tie"] += 1
    summ[(f, "sign5")]["W" if t["all_seeds_same_sign"] and t["sign"] < 0 else
                       "L" if t["all_seeds_same_sign"] and t["sign"] > 0 else "mixed"] += 1
out["summary"] = {f"{f}|{n}": dict(c) for (f, n), c in summ.items()}
print(f"{len(ps)} tests")
for f in ("GMM", "HMM", "DL", "LDA"):
    print(f, {n: dict(summ[(f, n)]) for n in ("raw", "holm", "bh", "sign5")})
for n in ("raw", "holm", "bh"):
    W = sum(1 for t in out["tests"] if t[n] and t["sign"] < 0)
    L = sum(1 for t in out["tests"] if t[n] and t["sign"] > 0)
    print(f"total {n}: W {W} / tie {len(ps) - W - L} / L {L}")
W = sum(1 for t in out["tests"] if t["all_seeds_same_sign"] and t["sign"] < 0)
L = sum(1 for t in out["tests"] if t["all_seeds_same_sign"] and t["sign"] > 0)
print(f"all seeds agree: recipe better {W}, worse {L}, mixed {len(ps) - W - L}")

# ---- 2. reference definition ---------------------------------------------------------------------------------------
for tr, meths in METHODS.items():
    groups = collections.defaultdict(list)
    for r in data[tr]:
        groups[r["extra"]["group"]].append(r)
    for g, recs in sorted(groups.items()):
        ref_curves = [_curve(r) for r in recs if r["extra"].get("reference")]
        rc, _ = _mean_curve(ref_curves)
        r25 = rc[25.0]
        best_it = min(rc, key=rc.get)
        rbest = rc[best_it]
        row = {"ref25": r25, "ref_best": rbest, "best_iter": best_it,
               "ref_rise_pct": 100 * (r25 - rbest) / abs(rbest), "methods": {}}
        for m in meths:
            cs = [_curve(r) for r in recs if r["method"] == m]
            cs = [c for c in cs if c]
            if not cs:
                continue
            mc, _ = _mean_curve(cs)
            row["methods"][m] = {"etm_ref25": epochs_to_match(mc, r25), "etm_best": epochs_to_match(mc, rbest),
                                 "n_seeds": len(cs)}
        out["reference"][f"{tr}|{g}"] = row

print("\ngroup | best iter | ref25 vs best | etm swept (ref25 -> best) | rev_tuned/tuned (ref25 -> best)")
f = lambda v: "  -  " if v is None else f"{v:5.2f}"
for k, row in out["reference"].items():
    ms = list(row["methods"])
    sw = row["methods"][ms[0]]
    base = row["methods"][ms[-1]]
    print(f"{k:42s} {row['best_iter']:5.0f} {row['ref_rise_pct']:+7.3f}%  {f(sw['etm_ref25'])} -> {f(sw['etm_best'])}"
          f"   {f(base['etm_ref25'])} -> {f(base['etm_best'])}")
json.dump(out, open("results/review_stats.json", "w"), indent=1)
