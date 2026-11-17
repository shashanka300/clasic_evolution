"""Diagnostic (2026-11-16, not pre-registered): is the cosine floor eps=0.1 too high for LDA at long budgets?

The recipe's held-out perplexity rises from T=5 to T=25 on all three corpora, and tuned online LDA with revival
(inverse-time decay) ends lower on Enron/NYTimes at T=25 (PAPER_LOG §4). This re-runs the headline cells
(k=50, B=256, full n) with eps=0.01 on the long-budget alpha0 set of each corpus (long_budget_plan.json) plus one
value above it, seeds 0-2, T in {5, 25}. The eps=0.1 counterpart of every (alpha0, T, seed) is read from sweep.jsonl.
Records go to results/lda-scaling/eps_diag.jsonl (not used by evaluate.py).

  .venv/bin/python -u tracks/lda-scaling/eps_diag.py --corpus enron --T 25 --eps 0.01 --seeds 0,1,2
  .venv/bin/python -u tracks/lda-scaling/eps_diag.py --report
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.schedule import CosineSchedule  # noqa: E402
from run_v2 import CORPORA, HEAD_B, SWEEP, make_problem, results_path, TRACK  # noqa: E402

OUT = results_path(TRACK, "eps_diag.jsonl")
PLAN = results_path(TRACK, "long_budget_plan.json")
GROUP = {"20ng": "20ng|n=12170|k=50|B=256", "enron": "enron|n=35518|k=50|B=256",
         "nytimes": "nytimes|n=95543|k=50|B=256"}
UP = {0.00139: 0.00268, 0.00268: 0.00518, 0.00518: 0.01, 0.01: 0.0193, 0.0193: 0.0373, 0.0373: 0.072}


DOWN = {v: k for k, v in UP.items()}


def alphas(corpus, below=False):
    a = json.load(open(PLAN))[f"alpha|lda_ema|{GROUP[corpus]}"]["alphas"]
    if below:  # one step below the long-budget grid (its optimum sat at the lower edge for 20NG at T=25)
        return [DOWN[min(a)]]
    return sorted(set(a) | {UP[max(a)]})


def done():
    if not Path(OUT).exists():
        return set()
    return {(r["corpus"], r["T"], r["eps"], r["alpha0"], r["seed"]) for r in map(json.loads, open(OUT))}


def run(corpus, T, eps, seeds, below=False):
    p = make_problem(corpus)
    have = done()
    for a in alphas(corpus, below):
        for s in seeds:
            if (corpus, T, eps, a, s) in have:
                continue
            steps = max(int(round(T * max(p.n // HEAD_B, 1))), 1)
            t0 = time.perf_counter()
            m = p.fit(a, T, HEAD_B, s, schedule=CosineSchedule(a, steps, eps=eps), revival=True)
            rec = {"corpus": corpus, "T": T, "eps": eps, "alpha0": a, "seed": s, "val": p.loss(m, "val"),
                   "test": p.loss(m, "test"), "wall_s": time.perf_counter() - t0}
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec, flush=True)


def report():
    rows = [json.loads(l) for l in open(OUT)]
    base = {}
    for l in open(SWEEP):
        r = json.loads(l)
        if r["method"] == "lda_ema" and r["extra"].get("sweep") and r["extra"]["group"] in GROUP.values():
            c = r["extra"]["group"].split("|")[0]
            base[(c, float(r["epochs"]), 0.1, r["alpha0"], r["seed"])] = (r["extra"]["val_loss"], r["test_loss"])
    for r in rows:  # eps=0.1 runs made here (alphas outside the sweep) join the sweep's rows
        if r["eps"] == 0.1:
            base[(r["corpus"], float(r["T"]), 0.1, r["alpha0"], r["seed"])] = (r["val"], r["test"])
    cells = {}
    for r in rows:
        if r["eps"] != 0.1:
            cells.setdefault((r["corpus"], float(r["T"]), r["eps"]), {})[(r["alpha0"], r["seed"])] = (r["val"], r["test"])
    print("corpus  T   eps    best a0 (val)   test mean over seeds   [per-alpha val means]  (alphas present for both eps)")
    for (c, T, eps), d in sorted(cells.items()):
        d = {k: v for k, v in d.items() if (c, T, 0.1, *k) in base}
        seeds = sorted({s for _, s in d})
        for e, src in ((0.1, {k: base[(c, T, 0.1, *k)] for k in d}), (eps, d)):
            by_a = {}
            for (a, s), (v, t) in src.items():
                by_a.setdefault(a, []).append((v, t))
            best = min(by_a, key=lambda a: st.mean(v for v, _ in by_a[a]))
            print(f"{c:8s}{T:4g} {e:<6g} {best:<10g} test {st.mean(t for _, t in by_a[best]):8.1f}   "
                  + " ".join(f"{a:g}:{st.mean(v for v, _ in by_a[a]):.0f}" for a in sorted(by_a)) + f"  seeds={seeds}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=CORPORA)
    ap.add_argument("--T", type=float)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--below", action="store_true", help="run only the alpha one step below the long-budget grid")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        run(a.corpus, a.T, a.eps, [int(x) for x in a.seeds.split(",")], a.below)
