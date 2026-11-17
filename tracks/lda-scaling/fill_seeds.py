"""Complete lda_ema sweep cells to every seed: run the (alpha0, seed) pairs a cell is missing.

The grid extension below alpha0=0.01 (extend_grid.py, 2026-11-14) ran on the v2 seeds 0-2 only. Protocol v3 adds
seeds 3-4, and select_alpha() keeps only seeds that have every alpha of a cell, so cells holding extension values
silently dropped seeds 3-4. This runs the missing pairs (same config, same record format; run_sweep skips pairs that
are already recorded), so every cell is scored on all its seeds. See PAPER_LOG.md §2.

  .venv/bin/python -u tracks/lda-scaling/fill_seeds.py --datasets enron,nytimes [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.harness import read_runs  # noqa: E402
from core.sweep import run_sweep  # noqa: E402
from run_v2 import CORPORA, SWEEP, TRACK, make_problem  # noqa: E402

METHOD = "lda_ema"


def gaps(name):
    """[(group, T, alpha0, [missing seeds])] for one corpus."""
    cells = defaultdict(lambda: defaultdict(set))
    for r in read_runs(SWEEP):
        if r["method"] == METHOD and r["extra"].get("sweep") and r["extra"]["group"].split("|")[0] == name:
            cells[(r["extra"]["group"], r["epochs"])][r["alpha0"]].add(r["seed"])
    out = []
    for (g, T), by_a in sorted(cells.items()):
        allseeds = set().union(*by_a.values())
        for a, s in sorted(by_a.items()):
            if allseeds - s:
                out.append((g, T, a, sorted(allseeds - s)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(CORPORA))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for name in args.datasets.split(","):
        todo = gaps(name)
        print(f"[fill] {name}: {len(todo)} (cell, alpha) gaps: " + ", ".join(f"{g.split('|')[3]}@T={T:g} a={a:g} seeds={s}"
                                                                         for g, T, a, s in todo), flush=True)
        if args.dry_run or not todo:
            continue
        p = make_problem(name)

        def fit_eval(x, kk, a0, T, B, s):
            m = p.fit(a0, T, B, s, n=len(x), k=kk)
            return {"val": p.loss(m, "val"), "test": p.loss(m, "test"), **p.extras(m)}

        for g, T, a, seeds in todo:
            kv = dict(x.split("=", 1) for x in g.split("|")[1:] if "=" in x)
            run_sweep(track=TRACK, method=METHOD, dataset=name, data={"train": list(range(p.n))}, k=int(kv["k"]),
                      d=p.d, fit_eval=fit_eval, configs=[{"n": int(kv["n"]), "B": int(kv["B"]), "T": float(T)}],
                      seeds=seeds, path=SWEEP, metric=p.metric, alphas=[a], tag=p.tag, extra_config=p.config)
        print(f"[fill] {name}: remaining gaps {len(gaps(name))}", flush=True)


if __name__ == "__main__":
    main()
