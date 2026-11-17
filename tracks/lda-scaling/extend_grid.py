"""Extend the lda_ema alpha0 grid below 0.01 for sweep cells whose optimum sits at the grid's lower edge.

The shared grid (core.sweep.ALPHA_GRID) is 8 log-spaced values in [0.01, 1]; a third of the LDA cells pick 0.01,
so their optimum is censored. This continues the same log spacing downwards (0.00518, 0.00268, 0.00139, then
0.00072, 0.000373 if still needed), cell by cell, only where the lowest tried alpha is still the best one, with the
seeds the cell already has. Records go to the same sweep.jsonl in the same format, so sweep_cells() merges them.

  .venv/bin/python -u tracks/lda-scaling/extend_grid.py --datasets enron [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
import zlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.harness import read_runs  # noqa: E402
from core.sweep import run_sweep, sweep_cells  # noqa: E402
from run_v2 import CORPORA, SWEEP, TRACK, make_problem  # noqa: E402

STEP = 10 ** (2 / 7)  # the ALPHA_GRID spacing
LOWER = [float(f"{0.01 / STEP ** j:.3g}") for j in range(1, 6)]  # 0.00518 .. 0.000373
METHOD = "lda_ema"


def edge_cells(name):
    out = []
    for (g, T, m), c in sweep_cells(SWEEP, METHOD).items():
        if g.split("|")[0] != name:
            continue
        lo = min(c["val_curve"])
        if c["alpha_star_grid"] == lo and lo > LOWER[-1]:
            kv = dict(p.split("=", 1) for p in g.split("|")[1:] if "=" in p)
            out.append({"group": g, "n": int(kv["n"]), "k": int(kv["k"]), "B": int(kv["B"]), "T": float(T),
                        "seeds": c["seeds"], "lo": lo})
    return out


def est_seconds(cells):
    wall = defaultdict(list)
    for r in read_runs(SWEEP):
        if r["method"] == METHOD and r["extra"].get("sweep") and r["alpha0"] == 0.01:
            wall[(r["extra"]["group"], r["epochs"])].append(r["wall_s"])
    return sum(len(c["seeds"]) * (sum(wall[(c["group"], c["T"])]) / max(len(wall[(c["group"], c["T"])]), 1))
               for c in cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(CORPORA))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shard", default="0/1", help="i/N: only cells whose stable hash mod N == i (parallel workers)")
    args = ap.parse_args()
    si, sn = (int(x) for x in args.shard.split("/"))
    for name in args.datasets.split(","):
        p = None
        while True:
            cells = [c for c in edge_cells(name) if zlib.crc32(f"{c['group']}|{c['T']}".encode()) % sn == si]
            if not cells:
                print(f"[extend] {name}: no cell at its lowest alpha; done", flush=True)
                break
            print(f"[extend] {name}: {len(cells)} cells at their lowest alpha (next step ~{est_seconds(cells) / 60:.0f} "
                  f"min at alpha=0.01 cost): " + ", ".join(f"{c['group']}@T={c['T']:g}<= {c['lo']:g}" for c in cells),
                  flush=True)
            if args.dry_run:
                break
            p = p or make_problem(name)

            def fit_eval(x, kk, a0, T, B, s):
                m = p.fit(a0, T, B, s, n=len(x), k=kk)
                return {"val": p.loss(m, "val"), "test": p.loss(m, "test"), **p.extras(m)}

            for c in cells:
                nxt = [a for a in LOWER if a < c["lo"]][:1]
                run_sweep(track=TRACK, method=METHOD, dataset=name, data={"train": list(range(p.n))}, k=c["k"],
                          d=p.d, fit_eval=fit_eval, configs=[{"n": c["n"], "B": c["B"], "T": c["T"]}],
                          seeds=c["seeds"], path=SWEEP, metric=p.metric, alphas=nxt, tag=p.tag,
                          extra_config=p.config)


if __name__ == "__main__":
    main()
