"""Epoch budgets, protocol v3 (2026-11-14). See PAPER_LOG.md §2 and §5.

v2 used budgets (1, 2, 3, 5): any method needing more than 5 epochs scored gain 0 (so the smallest measurable gain
was 5x while the kill criteria test 2x), and methods that already match the reference at epoch 1 could not be told
apart ("25x" ceiling). v3 adds sub-epoch budgets 0.25 and 0.5 and long budgets 10 and 25 (RESEARCH_PLAN rule 4).

Short budgets (<= 5) keep the full alpha0 grid and the full stepwise (tau0, kappa) grid. Long budgets (10, 25) use a
restricted search frozen in a per-track plan file before any long run starts (``long_budget_plan.json`` next to
the sweep file): the four alpha0 grid values around the T=5 optimum (two below, one above; the optimum falls with T),
and the three best (tau0, kappa) settings at T=5 for the tuned baseline. Fitting code treats an optimum at either
edge of a restricted grid as censored.
"""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from core.harness import read_runs
from core.sweep import ALPHA_GRID, sweep_cells

BUDGETS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 25.0)
LONG_BUDGETS = (10.0, 25.0)
SHORT_BUDGETS = tuple(T for T in BUDGETS if T not in LONG_BUDGETS)
ANCHOR_T = 5.0
# ALPHA_GRID continued downwards with the same log spacing (values as written by tracks/lda-scaling/extend_grid.py)
EXT_GRID = tuple(sorted(set(ALPHA_GRID) | {float(f"{0.01 / 10 ** (2 * j / 7):.3g}") for j in range(1, 6)}))
N_LONG_STEPWISE = 3


def active(budgets=BUDGETS) -> tuple[float, ...]:
    """Budgets this process works on: CLASSIC_ALGOS_BUDGETS="10" (comma list) splits a group's cells over workers."""
    env = os.environ.get("CLASSIC_ALGOS_BUDGETS")
    if not env:
        return tuple(budgets)
    sel = {float(x) for x in env.split(",")}
    return tuple(T for T in budgets if T in sel)


def plan_path(sweep_path) -> Path:
    return Path(sweep_path).with_name("long_budget_plan.json")


@contextmanager
def _locked_plan(sweep_path):
    path = plan_path(sweep_path)
    lock = path.with_suffix(".lock")
    with open(lock, "a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        plan = json.loads(path.read_text()) if path.exists() else {}
        n0 = len(plan)
        yield plan
        if len(plan) != n0:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(plan, indent=1, sort_keys=True))
            tmp.replace(path)
        fcntl.flock(lf, fcntl.LOCK_UN)


def long_alphas(sweep_path, method: str, sgroup: str) -> list[float]:
    """Frozen restricted alpha0 grid for the long budgets of one sweep group."""
    key = f"alpha|{method}|{sgroup}"
    with _locked_plan(sweep_path) as plan:
        if key not in plan:
            cell = sweep_cells(sweep_path, method).get((sgroup, ANCHOR_T, method))
            if cell is None:
                raise RuntimeError(f"no T={ANCHOR_T} sweep cell for {sgroup}: run the short budgets first")
            a5 = cell["alpha_star_grid"]
            i = int(np.argmin([abs(np.log(a / a5)) for a in EXT_GRID]))
            plan[key] = {"alphas": list(EXT_GRID[max(0, i - 2): i + 2]), "anchor_alpha": a5,
                         "anchor_seeds": cell["seeds"]}
        return plan[key]["alphas"]


def long_stepwise(sweep_path, sgroup: str, rows_at_anchor: list[dict]) -> list[tuple[float, float]]:
    """Frozen top-N (tau0, kappa) settings at T=5 (mean val over the seeds present when frozen)."""
    key = f"stepwise|{sgroup}"
    with _locked_plan(sweep_path) as plan:
        if key not in plan:
            cands: dict[tuple, list] = {}
            for r in rows_at_anchor:
                cands.setdefault((r["extra"]["tau0"], r["extra"]["kappa"]), []).append(r["extra"]["val_loss"])
            if not cands:
                raise RuntimeError(f"no T={ANCHOR_T} stepwise rows for {sgroup}: run the short budgets first")
            top = sorted(cands, key=lambda c: float(np.mean(cands[c])))[:N_LONG_STEPWISE]
            plan[key] = {"configs": [list(c) for c in top], "anchor_seeds": sorted({r["seed"] for r in rows_at_anchor})}
        return [tuple(c) for c in plan[key]["configs"]]


def anchor_rows(sweep_path, sgroup: str, flag: str) -> list[dict]:
    return [r for r in read_runs(sweep_path)
            if r["extra"].get(flag) and r["extra"]["group"] == sgroup and r["epochs"] == ANCHOR_T]
