"""Protocol-v3 budgets: frozen restricted grids for the long budgets (core/budgets.py)."""

import json
import math

from core.budgets import BUDGETS, EXT_GRID, LONG_BUDGETS, long_alphas, long_stepwise, plan_path
from core.sweep import ALPHA_GRID, run_sweep, sweep_group


def test_budget_tuple_and_grid():
    assert BUDGETS == (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 25.0)
    assert set(LONG_BUDGETS) == {10.0, 25.0}
    assert set(ALPHA_GRID) <= set(EXT_GRID)
    assert {0.00518, 0.00268, 0.00139} <= set(EXT_GRID)  # the values tracks/lda-scaling/extend_grid.py wrote


def _sweep(tmp_path, best):
    path = tmp_path / "sweep.jsonl"
    run_sweep(track="t", method="m", dataset="d", data={"train": list(range(100))}, k=2, d=1,
              fit_eval=lambda x, k, a, T, B, s: {"val": (math.log(a) - math.log(best)) ** 2 + 0.01 * s, "test": 1.0},
              configs=[{"n": 100, "B": 10, "T": 5.0}], seeds=[0, 1], path=path, metric="x")
    return path


def test_long_alphas_centre_and_freeze(tmp_path):
    path = _sweep(tmp_path, 0.0373)
    g = sweep_group("d", 100, 2, 10)
    al = long_alphas(path, "m", g)
    assert al == [0.01, 0.0193, 0.0373, 0.072]  # two below, one above the T=5 optimum
    plan = json.loads(plan_path(path).read_text())
    plan[f"alpha|m|{g}"]["alphas"] = [0.5]  # frozen: later calls read the plan, never recompute
    plan_path(path).write_text(json.dumps(plan))
    assert long_alphas(path, "m", g) == [0.5]


def test_long_alphas_at_top_and_bottom(tmp_path):
    assert long_alphas(_sweep(tmp_path, 1.0), "m", sweep_group("d", 100, 2, 10)) == [0.2683, 0.5179, 1.0]
    (tmp_path / "b").mkdir()
    assert long_alphas(_sweep(tmp_path / "b", 0.01), "m", sweep_group("d", 100, 2, 10))[-2:] == [0.01, 0.0193]


def test_long_stepwise_top3(tmp_path):
    rows = [{"seed": s, "extra": {"tau0": t0, "kappa": kp, "val_loss": v + 0.1 * s}}
            for s in (0, 1) for (t0, kp, v) in [(2, .5, 3.0), (10, .7, 1.0), (100, .9, 2.0), (2, 1.0, 0.5), (10, .5, 9.0)]]
    assert long_stepwise(tmp_path / "sweep.jsonl", "stepwise|g", rows) == [(2, 1.0), (10, .7), (100, .9)]
