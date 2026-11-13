"""Harness + evaluator: verdicts must come from data, and must be right."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from core.harness import make_run, read_runs, write_run  # noqa: E402
import evaluate as ev  # noqa: E402
from kill_criteria import CRITERIA  # noqa: E402


def _run(method, seed, curve, role="ours", reference=False, group="g", dataset="ds"):
    return make_run(
        track="t", method=method, dataset=dataset, n=100, d=2, k_or_units=4, epochs=max(e for e, _ in curve),
        batch=10, alpha0=0.1, seed=seed, train_loss=None, test_loss=curve[-1][1], wall_s=1.0,
        role=role, alpha0_source="fixed" if role == "ours" else "na", group=group, metric="mse",
        reference=reference, curve=[{"epoch": e, "test_loss": v, "wall_s": e} for e, v in curve],
    )


def test_schema_validation_rejects_bad_role():
    with pytest.raises(ValueError):
        make_run(track="t", method="m", dataset="d", n=1, d=1, k_or_units=1, epochs=1, batch=1, alpha0=None,
                 seed=0, train_loss=None, test_loss=1.0, wall_s=0, role="winner", alpha0_source="na",
                 group="g", metric="mse")


def test_roundtrip(tmp_path):
    p = tmp_path / "runs.jsonl"
    write_run(p, _run("a", 0, [(1, 2.0)]))
    (rec,) = read_runs(p)
    assert rec["extra"]["schema"] == 2 and rec["extra"]["config_hash"]


def test_epochs_to_match_interpolates():
    assert ev.epochs_to_match({1: 10.0, 2: 6.0, 3: 4.0}, 5.0) == pytest.approx(2.5)
    assert ev.epochs_to_match({1: 10.0, 2: 6.0}, 5.0) is None
    assert ev.epochs_to_match({2: 1.0, 3: 0.5}, 5.0) == 2  # never extrapolates below first point


def test_summary_and_gain_kill():
    grid = [1, 2, 3, 5, 10, 25]
    recs = []
    for s in range(3):
        recs.append(_run("lloyd", s, [(e, 10.0 / e + 1.0 + 0.01 * s) for e in grid], role="baseline", reference=True))
        # reaches 1.4 (=ref@25) at epoch ~ 2.5 -> gain ~10x
        recs.append(_run("fast", s, [(e, 5.0 / e - 0.6 + 0.01 * s) for e in grid]))
        recs.append(_run("slow", s, [(e, 10.0 / e + 1.2) for e in grid]))
    summ = ev.summarise_track(recs)
    m = summ["groups"]["g"]["methods"]
    assert m["fast"]["gain"] > 9
    assert m["slow"]["gain"] == 0.0 and m["slow"]["epochs_to_match"] is None
    killed, _ = CRITERIA["als-ema"][1](summ, recs)
    assert killed is False
    only_slow = [r for r in recs if r["method"] != "fast"]
    killed, _ = CRITERIA["als-ema"][1](ev.summarise_track(only_slow), only_slow)
    assert killed is True


def test_ivf_criterion_uses_matched_recall():
    def ivf(method, p99s):
        r = _run(method, 0, [(1, 1.0)], role="ours" if method == "balanced_ivf" else "baseline")
        r["extra"]["latency"] = [{"nprobe": n, "recall10": rc, "p50_us": 1, "p99_us": p} for n, rc, p in p99s]
        return r
    base = ivf("emeans_ivf", [(8, 0.85, 100), (16, 0.95, 200)])
    ours = ivf("balanced_ivf", [(8, 0.85, 95), (16, 0.95, 190)])
    killed, reason = CRITERIA["balanced-ivf"][1]({}, [base, ours])
    assert killed is True, reason  # ~5% < 10%
