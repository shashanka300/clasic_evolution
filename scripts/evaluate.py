"""Compute every verdict from v2 JSONL. The only place speedups / PASS / KILL are decided.

Usage::

    python scripts/evaluate.py                 # all tracks
    python scripts/evaluate.py gmm-ema als-ema # selected tracks

Reads ``results/<slug>/runs.jsonl`` (v2 records only; legacy rows are ignored), writes
``results/<slug>/VERDICT.md`` (auto-generated, never hand-edited) and ``results/summary.json``.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from core.harness import CURVE_EPOCHS, read_runs  # noqa: E402
from kill_criteria import ALPHA, CRITERIA, FROZEN_AT, _paired  # noqa: E402

N_BOOT = 2000


def _curve(rec: dict[str, Any], key: str = "test_loss") -> dict[float, float]:
    pts = rec["extra"].get("curve")
    if pts:
        return {float(p["epoch"]): float(p[key]) for p in pts if p.get(key) is not None}
    if rec.get("test_loss") is not None and key == "test_loss":
        return {float(rec["epochs"]): float(rec["test_loss"])}
    return {}


def _at(curve: dict[float, float], epoch: float) -> float | None:
    """Value at ``epoch`` (exact grid point, else linear interpolation inside the curve)."""
    if epoch in curve:
        return curve[epoch]
    xs = sorted(curve)
    for a, b in zip(xs, xs[1:]):
        if a < epoch < b:
            f = (epoch - a) / (b - a)
            return curve[a] + f * (curve[b] - curve[a])
    return None


def epochs_to_match(curve: dict[float, float], target: float) -> float | None:
    """First epoch where the curve reaches ``target`` (lower is better), linearly interpolated.

    If the first logged point already beats the target we return that point's epoch: the curve
    gives no evidence about earlier epochs, so we do not extrapolate.
    """
    xs = sorted(curve)
    if not xs:
        return None
    if curve[xs[0]] <= target:
        return xs[0]
    for a, b in zip(xs, xs[1:]):
        if curve[b] <= target:
            f = (curve[a] - target) / (curve[a] - curve[b])
            return a + f * (b - a)
    return None


def _mean_curve(curves: list[dict[float, float]]) -> tuple[dict[float, float], dict[float, float]]:
    common = set.intersection(*(set(c) for c in curves)) if curves else set()
    mean = {e: float(np.mean([c[e] for c in curves])) for e in sorted(common)}
    std = {e: float(np.std([c[e] for c in curves])) for e in sorted(common)}
    return mean, std


def _boot_etm(curves: list[dict[float, float]], target_by_seed: list[float], rng) -> tuple[float, float] | None:
    """Bootstrap 95% CI of epochs-to-match, resampling seeds of method and reference."""
    if len(curves) < 2:
        return None
    vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(curves), len(curves))
        mc, _ = _mean_curve([curves[i] for i in idx])
        tgt = float(np.mean(rng.choice(target_by_seed, len(target_by_seed))))
        e = epochs_to_match(mc, tgt)
        vals.append(np.inf if e is None else e)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def summarise_track(records: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r["extra"]["group"]].append(r)

    out: dict[str, Any] = {"groups": {}}
    for gname, recs in sorted(groups.items()):
        ref_recs = [r for r in recs if r["extra"].get("reference")]
        ref = None
        if ref_recs:
            ref_by_seed, ref_wall = {}, []
            for r in ref_recs:
                c = _curve(r)
                v = c.get(25.0)
                if v is not None:
                    ref_by_seed[r["seed"]] = v
                    w = _curve(r, "wall_s").get(25.0, r["wall_s"] if r["epochs"] == 25 else None)
                    if w is not None:
                        ref_wall.append(w)
            if ref_by_seed:
                vals = list(ref_by_seed.values())
                ref = {
                    "method": ref_recs[0]["method"],
                    "loss25_mean": float(np.mean(vals)),
                    "loss25_std": float(np.std(vals)),
                    "by_seed": ref_by_seed,
                    "wall25_mean": float(np.mean(ref_wall)) if ref_wall else None,
                }
        by_method: dict[str, list[dict]] = defaultdict(list)
        for r in recs:
            by_method[r["method"]].append(r)

        methods = {}
        for mname, mrecs in sorted(by_method.items()):
            curves = [_curve(r) for r in mrecs]
            curves = [c for c in curves if c]
            if not curves:
                continue
            mc, sc = _mean_curve(curves)
            finals = {r["seed"]: float(r["test_loss"]) for r in mrecs if r.get("test_loss") is not None}
            m: dict[str, Any] = {
                "role": mrecs[0]["extra"]["role"],
                "alpha0_source": mrecs[0]["extra"]["alpha0_source"],
                "alpha0": mrecs[0]["alpha0"],
                "metric": mrecs[0]["extra"]["metric"],
                "n_seeds": len(mrecs),
                "curve_mean": mc,
                "curve_std": sc,
                "final_mean": float(np.mean(list(finals.values()))) if finals else None,
                "final_std": float(np.std(list(finals.values()))) if finals else None,
                "epochs_to_match": None,
                "etm_ci95": None,
                "gain": None,
                "wall_to_match": None,
                "wall_speedup": None,
                "final_vs_ref_diff": None,
                "final_vs_ref_p": None,
                "match_p": None,
            }
            if ref is not None and mname != ref["method"]:
                etm = epochs_to_match(mc, ref["loss25_mean"])
                m["epochs_to_match"] = etm
                m["gain"] = (25.0 / etm) if etm else 0.0
                m["etm_ci95"] = _boot_etm(curves, list(ref["by_seed"].values()), rng)
                if etm is not None:
                    wcurves = [_curve(r, "wall_s") for r in mrecs]
                    wcurves = [w for w in wcurves if w]
                    if wcurves:
                        wm, _ = _mean_curve(wcurves)
                        w = _at(wm, etm)
                        m["wall_to_match"] = w
                        if w and ref["wall25_mean"]:
                            m["wall_speedup"] = ref["wall25_mean"] / w
                    # paired test at the first grid epoch at/after etm
                    grid_e = min(e for e in mc if e >= etm)
                    at_e = {r["seed"]: _curve(r)[grid_e] for r in mrecs if grid_e in _curve(r)}
                    d, p, _ = _paired(at_e, ref["by_seed"])
                    m["match_p"] = {"epoch": grid_e, "diff": d, "p": p}
                d, p, _ = _paired(finals, ref["by_seed"])
                m["final_vs_ref_diff"], m["final_vs_ref_p"] = d, p
            methods[mname] = m
        out["groups"][gname] = {"dataset": recs[0]["dataset"], "n": recs[0]["n"], "k": recs[0]["k_or_units"],
                                "reference": ref, "methods": methods}
    return out


def _fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != v:
            return "nan"
        return f"{v:.{nd}g}"
    return str(v)


def render_verdict(slug: str, records, summary, verdict) -> str:
    commits = sorted({r["extra"]["git_commit"] for r in records})
    lines = [
        f"# {slug} — VERDICT (auto-generated)",
        "",
        f"Generated by `scripts/evaluate.py` at {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. "
        "Do not edit by hand; re-run the evaluator.",
        "",
        f"- Records: {len(records)} (v2 schema). Commits: {', '.join(commits)}",
        f"- Kill criterion (frozen {FROZEN_AT}): {verdict['criterion']}",
        f"- **Outcome: {verdict['status']}**: {verdict['reason']}",
        "",
        "Epochs-to-match = first epoch where the seed-mean test curve reaches the seed-mean reference@25,",
        "linearly interpolated; gain = 25 / that. 95% CI from a seed bootstrap (inf = not reached in the",
        f"resample). p-values: paired t-test by seed, alpha = {ALPHA}.",
        "",
    ]
    for gname, g in summary["groups"].items():
        lines += [f"## {gname}", ""]
        ref = g["reference"]
        if ref:
            lines.append(f"Reference `{ref['method']}`@25: {_fmt(ref['loss25_mean'])} ± {_fmt(ref['loss25_std'])} "
                         f"(wall {_fmt(ref['wall25_mean'], 3)} s)")
            lines.append("")
        eps = sorted({e for m in g["methods"].values() for e in m["curve_mean"]})
        show = [e for e in eps if e in CURVE_EPOCHS] or eps[:8]
        hdr = "| method | role | α0 (src) | seeds | " + " | ".join(f"@{_fmt(e, 3)}" for e in show) + \
              " | final | epochs→ref | CI95 | gain | wall× | p(final vs ref) |"
        lines += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
        for mname, m in g["methods"].items():
            cells = [_fmt(m["curve_mean"].get(e)) for e in show]
            ci = m["etm_ci95"]
            lines.append(
                f"| {mname} | {m['role']} | {_fmt(m['alpha0'], 3)} ({m['alpha0_source']}) | {m['n_seeds']} | "
                + " | ".join(cells)
                + f" | {_fmt(m['final_mean'])} ± {_fmt(m['final_std'], 2)} | {_fmt(m['epochs_to_match'], 3)} | "
                f"{'—' if ci is None else f'[{ci[0]:.2f}, {ci[1]:.2f}]'} | {_fmt(m['gain'], 3)} | "
                f"{_fmt(m['wall_speedup'], 3)} | {_fmt(m['final_vs_ref_p'], 2)} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def evaluate(slug: str) -> dict[str, Any] | None:
    path = ROOT / "results" / slug / "runs.jsonl"
    records = read_runs(path)
    if not records:
        return None
    summary = summarise_track(records)
    crit_name, fn = CRITERIA.get(slug, ("(no kill criterion in plan)", None))
    if fn is None:
        killed, reason = None, "no kill criterion defined; descriptive results only"
    else:
        killed, reason = fn(summary, records)
    status = "INCOMPLETE" if killed is None else ("KILLED" if killed else "SURVIVES")
    verdict = {"criterion": crit_name, "status": status, "reason": reason}
    (ROOT / "results" / slug / "VERDICT.md").write_text(render_verdict(slug, records, summary, verdict))
    return {"verdict": verdict, "summary": summary, "n_records": len(records)}


def main(argv: list[str]) -> None:
    slugs = argv or sorted(p.name for p in (ROOT / "results").iterdir() if (p / "runs.jsonl").exists())
    allres = {}
    for s in slugs:
        res = evaluate(s)
        if res is None:
            print(f"{s:24s} no v2 records")
            continue
        allres[s] = res
        v = res["verdict"]
        print(f"{s:24s} {v['status']:10s} {v['reason']}")
    summ_path = ROOT / "results" / "summary.json"
    prev = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    prev.update(allres)
    summ_path.write_text(json.dumps(prev, indent=1, default=lambda o: o if not isinstance(o, float) else o))


if __name__ == "__main__":
    main(sys.argv[1:])
