"""Provenance-carrying run records (v2 schema) for every track.

Rules this module enforces (see ``scripts/evaluate.py`` for the consumer):

* Track scripts write JSONL **only** through :func:`write_run`. They never write verdicts,
  speedups, PASS/FAIL strings or fitted laws into prose. ``scripts/evaluate.py`` computes all of
  that from the JSONL.
* Every record says what it is: ``extra.role`` (baseline / ours / ablation), where its alpha0 came
  from (``extra.alpha0_source``), which comparison it belongs to (``extra.group``), which git commit
  produced it and a hash of its full config.
* ``test_loss`` is always *lower is better*. The metric name goes in ``extra.metric``.
* Iterative methods log a ``extra.curve``: a list of ``{"epoch", "test_loss", "wall_s"}`` points so
  "epochs to reach baseline@25" is measured by interpolation, never asserted.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

from core.bench import REQUIRED_RESULT_KEYS, read_results

SCHEMA_VERSION = 2
ROLES = ("baseline", "ours", "ablation")
ALPHA0_SOURCES = ("fixed", "swept", "law", "na")
CURVE_EPOCHS = (1, 2, 3, 5, 10, 25)
REQUIRED_EXTRA_KEYS = ("schema", "role", "alpha0_source", "group", "metric", "git_commit", "config_hash")


@functools.lru_cache(maxsize=1)
def git_commit() -> str:
    """Return ``<sha>`` or ``<sha>-dirty`` for the repo containing this file."""
    root = Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=root, text=True).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", "core", "tracks"], cwd=root
        ).returncode != 0
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def config_hash(config: dict[str, Any]) -> str:
    """Stable 12-hex-digit hash of a JSON-serialisable config dict."""
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _clean(v: Any) -> Any:
    """Convert numpy scalars / arrays and non-finite floats to JSON-safe values."""
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def make_run(
    *,
    track: str,
    method: str,
    dataset: str,
    n: int,
    d: int,
    k_or_units: int,
    epochs: float,
    batch: int,
    alpha0: float | None,
    seed: int,
    train_loss: float | None,
    test_loss: float | None,
    wall_s: float,
    role: str,
    alpha0_source: str,
    group: str,
    metric: str,
    config: dict[str, Any] | None = None,
    curve: Iterable[dict[str, float]] | None = None,
    gpu_s: float = 0.0,
    reference: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    """Build one validated v2 record.

    ``reference=True`` marks the classic baseline whose 25-epoch loss is the bar the group's
    ``ours`` methods must reach (exactly one reference method per group).
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    if alpha0_source not in ALPHA0_SOURCES:
        raise ValueError(f"alpha0_source must be one of {ALPHA0_SOURCES}, got {alpha0_source!r}")
    if reference and role != "baseline":
        raise ValueError("only a baseline can be the reference")
    cfg = dict(config or {})
    cfg.update(method=method, dataset=dataset, n=n, k=k_or_units, epochs=epochs, batch=batch, alpha0=alpha0)
    ext: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "role": role,
        "alpha0_source": alpha0_source,
        "group": group,
        "metric": metric,
        "reference": bool(reference),
        "git_commit": git_commit(),
        "config_hash": config_hash(cfg),
        "config": cfg,
    }
    if curve is not None:
        pts = sorted((dict(p) for p in curve), key=lambda p: p["epoch"])
        for p in pts:
            if "epoch" not in p or "test_loss" not in p:
                raise ValueError("curve points need 'epoch' and 'test_loss'")
        ext["curve"] = pts
    ext.update(extra)
    rec = {
        "track": track,
        "method": method,
        "dataset": dataset,
        "n": int(n),
        "d": int(d),
        "k_or_units": int(k_or_units),
        "epochs": float(epochs),
        "batch": int(batch),
        "alpha0": None if alpha0 is None else float(alpha0),
        "seed": int(seed),
        "train_loss": train_loss,
        "test_loss": test_loss,
        "wall_s": float(wall_s),
        "gpu_s": float(gpu_s),
        "extra": ext,
    }
    return _clean(rec)


def validate_run(rec: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_RESULT_KEYS if k not in rec]
    if missing:
        raise ValueError(f"record missing keys {missing}")
    ext = rec["extra"]
    missing_x = [k for k in REQUIRED_EXTRA_KEYS if k not in ext]
    if missing_x:
        raise ValueError(f"record.extra missing keys {missing_x}")
    if ext["role"] not in ROLES or ext["alpha0_source"] not in ALPHA0_SOURCES:
        raise ValueError(f"bad role/alpha0_source in {ext}")


def results_root() -> Path:
    """``results/`` at the repo root, or ``$CLASSIC_ALGOS_RESULTS_DIR`` (tests point this at a tmp dir)."""
    env = os.environ.get("CLASSIC_ALGOS_RESULTS_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[1] / "results"


def results_path(slug: str, name: str = "runs.jsonl") -> Path:
    return results_root() / slug / name


def write_run(path: str | Path, rec: dict[str, Any]) -> None:
    """Validate and append one v2 record."""
    validate_run(rec)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def read_runs(path: str | Path) -> list[dict[str, Any]]:
    """Read v2 records, silently skipping legacy (schema<2) rows."""
    return [r for r in read_results(path) if r.get("extra", {}).get("schema", 1) >= SCHEMA_VERSION]


def done_hashes(path: str | Path) -> set[tuple[str, int]]:
    """``(config_hash, seed)`` pairs already present, so sweeps can resume."""
    return {(r["extra"]["config_hash"], r["seed"]) for r in read_runs(path)}
