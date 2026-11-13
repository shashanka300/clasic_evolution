"""Test isolation.

1. Tests never write into the real ``results/`` tree (that is how legacy LOG.md/JSONL files got
   polluted by test runs): ``CLASSIC_ALGOS_RESULTS_DIR`` points at a session tmp dir.
2. Track modules are imported by bare name (``from plugin import ...``) after a ``sys.path`` insert.
   Several tracks have a ``plugin.py``/``bench.py``, so before collecting each test module we drop
   any cached bare-name track module, letting each test file bind to its own track.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_TRACK_MODULES = {p.stem for p in (ROOT / "tracks").glob("*/*.py") if p.stem != "__init__"}

os.environ.setdefault("CLASSIC_ALGOS_RESULTS_DIR", tempfile.mkdtemp(prefix="classic_algos_test_results_"))


_SNAPSHOTS: dict[str, dict] = {}


def _purge() -> None:
    for name in list(sys.modules):
        if name.split(".")[0] in _TRACK_MODULES:
            del sys.modules[name]


def pytest_collectstart(collector):
    if isinstance(collector, pytest.Module):
        _purge()


def pytest_itemcollected(item):
    key = str(item.path)
    if key not in _SNAPSHOTS:
        mods = {n: m for n, m in sys.modules.items() if n.split(".")[0] in _TRACK_MODULES}
        dirs = sorted({str(Path(m.__file__).parent) for m in mods.values() if getattr(m, "__file__", None)})
        _SNAPSHOTS[key] = {"mods": mods, "dirs": dirs}


def pytest_runtest_setup(item):
    snap = _SNAPSHOTS.get(str(item.path))
    if snap is None:
        return
    _purge()
    sys.modules.update(snap["mods"])
    for d in snap["dirs"]:
        if d in sys.path:
            sys.path.remove(d)
        sys.path.insert(0, d)
