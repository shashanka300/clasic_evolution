"""Regenerate every swept headline record (<ema>_swept) from the final, complete sweep cells.

fin_emit_* jobs ran per group as soon as that group's sweep jobs ended, and emit_swept_curves keeps only seeds present
in every cell and never rewrites an existing (method, group, seed) record. Some cells were still incomplete then (LDA
grid-extension values on seeds 0-2 only; three GMM groups emitted before seed 4 was readable), so those records used
fewer seeds for alpha selection. This drops all <ema>_swept rows (copy kept in results/<track>/reemit_removed.jsonl)
and re-runs each track's emit stage over all its datasets. Idempotent. See PAPER_LOG.md §2.

  python3 scripts/reemit_v3.py [track-slug ...]   # default: all five tracks
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACKS = {"gmm-ema": "tracks/gmm-ema/run_v2.py", "dictlearn-scaling": "tracks/dictlearn-scaling/run_v2.py",
          "lda-scaling": "tracks/lda-scaling/run_v2.py", "als-ema": "tracks/als-ema/run_v2.py",
          "hmm-ema": "tracks/hmm-ema/run_v2.py"}


def main():
    only = set(sys.argv[1:])
    for slug, runner in TRACKS.items():
        if only and slug not in only:
            continue
        runs = ROOT / "results" / slug / "runs.jsonl"
        lines = runs.read_text().splitlines()
        keep, drop = [], []
        for line in lines:
            if line.strip():
                (drop if json.loads(line)["method"].endswith("_swept") else keep).append(line)
        if drop:
            with open(runs.with_name("reemit_removed.jsonl"), "a") as f:
                f.write("\n".join(drop) + "\n")
            tmp = runs.with_suffix(".tmp")
            tmp.write_text("\n".join(keep) + "\n")
            tmp.replace(runs)
        print(f"[reemit] {slug}: removed {len(drop)} swept rows; re-emitting", flush=True)
        subprocess.run([str(ROOT / ".venv/bin/python"), "-u", runner, "--stage", "emit", "--seeds", "5"], cwd=ROOT, check=True)
        n = sum(1 for line in runs.read_text().splitlines() if line.strip() and json.loads(line)["method"].endswith("_swept"))
        print(f"[reemit] {slug}: {n} swept rows now", flush=True)


if __name__ == "__main__":
    sys.exit(main())
