"""Move records out of a runs.jsonl so a guarded stage regenerates them; the moved rows are appended to
<same dir>/superseded_<tag>.jsonl. Used for the 2026-11-16 HMM rerun after CUDA crashes (PAPER_LOG.md §2).

  python3 scripts/drop_rows.py results/hmm-ema/runs.jsonl --method stepwise_em_tuned --dataset brown_pos --tag cuda_crash
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--method", required=True, help="exact method name, or a suffix if it starts with '*'")
    ap.add_argument("--dataset", nargs="*", default=None, help="dataset names (group prefix before '|'); default all")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    runs = Path(a.runs)
    keep, drop = [], []
    for line in runs.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        m = r["method"]
        hit = m.endswith(a.method[1:]) if a.method.startswith("*") else m == a.method
        if hit and a.dataset is not None:
            hit = r["extra"]["group"].split("|")[0] in a.dataset
        (drop if hit else keep).append(line)
    if drop:
        with open(runs.with_name(f"superseded_{a.tag}.jsonl"), "a") as f:
            f.write("\n".join(drop) + "\n")
        tmp = runs.with_suffix(".tmp")
        tmp.write_text("\n".join(keep) + "\n")
        tmp.replace(runs)
    print(f"[drop_rows] {runs}: moved {len(drop)} rows ({a.method}, {a.dataset})", flush=True)


if __name__ == "__main__":
    main()
