"""Archive the budget-v2 derived records before the protocol-v3 regeneration (core/budgets.py).

Derived records (val-selected swept curves, tuned stepwise / online-EM baselines, untuned law curves, LDA Hoffman
default and cosine law, sklearn mini-batch DL) were built on budgets (1, 2, 3, 5) and 3 seeds. v3 rebuilds them on
budgets (0.25 .. 25) and 5 seeds. Sweep rows and full-batch references are NOT touched: they stay valid and are
extended in place. Also archives the timing pass's law and swept records (re-timed on the v3 budgets).

Idempotent: a track that already has superseded_budget_v2/ is skipped, so a queue restart never archives the
regenerated v3 records.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DERIVED = {
    "gmm-ema": ("gmm_ema_swept", "online_em_tuned", "gmm_ema_kmlaw"),
    "dictlearn-scaling": ("dl_ema_swept", "stepwise_dl_tuned", "dl_ema_kmlaw", "sklearn_minibatch"),
    "lda-scaling": ("lda_ema_swept", "online_lda_tuned", "lda_ema_kmlaw", "hoffman_default", "cosine_law"),
    "als-ema": ("als_ema_swept", "stepwise_als_tuned", "als_ema_kmlaw"),
    "hmm-ema": ("hmm_ema_swept", "stepwise_em_tuned", "hmm_ema_kmlaw"),
}
README = """# Superseded: budget-v2 derived records (2026-11-14)

Built on epoch budgets (1, 2, 3, 5) and 3 seeds. Protocol v3 (core/budgets.py) rebuilds them on budgets
(0.25, 0.5, 1, 2, 3, 5, 10, 25) and 5 seeds; see PAPER_LOG.md §2. Sweep rows and full-batch references were kept.
- runs_removed.jsonl: {methods} from results/{track}/runs.jsonl
- timing_runs_removed.jsonl: the law and swept records of the isolated timing pass (results/_timing/{track})
"""


def split(path: Path, methods: tuple[str, ...], out: Path) -> tuple[int, int]:
    if not path.exists():
        out.write_text("")
        return 0, 0
    keep, gone = [], []
    for line in path.read_text().splitlines():
        if line.strip():
            (gone if json.loads(line)["method"] in methods else keep).append(line)
    out.write_text("\n".join(gone) + ("\n" if gone else ""))
    path.write_text("\n".join(keep) + ("\n" if keep else ""))
    return len(gone), len(keep)


def main():
    for track, methods in DERIVED.items():
        arc = ROOT / "results" / track / "superseded_budget_v2"
        if arc.exists():
            print(f"{track}: already archived, skipped")
            continue
        arc.mkdir()
        g, k = split(ROOT / "results" / track / "runs.jsonl", methods, arc / "runs_removed.jsonl")
        timing_methods = tuple(m for m in methods if m.endswith(("_swept", "_kmlaw")))
        tg, tk = split(ROOT / "results" / "_timing" / track / "runs.jsonl", timing_methods, arc / "timing_runs_removed.jsonl")
        (arc / "README.md").write_text(README.format(methods=", ".join(methods), track=track))
        print(f"{track}: archived {g} (kept {k}); timing archived {tg} (kept {tk})")


if __name__ == "__main__":
    main()
