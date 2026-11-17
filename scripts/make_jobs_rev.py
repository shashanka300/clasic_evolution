"""Write results/jobs_rev_v3.txt: the schedule-vs-revival ablation baseline (PAPER_LOG §2, 2026-11-16).

Tuned stepwise / online EM *plus the recipe's revival rule* (``<stepwise>_rev_tuned``), same (tau0, kappa) grids,
budgets, seeds and selection as the v3 tuned baselines. ALS is excluded: neither the recipe nor its baseline uses
revival there. Short budgets first (full grid, all 5 seeds); the long budgets (10, 25) then use a top-3 grid frozen
from this baseline's own T=5 cells over all 5 seeds; a final full-seed run selects on val and writes the record.

  python3 scripts/make_jobs_rev.py && python3 scripts/pqueue.py --jobs results/jobs_rev_v3.txt \
      --done results/pqueue_done_rev.txt --budget 32 --max-jobs 16
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_jobs_v3 import CHUNKS, PY, SEEDS, TRACKS  # noqa: E402


def main():
    L = ["# Schedule-vs-revival ablation: tuned stepwise / online EM + revival (scripts/make_jobs_rev.py, PAPER_LOG §2)",
         "# Generated; do not edit by hand."]
    fins = []
    for slug, (sh, runner, port, dsets, thr, _extra) in TRACKS.items():
        if sh == "als":
            continue
        env = f"CLASSIC_ALGOS_GPU_PORTS={port}"
        L.append(f"# --- {slug}")
        for ds in dsets:
            name, cov = (ds.split(":") + [None])[:2]
            sel = f"--datasets {name}" + (f" --covs {cov}" if cov else "")
            tag = f"{sh}_{name}" + (f"_{cov}" if cov else "")
            log = f"results/{slug}/console_rev_{name}{'_' + cov if cov else ''}"
            short = [f"rv_{tag}_s{s}_short" for s in SEEDS]
            longs = []
            for s in SEEDS:
                for ck, bud in CHUNKS.items():
                    jid = f"rv_{tag}_s{s}_{ck}"
                    dep = "" if ck == "short" else f" after={','.join(short)}"
                    L.append(f"id={jid} threads={thr} needs=data{dep} :: CLASSIC_ALGOS_STEPWISE_CELLS_ONLY=1 "
                             f"CLASSIC_ALGOS_BUDGETS={bud} {env} {PY} {runner} {sel} --stage online_rev --seed-list {s} "
                             f">> {log}_{s}_{ck}.log 2>&1")
                    if ck != "short":
                        longs.append(jid)
            L.append(f"id=fin_rev_{tag} threads=1 needs=data after={','.join(short + longs)} :: {env} {PY} {runner} {sel} "
                     f"--stage online_rev --seeds 5 >> {log}_final.log 2>&1")
            fins.append(f"fin_rev_{tag}")
    L.append("id=rev_evaluate threads=4 barrier :: .venv/bin/python scripts/evaluate.py > results/evaluate_all.log 2>&1 "
             "&& .venv/bin/python scripts/make_figures.py >> results/evaluate_all.log 2>&1")
    out = ROOT / "results" / "jobs_rev_v3.txt"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}: {sum(1 for x in L if x.startswith('id='))} jobs")


if __name__ == "__main__":
    main()
