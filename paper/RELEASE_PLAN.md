# Code and data release plan

*Drafted 2026-11-17. Decisions marked **(author)** are yours; everything else is a recommendation.*

## What we release

| item | release? | how |
|---|---|---|
| Code (`core/`, `tracks/`, `scripts/`, `tests/`), `pyproject.toml`, `uv.lock` | yes | public repo, MIT or Apache-2.0 **(author: pick)** |
| Job files (`results/jobs_*.txt`) and pre-registration (`RESEARCH_PLAN.md`, `scripts/kill_criteria.py`) | yes | same repo |
| Run records: 67 JSONL files, 44,197 rows (one per run) | yes | same repo; ~50 MB, below GitHub's limits, so no LFS needed |
| Verdict tables, law fits, figure sources (`results/**/VERDICT.md`, `*.json`, `results/figures/`) | yes | same repo |
| Console logs (`results/**/*.log`, 1,088 files) | yes, after scrubbing | they contain the local home path (`/home/<user>`, ~25k occurrences in 199 files); replace with `$HOME` |
| Raw datasets | **no** | `scripts/fetch_data.py` downloads every dataset from its provider; MovieLens forbids redistribution, AG News and CIFAR have no stated licence |
| Derived caches (embeddings, PCA projections, VQ codes, TF-IDF) | no | rebuilt by `fetch_data.py` steps 2–3 (DINOv2 / MiniLM / BGE embeddings need a GPU) |
| `PAPER_LOG.md`, `paper/` | yes at camera-ready / arXiv | keep in the repo; drop from the anonymous review copy (see below) |
| `MIGRATION_5090.md`, `MAC_AND_DATA_PLAN.md`, `research_notes/`, `reports/` | no | internal machine notes and the literature review; they name the GitHub account |

The JSONL records contain no local paths or personal data (checked 2026-11-17).

## Two release moments

1. **Submission (double-blind).** An anonymised snapshot, e.g. via anonymous.4open.science, built from a clean export:
   - export from the release commit with `git archive`, dropping the excluded files above;
   - scrub home paths in the logs and the GitHub user name from all files;
   - the paper cites it as "code and all run records: anonymised link".
   - If the venue is not double-blind (TMLR is, arXiv is not), skip this step.
2. **arXiv / camera-ready.** Make the repo public, or push the clean export to a new public repo **(author: same repo with
   its history, or a fresh one?)**. The history contains internal notes and pre-scrub logs, so a fresh repo with a
   single squashed release commit is simpler. Tag the release `v1.0-paper`, archive it on Zenodo for a DOI, and cite
   the DOI.

## Steps (when you say go)

1. Choose the licence; add `LICENSE` and a `CITATION.cff`.
2. Write a `scripts/make_release.sh` that exports the tree, drops the excluded paths and scrubs paths.
3. Rewrite `README.md` as a user-facing guide: install (`uv sync`), fetch data, reproduce one table, regenerate
   all verdicts and figures from the JSONL without re-running anything.
4. Check that `scripts/evaluate.py` and `scripts/make_figures.py` rebuild every number in the draft from the release
   tree alone (this is the paper's real reproducibility claim; the full re-run is ~575 job-hours).
5. Add `paper/datasets.md`'s licence and citation table to the README.

Nothing is published or pushed until you approve each step.
