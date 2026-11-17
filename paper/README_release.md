# Scaling laws for stochastic-approximation EM

Code, job files and every run record for the paper *Scaling laws for stochastic-approximation EM: one recipe and
one step-size rule across five model families* (draft in `paper/draft.md`).

The e-means recipe (an EMA of sufficient statistics, a cosine-annealed step, dead-unit revival and a step-size law)
is applied to Gaussian mixtures, HMMs, LDA, online dictionary learning / NMF and ALS, and compared with full-batch
EM and tuned stepwise EM.

## Layout

| path | contents |
|---|---|
| `core/` | shared engine: EMA engine, cosine schedule, revival, budgets, benchmark harness, k-means (e-means) |
| `tracks/gmm-ema`, `hmm-ema`, `lda-scaling`, `dictlearn-scaling`, `als-ema` | one model family each |
| `tracks/universal-scaling-law` | law fits (`fit_v2.py`, `refit_coords.py`, `missing_info.py`) |
| `scripts/` | data download, job generation, queue, evaluation, figures, timing |
| `results/` | one JSONL line per run (44k runs), verdict tables, law fits, figures, logs |
| `RESEARCH_PLAN.md`, `scripts/kill_criteria.py` | pre-registered hypotheses and kill criteria |
| `PAPER_LOG.md`, `paper/` | lab notebook (every result, decision and deviation) and the paper draft |

## Reproduce the numbers without re-running anything

```bash
uv sync
uv run python scripts/evaluate.py          # verdict tables from results/**/*.jsonl
uv run python scripts/make_figures.py      # all figures into results/figures/
uv run python tracks/universal-scaling-law/refit_coords.py   # law fits (Qe0 etc.)
```

## Re-run the experiments

```bash
uv run python scripts/fetch_data.py        # downloads every dataset from its provider; GPU needed for embeddings
uv run python scripts/make_jobs_v3.py
uv run python scripts/pqueue.py --jobs results/jobs_budget_v3.txt --done results/pqueue_done_v3.txt --budget 32 --max-jobs 16
uv run pytest tests
```

The full protocol is about 575 job-hours on one desktop (RTX 5090 + CPU). Datasets are not redistributed; see
`paper/datasets.md` for sources, licences and required citations (MovieLens, for example, forbids redistribution).

## Licence

MIT (`LICENSE`). The datasets keep their own licences.
