# Superseded: LDA records selected on the [0.01, 1] alpha0 grid (2026-11-14)

20 of 60 lda_ema sweep cells had their optimum at the grid floor 0.01. `tracks/lda-scaling/extend_grid.py` extended
the grid downward; 6 cells moved to 0.00518, including the NYTimes headline cell at T=5. Archived and regenerated:
- `runs_removed.jsonl`: NYTimes `lda_ema_swept` (emit) and all `cosine_law` records (their leave-one-corpus-out law
  is fitted on law-sweep cells, several of which moved).
- `timing_runs_removed.jsonl`: NYTimes `lda_ema_swept` from the isolated timing pass (results/_timing).
See PAPER_LOG.md §5.
