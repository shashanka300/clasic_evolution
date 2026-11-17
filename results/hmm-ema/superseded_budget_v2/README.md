# Superseded: budget-v2 derived records (2026-11-14)

Built on epoch budgets (1, 2, 3, 5) and 3 seeds. Protocol v3 (core/budgets.py) rebuilds them on budgets
(0.25, 0.5, 1, 2, 3, 5, 10, 25) and 5 seeds; see PAPER_LOG.md §2. Sweep rows and full-batch references were kept.
- runs_removed.jsonl: hmm_ema_swept, stepwise_em_tuned, hmm_ema_kmlaw from results/hmm-ema/runs.jsonl
- timing_runs_removed.jsonl: the law and swept records of the isolated timing pass (results/_timing/hmm-ema)
