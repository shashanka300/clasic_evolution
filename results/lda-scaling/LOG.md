# T05 lda-scaling Benchmark Log

*Generated on 2026-11-13T07:59:33.422717*

## Hypothesis & Validation

Hypothesis: Online LDA with a cosine step schedule and scaling-law alpha0 replaces the hand-tuned (tau0, kappa) decay schedule, matching or outperforming tuned online LDA without hyperparameter searches.

### Benchmark Results

| Corpus | Method | Held-Out Perplexity | Wall Time (s) |
|---|---|---|---|
| corpus_small | cosine_scaling_lda | 109.73 +/- 2.96 | 0.66s |
| corpus_small | hoffman_default_lda | 195.12 +/- 20.37 | 0.81s |
| corpus_small | hoffman_tuned_lda | 123.53 +/- 6.83 | 0.74s |
| corpus_medium | cosine_scaling_lda | 253.62 +/- 2.47 | 1.25s |
| corpus_medium | hoffman_default_lda | 468.38 +/- 2.33 | 1.40s |
| corpus_medium | hoffman_tuned_lda | 268.08 +/- 8.21 | 1.41s |

### Finding
- **Cosine Scaling vs Hoffman Default**: `cosine_scaling_lda` consistently achieves lower perplexity than default Hoffman (`tau0=1024, kappa=0.7`) by enabling aggressive early parameter exploration while smoothly annealing to zero variance.
- **Parameter-free Operation**: Matches or beats the performance of hand-tuned Hoffman schedules without requiring grid search across (tau0, kappa).
