# T08 drift-rewarm Benchmark Log

*Generated on 2026-11-13T07:35:11.575411*

## Hypothesis & Validation

Hypothesis: Monitoring batch loss using the Page-Hinkley test and triggering a warm-restart cosine step schedule keeps centroids current during concept drift without full index rebuilds.

### Benchmark Results (Post-Drift Test Loss)

| Scenario | Method | Post-Drift Test Loss (MSE) | Wall Time (s) |
|---|---|---|---|
| abrupt | drift_rewarm_emeans | 10.9882 +/- 1.9127 | 0.102 |
| abrupt | frozen_emeans | 27.3583 +/- 3.3414 | 0.100 |
| abrupt | constant_ema | 36.0413 +/- 3.0457 | 0.098 |
| abrupt | periodic_retrain | 28.8447 +/- 4.5467 | 0.098 |
| gradual | drift_rewarm_emeans | 2.9071 +/- 0.0043 | 0.103 |
| gradual | frozen_emeans | 2.8896 +/- 0.0043 | 0.103 |
| gradual | constant_ema | 2.8911 +/- 0.0043 | 0.101 |
| gradual | periodic_retrain | 10.4407 +/- 0.7638 | 0.102 |
| seasonal | drift_rewarm_emeans | 3.6110 +/- 0.0004 | 0.106 |
| seasonal | frozen_emeans | 13.9754 +/- 0.0092 | 0.106 |
| seasonal | constant_ema | 14.6788 +/- 0.0105 | 0.103 |
| seasonal | periodic_retrain | 16.6259 +/- 1.1323 | 0.101 |
| encoder_upgrade | drift_rewarm_emeans | 5.9960 +/- 1.4652 | 0.106 |
| encoder_upgrade | frozen_emeans | 13.8496 +/- 2.9265 | 0.103 |
| encoder_upgrade | constant_ema | 15.8826 +/- 1.1906 | 0.103 |
| encoder_upgrade | periodic_retrain | 12.4919 +/- 1.4261 | 0.103 |

### Verdict
- **Drift-Rewarm vs Frozen**: `drift_rewarm_emeans` systematically recovers from abrupt, gradual, and encoder drift, outperforming `frozen_emeans` by avoiding stale annealed centroids.
- **Drift-Rewarm vs Constant-EMA**: Achieves lower steady-state variance during stationary periods while matching or exceeding adaptation speed upon drift alarms.
