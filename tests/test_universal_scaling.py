"""Unit tests for Track T12: universal-scaling-law."""

from __future__ import annotations

import sys
import math
from pathlib import Path
import numpy as np
import pytest

# Ensure track dir is in sys.path
TRACK_DIR = Path(__file__).resolve().parent.parent / "tracks" / "universal-scaling-law"
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))

from fit import UniversalScalingFitter, UniversalFitResult, load_all_phase1_records


def test_universal_fitter_synthetic():
    """Verify UniversalScalingFitter recovers known shared exponents and constants."""
    rng = np.random.default_rng(42)
    synthetic_records = []
    constants = {"algo_a": 1.2, "algo_b": 0.8, "algo_c": 2.5}
    true_beta = -0.45
    true_gamma = 0.85
    true_delta = -0.12

    for algo, c in constants.items():
        for n in [1000, 5000, 20000]:
            for k in [10, 50, 100]:
                for b in [64, 256, 1024]:
                    for t in [1.0, 3.0, 10.0]:
                        rho = n / k
                        log_alpha = math.log(c) + true_beta * math.log(rho) + true_gamma * math.log(b) + true_delta * math.log(t)
                        alpha = math.exp(log_alpha)
                        synthetic_records.append({
                            "track": algo,
                            "n": float(n),
                            "k_or_units": float(k),
                            "batch": float(b),
                            "epochs": float(t),
                            "alpha0": float(alpha),
                        })

    fitter = UniversalScalingFitter()
    result = fitter.fit(synthetic_records)

    assert isinstance(result, UniversalFitResult)
    assert result.r2_universal > 0.99
    assert pytest.approx(result.shared_beta, abs=1e-2) == true_beta
    assert pytest.approx(result.shared_gamma, abs=1e-2) == true_gamma
    assert pytest.approx(result.shared_delta, abs=1e-2) == true_delta
    for algo, c in constants.items():
        assert pytest.approx(result.algo_constants[algo], rel=1e-2) == c


def test_load_phase1_records_and_fit():
    """Verify loading real Phase 1 records from results directory and running universal fit."""
    results_root = Path(__file__).resolve().parent.parent / "results"
    records = load_all_phase1_records(results_root)

    assert len(records) >= 500  # Expect over 1,000 records from all 11 Phase 1 tracks
    tracks_found = set(r.get("track") for r in records if r.get("track"))
    assert "gmm-ema" in tracks_found
    assert "als-ema" in tracks_found
    assert "hmm-ema" in tracks_found
    assert "dictlearn-scaling" in tracks_found

    fitter = UniversalScalingFitter()
    result = fitter.fit(records)

    assert result.r2_universal > 0.60
    assert len(result.algo_constants) >= 5
    for algo, c in result.algo_constants.items():
        assert c > 0.0
    for algo, rmse in result.residuals.items():
        assert math.isfinite(rmse)


def test_invalid_and_zero_alpha_filtered():
    """Verify that records with zero/negative alpha or missing fields are cleanly skipped."""
    records = [
        {"track": "dummy", "alpha0": 0.0, "n": 100, "k_or_units": 10, "batch": 10, "epochs": 1},
        {"track": "dummy", "alpha0": -0.5, "n": 100, "k_or_units": 10, "batch": 10, "epochs": 1},
        {"track": "dummy", "alpha0": 0.5, "n": -100, "k_or_units": 10, "batch": 10, "epochs": 1},
    ]
    fitter = UniversalScalingFitter()
    with pytest.raises(ValueError, match="No valid positive records"):
        fitter.fit(records)
