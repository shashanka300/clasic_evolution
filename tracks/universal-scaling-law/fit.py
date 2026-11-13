"""Universal Scaling Law Cross-Algorithm Synthesis (Track T12)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np


@dataclass
class UniversalFitResult:
    """Result of cross-algorithm universal scaling law fit."""

    shared_beta: float  # Exponent on (N/K)
    shared_gamma: float  # Exponent on Batch size
    shared_delta: float  # Exponent on Epochs T
    algo_constants: Dict[str, float]  # Algorithm-specific constant C_algo
    r2_universal: float  # Overall R^2 across all algorithms
    residuals: Dict[str, float]  # Per-algorithm root mean square error (RMSE)


@dataclass
class UniversalScalingFitter:
    """Fits shared log-linear scaling law: ln alpha0 = ln C_algo + beta*ln(N/K) + gamma*ln(B) + delta*ln(T)."""

    def fit(self, records: List[Dict[str, Any]]) -> UniversalFitResult:
        """Fit joint linear model with dummy variables for algorithm offsets and shared exponents."""
        valid_records = []
        for r in records:
            alpha = float(r.get("alpha0", 0.0))
            n = float(r.get("n", 0.0))
            k = float(r.get("k_or_units", 1.0))
            b = float(r.get("batch", 1.0))
            t = float(r.get("epochs", 1.0))
            track = str(r.get("track", "unknown"))

            if alpha <= 1e-4 or n <= 0.0 or k <= 0.0 or b <= 0.0 or t <= 0.0:
                continue
            valid_records.append({
                "track": track,
                "log_alpha": math.log(alpha),
                "log_rho": math.log(n / k),  # Effective points per unit
                "log_b": math.log(b),
                "log_t": math.log(t),
            })

        if not valid_records:
            raise ValueError("No valid positive records with alpha0 > 0 found for universal fit.")

        algorithms = sorted(list(set(r["track"] for r in valid_records)))
        algo_idx = {algo: i for i, algo in enumerate(algorithms)}
        n_algos = len(algorithms)

        # Design matrix X: [one-hot for algo (n_algos columns), log_rho, log_b, log_t]
        n_samples = len(valid_records)
        X = np.zeros((n_samples, n_algos + 3), dtype=np.float64)
        y = np.zeros(n_samples, dtype=np.float64)

        for i, rec in enumerate(valid_records):
            X[i, algo_idx[rec["track"]]] = 1.0
            X[i, n_algos] = rec["log_rho"]
            X[i, n_algos + 1] = rec["log_b"]
            X[i, n_algos + 2] = rec["log_t"]
            y[i] = rec["log_alpha"]

        # OLS fit via pseudo-inverse: theta = (X^T X)^{-1} X^T y
        theta, residuals_sum, rank, s = np.linalg.lstsq(X, y, rcond=None)

        algo_constants = {algo: float(math.exp(theta[i])) for i, algo in enumerate(algorithms)}
        shared_beta = float(theta[n_algos])
        shared_gamma = float(theta[n_algos + 1])
        shared_delta = float(theta[n_algos + 2])

        y_pred = X @ theta
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        ss_res = float(np.sum((y - y_pred) ** 2))
        r2 = 1.0 - (ss_res / max(ss_tot, 1e-12))

        # Per-algorithm residuals
        per_algo_rmse = {}
        for algo in algorithms:
            mask = [r["track"] == algo for r in valid_records]
            if any(mask):
                algo_res = y[mask] - y_pred[mask]
                per_algo_rmse[algo] = float(np.sqrt(np.mean(algo_res ** 2)))

        return UniversalFitResult(
            shared_beta=shared_beta,
            shared_gamma=shared_gamma,
            shared_delta=shared_delta,
            algo_constants=algo_constants,
            r2_universal=max(r2, 0.0),
            residuals=per_algo_rmse,
        )


def load_all_phase1_records(results_root: Path) -> List[Dict[str, Any]]:
    """Scan and aggregate benchmark records across all completed tracks."""
    records = []
    for jsonl_file in sorted(results_root.glob("*/benchmarks.jsonl")):
        if jsonl_file.parent.name == "universal-scaling-law":
            continue
        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as e:
            print(f"Warning: could not read {jsonl_file}: {e}")
    return records
