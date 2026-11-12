"""Log-linear scaling laws, knee detection, and sweep driver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence
import math
import numpy as np


def emeans_alpha0(n: int, k: int, epochs: float, batch_size: int) -> float:
    """Paper scaling law (Bessa et al., Table 2 Eq. 6) for peak EMA step alpha0."""
    val = 0.824 * (k ** -0.46) * (batch_size ** 0.90) * (n ** -0.45) * (max(epochs, 1e-6) ** -0.11)
    return float(min(max(val, 1e-6), 1.0))


def emeans_batch_size(n: int, k: int, epochs: float) -> int:
    """Paper scaling law (Bessa et al., Table 2 Eq. 7) for knee batch size."""
    raw = 0.035 * (k ** 0.13) * (n ** 0.88) * (max(epochs, 1e-6) ** 0.81)
    if n > 2_000_000:
        raw /= 8.0
    raw = max(raw, 1.0)
    pow2 = 2 ** int(round(math.log2(raw)))
    return int(min(max(pow2, 1), n))


@dataclass(frozen=True)
class LogLinearLaw:
    """Fitted law: ``log(target) = log_c + sum_i coeff_i * log(feat_i)``."""

    target: str
    features: tuple[str, ...]
    constant: float
    coeffs: dict[str, float]
    r2: float

    def predict(self, **kwargs: float) -> float:
        """Evaluate ``constant * prod_i (feat_i ** coeff_i)``."""
        log_val = math.log(max(self.constant, 1e-30))
        for feat in self.features:
            log_val += self.coeffs[feat] * math.log(float(kwargs[feat]))
        return float(math.exp(log_val))


def fit_log_linear(
    records: Sequence[dict[str, Any]],
    target: str,
    features: Sequence[str] = ("n", "k", "T", "B"),
) -> LogLinearLaw:
    """Fit an ordinary-least-squares log-linear scaling law over positive records."""
    feats = tuple(features)
    rows_x: list[list[float]] = []
    rows_y: list[float] = []
    for rec in records:
        y_val = float(rec[target])
        if y_val <= 0.0:
            continue
        x_vals = [float(rec[f]) for f in feats]
        if any(v <= 0.0 for v in x_vals):
            continue
        rows_x.append([1.0] + [math.log(v) for v in x_vals])
        rows_y.append(math.log(y_val))

    if len(rows_x) < len(feats) + 1:
        raise ValueError(f"Need at least {len(feats) + 1} valid records to fit {feats}, got {len(rows_x)}")

    x_mat = np.asarray(rows_x, dtype=np.float64)
    y_vec = np.asarray(rows_y, dtype=np.float64)
    beta, _, _, _ = np.linalg.lstsq(x_mat, y_vec, rcond=None)
    y_hat = x_mat @ beta
    ss_res = float(np.sum((y_vec - y_hat) ** 2))
    ss_tot = float(np.sum((y_vec - np.mean(y_vec)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 1.0
    coeffs = {feat: float(beta[i + 1]) for i, feat in enumerate(feats)}
    return LogLinearLaw(
        target=target,
        features=feats,
        constant=float(math.exp(beta[0])),
        coeffs=coeffs,
        r2=float(r2),
    )


def find_batch_knee(batch_sizes: Sequence[int], losses: Sequence[float], rel_tol: float = 0.002) -> int:
    """Interpolate the largest batch size within ``(1 + rel_tol) * min_loss``."""
    pairs = sorted(zip(batch_sizes, losses), key=lambda p: p[0])
    min_loss = min(loss for _, loss in pairs)
    thresh = min_loss * (1.0 + rel_tol)
    last_good_b, last_good_l = pairs[0]
    for b, loss in pairs:
        if loss <= thresh:
            last_good_b, last_good_l = b, loss
        else:
            if b > last_good_b and loss > last_good_l:
                frac = (thresh - last_good_l) / (loss - last_good_l)
                log_b = math.log(last_good_b) + frac * (math.log(b) - math.log(last_good_b))
                return int(round(math.exp(log_b)))
            break
    return int(last_good_b)


def run_sweep(
    run_fn: Callable[[dict[str, Any]], dict[str, Any]],
    configs: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate a sequence of hyperparameter/scale configs and collect result records."""
    return [run_fn(dict(cfg)) for cfg in configs]

