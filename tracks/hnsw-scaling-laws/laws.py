"""HNSW Graph Index Scaling Laws and Auto-Tuning (Track T11)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Tuple, List, Dict
import time
import math
import numpy as np
import hnswlib

from core.scaling import LogLinearLaw, fit_log_linear


def compute_ground_truth_topk(queries: np.ndarray, base: np.ndarray, k: int = 10) -> np.ndarray:
    """Brute-force exact top-k nearest neighbors by L2 distance."""
    # (n_queries, n_base) pairwise squared L2 distances
    dists = np.sum((queries[:, None, :] - base[None, :, :]) ** 2, axis=-1)
    return np.argsort(dists, axis=1)[:, :k]


def compute_recall_at_k(approx_neighbors: np.ndarray, true_neighbors: np.ndarray) -> float:
    """Compute recall@k: average fraction of true nearest neighbors retrieved."""
    n_queries, k = approx_neighbors.shape
    recalls = []
    for q in range(n_queries):
        n_inter = len(set(approx_neighbors[q]).intersection(set(true_neighbors[q])))
        recalls.append(n_inter / k)
    return float(np.mean(recalls))


@dataclass
class HNSWPredictor:
    """Predicts optimal HNSW build parameters (M, efConstruction) from problem specifications."""

    m_law: LogLinearLaw
    ef_law: LogLinearLaw

    def predict(self, n: int, d: int, target_recall: float) -> Tuple[int, int]:
        """Predict optimal (M, efConstruction) for target recall."""
        # Feature: neg_log_err = -log(1 - recall)
        neg_log_err = max(-math.log(max(1.0 - target_recall, 1e-4)), 0.1)
        pred_m = self.m_law.predict(n=float(n), d=float(d), r=neg_log_err)
        pred_ef = self.ef_law.predict(n=float(n), d=float(d), r=neg_log_err)

        # Clip to valid operational ranges
        m = int(np.clip(round(pred_m), 6, 96))
        ef_c = int(np.clip(round(pred_ef), 24, 500))
        return m, ef_c


def fit_hnsw_scaling_laws(calibration_records: List[Dict[str, Any]]) -> HNSWPredictor:
    """Fit log-linear laws for M and efConstruction from calibration records."""
    # Fit M law: M = c * n^a * d^b * r^g
    m_law = fit_log_linear(calibration_records, target="M", features=("n", "d", "r"))
    # Fit efConstruction law: ef = c * n^a * d^b * r^g
    ef_law = fit_log_linear(calibration_records, target="ef_construction", features=("n", "d", "r"))
    return HNSWPredictor(m_law=m_law, ef_law=ef_law)


def evaluate_hnsw_index(
    train_x: np.ndarray,
    test_x: np.ndarray,
    ground_truth: np.ndarray,
    M: int,
    ef_construction: int,
    ef_search: int = 50,
    top_k: int = 10,
) -> Tuple[float, float, float]:
    """Build HNSW index and evaluate recall@k, queries-per-second, and build time."""
    n, d = train_x.shape
    index = hnswlib.Index(space="l2", dim=d)

    # 1. Build index
    t0 = time.perf_counter()
    index.init_index(max_elements=n, ef_construction=ef_construction, M=M)
    index.add_items(train_x, np.arange(n))
    build_time_s = time.perf_counter() - t0

    # 2. Query index
    index.set_ef(max(ef_search, top_k))
    t1 = time.perf_counter()
    labels, _ = index.knn_query(test_x, k=top_k)
    query_time_s = time.perf_counter() - t1

    qps = float(len(test_x) / max(query_time_s, 1e-6))
    recall = compute_recall_at_k(labels, ground_truth)

    return recall, qps, build_time_s

