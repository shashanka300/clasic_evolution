"""Live Balanced Inverted File Index with Online Drift Rewarm (Track T13)."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Ensure dependencies are available
TRACK_ROOT = Path(__file__).resolve().parent.parent
T03_DIR = TRACK_ROOT / "balanced-ivf"
T08_DIR = TRACK_ROOT / "drift-rewarm"
for p in (T03_DIR, T08_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from balanced_ivf import SinkhornBalancedAssigner
from detector import CUSUM
from core.schedule import WarmRestartCosineSchedule
from core.stats import EMAStats


@dataclass
class IndexStats:
    """Statistics for an IVF index state."""

    imbalance_ratio: float  # max_list_size / mean_list_size
    p99_latency_ms: float
    mean_latency_ms: float
    recall_at_10: float
    total_compute_s: float


class LiveDriftBalancedIVF:
    """Continuously updated balanced IVF index with drift detection and warm restarts."""

    def __init__(
        self,
        d: int,
        k: int,
        sinkhorn_reg: float = 0.05,
        alpha0: float = 0.6,
        alpha_min: float = 0.05,
        rewarm_period: int = 20,
        drift_threshold: float = 8.0,
        nlist_probe: int = 8,
    ) -> None:
        self.d = d
        self.k = k
        self.nlist_probe = min(nlist_probe, k)
        self.assigner = SinkhornBalancedAssigner(epsilon=sinkhorn_reg, num_iters=10)
        self.detector = CUSUM(threshold=drift_threshold, drift_tol=0.02)
        self.schedule = WarmRestartCosineSchedule(
            alpha0=alpha0,
            period=rewarm_period,
            eps=0.1,
            t_mult=1.0,
        )
        self.step_idx: int = 0
        self.restarts_triggered: int = 0

        # Centroids and lists
        self.centroids: np.ndarray = np.zeros((k, d), dtype=np.float32)
        self.stats: EMAStats = EMAStats(
            counts=np.zeros(k, dtype=np.float32),
            sums=np.zeros((k, d), dtype=np.float32),
        )
        self.invlists: List[List[Tuple[int, np.ndarray]]] = [[] for _ in range(k)]
        self.is_initialized: bool = False

    def init_centroids(self, x: np.ndarray) -> None:
        """Initialize centroids using random sample or k-means."""
        n = len(x)
        rng = np.random.default_rng(42)
        chosen = rng.choice(n, size=self.k, replace=False)
        self.centroids = np.copy(x[chosen]).astype(np.float32)
        self.stats = EMAStats(
            counts=np.ones(self.k, dtype=np.float32) * 1e-4,
            sums=np.copy(self.centroids) * 1e-4,
        )
        self.is_initialized = True

    def insert_batch(self, batch_x: np.ndarray, doc_ids: np.ndarray, adapt: bool = True) -> bool:
        """Insert a streaming batch of vectors. If adapt=True, updates centroids and triggers rewarm on drift."""
        n_b = len(batch_x)
        if not self.is_initialized:
            self.init_centroids(batch_x)

        # 1. Balanced assignment using Sinkhorn optimal transport
        assignments, _, _ = self.assigner.assign(batch_x, self.centroids)

        # 2. Add to inverted posting lists
        for i in range(n_b):
            c_id = int(assignments[i])
            self.invlists[c_id].append((int(doc_ids[i]), batch_x[i]))

        if not adapt:
            return False

        # 3. Monitor reconstruction / assignment error for drift
        assigned_centroids = self.centroids[assignments]
        batch_mse = float(np.mean(np.sum((batch_x - assigned_centroids) ** 2, axis=1)))
        alarm = self.detector.update(batch_mse)

        if alarm:
            self.schedule.trigger_restart(self.step_idx)
            self.detector.reset()
            self.restarts_triggered += 1

        # 4. EMA update of centroids with cosine schedule
        alpha = self.schedule(self.step_idx)
        self.step_idx += 1

        for c_id in range(self.k):
            members = batch_x[assignments == c_id]
            if len(members) > 0:
                count_delta = float(len(members))
                sum_delta = np.sum(members, axis=0)
                # EMA update
                self.stats.counts[c_id] = (1.0 - alpha) * self.stats.counts[c_id] + alpha * count_delta
                self.stats.sums[c_id] = (1.0 - alpha) * self.stats.sums[c_id] + alpha * sum_delta
                if self.stats.counts[c_id] > 1e-6:
                    self.centroids[c_id] = self.stats.sums[c_id] / self.stats.counts[c_id]

        return alarm

    def query(self, query_x: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, float]:
        """Search top-k nearest neighbors across nlist_probe nearest inverted lists."""
        t0 = time.perf_counter()
        # Find closest centroids
        dists_to_c = np.sum((query_x[None, :] - self.centroids) ** 2, axis=1)
        probe_lists = np.argsort(dists_to_c)[: self.nlist_probe]

        candidates = []
        for c_id in probe_lists:
            candidates.extend(self.invlists[c_id])

        if not candidates:
            # Fallback
            return np.array([], dtype=np.int64), float((time.perf_counter() - t0) * 1000.0)

        cand_ids = np.array([c[0] for c in candidates], dtype=np.int64)
        cand_vecs = np.array([c[1] for c in candidates], dtype=np.float32)

        cand_dists = np.sum((query_x[None, :] - cand_vecs) ** 2, axis=1)
        top_idx = np.argsort(cand_dists)[:top_k]

        latency_ms = float((time.perf_counter() - t0) * 1000.0)
        return cand_ids[top_idx], latency_ms

    def evaluate(self, queries: np.ndarray, ground_truth_ids: np.ndarray, top_k: int = 10) -> IndexStats:
        """Evaluate retrieval recall, latency, and balance ratio."""
        n_q = len(queries)
        recalls = []
        latencies = []

        for q in range(n_q):
            res_ids, lat = self.query(queries[q], top_k=top_k)
            gt_set = set(ground_truth_ids[q])
            recall = len(set(res_ids).intersection(gt_set)) / max(len(gt_set), 1)
            recalls.append(recall)
            latencies.append(lat)

        list_sizes = [len(l) for l in self.invlists]
        max_size = max(list_sizes) if list_sizes else 1
        mean_size = np.mean(list_sizes) if list_sizes else 1.0
        imbalance = float(max_size / max(mean_size, 1e-4))

        return IndexStats(
            imbalance_ratio=imbalance,
            p99_latency_ms=float(np.percentile(latencies, 99)),
            mean_latency_ms=float(np.mean(latencies)),
            recall_at_10=float(np.mean(recalls)),
            total_compute_s=0.0,
        )
