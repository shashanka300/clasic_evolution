"""Balanced Inverted-File Indexing (balanced-ivf) package."""

from __future__ import annotations

import sys
from pathlib import Path

_TRACK_DIR = Path(__file__).resolve().parent
if str(_TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(_TRACK_DIR))

from .balanced_ivf import (
    BalancedIVFIndex,
    BalancedKMeansIVFIndex,
    EMeansIVFIndex,
    FaissIVFBenchmark,
    SinkhornBalancedAssigner,
    compute_imbalance_factor,
    generate_skewed_data,
    pairwise_sq_dists,
    sinkhorn_knopp_transport,
)
from .plugin import BalancedIVFPlugin, fit_balanced_ivf_plugin

__all__ = [
    "BalancedIVFIndex",
    "BalancedIVFPlugin",
    "BalancedKMeansIVFIndex",
    "EMeansIVFIndex",
    "FaissIVFBenchmark",
    "SinkhornBalancedAssigner",
    "compute_imbalance_factor",
    "fit_balanced_ivf_plugin",
    "generate_skewed_data",
    "pairwise_sq_dists",
    "sinkhorn_knopp_transport",
]
