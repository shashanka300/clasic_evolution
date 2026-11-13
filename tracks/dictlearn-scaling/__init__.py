"""Track T04: dictlearn-scaling package exports."""

import sys
from pathlib import Path

_track_dir = Path(__file__).resolve().parent
if str(_track_dir) not in sys.path:
    sys.path.insert(0, str(_track_dir))

from dictlearn import (
    OnlineDictionaryLearning,
    OnlineNMF,
    block_coordinate_descent_update,
    fista_sparse_code,
    revive_dead_atoms,
)
from plugin import (
    DictionaryLearningPlugin,
    NMFPlugin,
    fit_dictlearn_emeans,
    fit_nmf_emeans,
)

__all__ = [
    "OnlineDictionaryLearning",
    "OnlineNMF",
    "block_coordinate_descent_update",
    "fista_sparse_code",
    "revive_dead_atoms",
    "DictionaryLearningPlugin",
    "NMFPlugin",
    "fit_dictlearn_emeans",
    "fit_nmf_emeans",
]
