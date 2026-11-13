"""Track T01: gmm-ema — EM for Gaussian Mixture Models via EMA sufficient statistics."""

import sys
from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

try:
    from .gmm import (
        CovarianceType,
        GMMParameters,
        compute_batch_stats,
        compute_log_responsibilities,
        compute_responsibilities,
        evaluate_gmm_nll,
        init_gmm_parameters,
        m_step_from_stats,
        revive_gmm_components,
    )
    from .plugin import (
        GMMEmaPlugin,
        fit_full_batch_em,
        fit_gmm_ema,
        fit_online_em,
    )
except ImportError:
    from gmm import (
        CovarianceType,
        GMMParameters,
        compute_batch_stats,
        compute_log_responsibilities,
        compute_responsibilities,
        evaluate_gmm_nll,
        init_gmm_parameters,
        m_step_from_stats,
        revive_gmm_components,
    )
    from plugin import (
        GMMEmaPlugin,
        fit_full_batch_em,
        fit_gmm_ema,
        fit_online_em,
    )

__all__ = [
    "CovarianceType",
    "GMMParameters",
    "GMMEmaPlugin",
    "compute_batch_stats",
    "compute_log_responsibilities",
    "compute_responsibilities",
    "evaluate_gmm_nll",
    "fit_full_batch_em",
    "fit_gmm_ema",
    "fit_online_em",
    "init_gmm_parameters",
    "m_step_from_stats",
    "revive_gmm_components",
]
