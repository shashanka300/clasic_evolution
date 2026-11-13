"""Strict epoch budget accounting and distance evaluation tracking for Track T09.

Mathematical foundation:
- 1 epoch budget == N * k pairwise distance evaluations.
- Seeding distance evaluations: D_seed.
- Seeding cost in epoch units: s = D_seed / (N * k).
- Total available budget: T in {1.0, 2.0, 3.0} epochs.
- Remaining clustering budget: T_clust = max(0.0, T - s).
- Steps per epoch: spe = max(1, N // B).
- Clustering steps:
    If T_clust <= 0.0: M_clust = 0
    Else: M_clust = max(1, int(round(T_clust * spe)))
- Clustering distance evaluations:
    Standard / seeded: D_clust = M_clust * B * k.
    Split-as-you-go: sum_{t=0}^{M_clust-1} B * k_active(t).
- Total distance evaluations: D_total = D_seed + D_clust.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


@dataclass(frozen=True)
class BudgetPlan:
    """Rigorous compute budget allocation between seeding and clustering."""

    total_budget_epochs: float
    seeding_cost_epochs: float
    clustering_epochs: float
    dist_evals_seed: int
    dist_evals_clust: int
    dist_evals_total: int
    dist_evals_budget: int
    steps_per_epoch: int
    clustering_steps: int
    batch_size: int
    n: int
    k: int

    @property
    def budget_fraction_used(self) -> float:
        """Fraction of total budget distance evaluations used."""
        if self.dist_evals_budget <= 0:
            return 0.0
        return float(self.dist_evals_total) / float(self.dist_evals_budget)

    @property
    def is_within_budget(self) -> bool:
        """Whether total evaluations are within 5% tolerance of budget."""
        if self.dist_evals_budget <= 0:
            return True
        return self.dist_evals_total <= int(round(1.05 * self.dist_evals_budget))

    def to_dict(self) -> dict[str, Any]:
        """Convert budget plan to dictionary for logging in extra metadata."""
        return asdict(self)


def compute_budget_plan(
    n: int,
    k: int,
    total_epochs: float,
    batch_size: int,
    dist_evals_seed: int,
) -> BudgetPlan:
    """Compute exact budget allocation and mini-batch step horizon.

    Parameters
    ----------
    n : int
        Dataset size.
    k : int
        Number of clusters.
    total_epochs : float
        Total budget T in epochs (e.g. 1.0, 2.0, 3.0).
    batch_size : int
        Mini-batch size B.
    dist_evals_seed : int
        Number of pairwise distance evaluations consumed during seeding.

    Returns
    -------
    BudgetPlan
        Fully specified budget plan.
    """
    epoch_dist_evals = n * k
    seeding_cost_epochs = float(dist_evals_seed) / float(epoch_dist_evals) if epoch_dist_evals > 0 else 0.0
    clustering_epochs = max(0.0, float(total_epochs) - seeding_cost_epochs)

    b = min(max(int(batch_size), 1), n)
    steps_per_epoch = max(n // b, 1)

    # If clustering_epochs is 0, no clustering steps are run
    # Use floor to strictly guarantee total evaluations <= total budget (no overrun)
    raw_steps = clustering_epochs * float(steps_per_epoch)
    if clustering_epochs <= 0.0 or raw_steps < 1.0:
        clustering_steps = 0
    else:
        clustering_steps = max(0, int(math.floor(raw_steps)))

    dist_evals_clust = clustering_steps * b * k
    dist_evals_total = dist_evals_seed + dist_evals_clust
    dist_evals_budget = int(round(float(total_epochs) * float(epoch_dist_evals)))

    return BudgetPlan(
        total_budget_epochs=float(total_epochs),
        seeding_cost_epochs=round(seeding_cost_epochs, 6),
        clustering_epochs=round(clustering_epochs, 6),
        dist_evals_seed=int(dist_evals_seed),
        dist_evals_clust=int(dist_evals_clust),
        dist_evals_total=int(dist_evals_total),
        dist_evals_budget=int(dist_evals_budget),
        steps_per_epoch=int(steps_per_epoch),
        clustering_steps=int(clustering_steps),
        batch_size=int(b),
        n=int(n),
        k=int(k),
    )


def compute_loss_ratios(
    candidate_loss: float,
    random_loss: float,
    unbudgeted_kmpp_loss: float | None = None,
) -> dict[str, float]:
    """Compute performance relative to random baseline and unbudgeted k-means++.

    Parameters
    ----------
    candidate_loss : float
        Test loss of the candidate / evaluated method.
    random_loss : float
        Test loss of uniform random initialization under equal budget.
    unbudgeted_kmpp_loss : float, optional
        Test loss of unbudgeted k-means++ (T full epochs of clustering after k-means++).

    Returns
    -------
    dict
        Dictionary containing 'ratio_vs_random' (< 1.0 is better than random)
        and 'recovery_fraction' (fraction of k-means++ gain achieved).
    """
    ratio_vs_random = candidate_loss / max(random_loss, 1e-12)
    metrics = {
        "ratio_vs_random": round(float(ratio_vs_random), 6),
    }

    if unbudgeted_kmpp_loss is not None:
        gain_possible = random_loss - unbudgeted_kmpp_loss
        if abs(gain_possible) > 1e-12:
            gain_achieved = random_loss - candidate_loss
            recovery = gain_achieved / gain_possible
            metrics["recovery_fraction"] = round(float(recovery), 6)
        else:
            metrics["recovery_fraction"] = 1.0 if candidate_loss <= random_loss else 0.0

    return metrics
