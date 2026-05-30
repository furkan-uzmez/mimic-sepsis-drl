"""
Patient-level bootstrap confidence intervals for offline policy evaluation.

Implements percentile bootstrap CIs for FQE and WIS estimators with
resampling at the *episode* (patient) level to avoid within-episode
auto-correlation leakage.

References
----------
- Efron & Tibshirani (1993) "An Introduction to the Bootstrap"
- Gottesman et al. (2019) "Evaluating RL Algorithms in Observational
  Health Settings"
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from mimic_sepsis_rl.evaluation.ope import (
    ActionSelectionPolicy,
    FrozenFQEOutputs,
    HeldOutEpisode,
    compute_wis_and_ess,
)


@dataclass(frozen=True)
class BootstrapCI:
    """Percentile bootstrap confidence interval for one OPE metric."""

    mean: float
    lower: float
    upper: float
    ci_level: int
    n_resamples: int
    n_episodes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WISBootstrapCI(BootstrapCI):
    """Bootstrap CI for WIS with additional ESS diagnostic."""

    ess: float


def _episode_ids(episodes: Sequence[HeldOutEpisode]) -> np.ndarray:
    """Return array of episode IDs for resampling."""
    return np.array([ep.episode_id for ep in episodes])


def _episodes_by_ids(
    episodes: Sequence[HeldOutEpisode],
    ids: Sequence[str | int],
) -> list[HeldOutEpisode]:
    """Collect held-out episodes by a list of IDs (may duplicate for bootstrap)."""
    lookup = {ep.episode_id: ep for ep in episodes}
    return [lookup[eid] for eid in ids]


def _percentile_ci(
    samples: np.ndarray,
    ci: int,
) -> tuple[float, float, float]:
    """Return (mean, lower_percentile, upper_percentile) from bootstrap samples."""
    mean = float(np.mean(samples))
    alpha = (100 - ci) / 2.0
    lower = float(np.percentile(samples, alpha))
    upper = float(np.percentile(samples, 100 - alpha))
    return mean, lower, upper


def bootstrap_fqe(
    fqe_outputs: FrozenFQEOutputs,
    episodes: Sequence[HeldOutEpisode],
    policy: ActionSelectionPolicy,
    *,
    n_resamples: int = 2000,
    ci: int = 95,
    seed: int | None = None,
) -> BootstrapCI:
    """Compute patient-level bootstrap CI for FQE.

    Each bootstrap iteration resamples patients (episodes) with replacement,
    then computes the mean FQE over the resampled set. Clinical outcomes
    are naturally clustered within patients, so patient-level bootstrap is
    the correct approach.

    Parameters
    ----------
    fqe_outputs : FrozenFQEOutputs
        Pre-fitted FQE Q-value estimates keyed by episode ID.
    episodes : sequence of HeldOutEpisode
        Held-out trajectories (must all have entries in *fqe_outputs*).
    policy : ActionSelectionPolicy
        Target policy to evaluate.
    n_resamples : int
        Number of bootstrap resamples (default 2000).
    ci : int
        Confidence level in percent (default 95).
    seed : int or None
        Seed for reproducible bootstrap sampling.

    Returns
    -------
    BootstrapCI
        Mean FQE value, lower/upper CI bounds, and metadata.
    """
    if not episodes:
        raise ValueError("At least one episode is required for bootstrap CI.")
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    ep_ids = _episode_ids(episodes)
    n_episodes = len(ep_ids)
    fqe_samples = np.empty(n_resamples, dtype=np.float64)
    n_actions = 25  # frozen contract constant for 5×5 grid

    for i in range(n_resamples):
        resampled_ids = rng.choice(ep_ids, size=n_episodes, replace=True)
        resampled_episodes = _episodes_by_ids(episodes, list(resampled_ids))
        fqe_samples[i] = fqe_outputs.estimate_policy_value(
            policy,
            resampled_episodes,
            n_actions=n_actions,
        )

    mean, lower, upper = _percentile_ci(fqe_samples, ci)
    return BootstrapCI(
        mean=mean,
        lower=lower,
        upper=upper,
        ci_level=ci,
        n_resamples=n_resamples,
        n_episodes=n_episodes,
    )


def _wis_from_weights_returns(weights: np.ndarray, returns: np.ndarray) -> float:
    """Compute WIS from precomputed per-episode weights and returns."""
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return 0.0
    return float(np.sum(weights * returns) / weight_sum)


def bootstrap_wis(
    episodes: Sequence[HeldOutEpisode],
    policy: ActionSelectionPolicy,
    *,
    n_resamples: int = 2000,
    ci: int = 95,
    gamma: float = 1.0,
    max_importance_ratio: float | None = None,
    seed: int | None = None,
) -> WISBootstrapCI:
    """Compute patient-level bootstrap CI for WIS.

    Each bootstrap iteration resamples patients (episodes) with replacement,
    then recomputes WIS on the resampled set. ESS is computed on the full
    dataset (not bootstrapped) to avoid inflated estimates.

    Parameters
    ----------
    episodes : sequence of HeldOutEpisode
        Held-out trajectories.
    policy : ActionSelectionPolicy
        Target policy to evaluate.
    n_resamples : int
        Number of bootstrap resamples (default 2000).
    ci : int
        Confidence level in percent (default 95).
    gamma : float
        MDP discount factor for return computation.
    max_importance_ratio : float or None
        Cap on per-step importance ratios for WIS.
    seed : int or None
        Seed for reproducible bootstrap sampling.

    Returns
    -------
    WISBootstrapCI
        Mean WIS, lower/upper CI bounds, ESS, and metadata.
    """
    if not episodes:
        raise ValueError("At least one episode is required for bootstrap CI.")
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    n_episodes = len(episodes)
    full_metrics, per_episode = compute_wis_and_ess(
        episodes,
        policy,
        gamma=gamma,
        max_importance_ratio=max_importance_ratio,
    )
    weights = np.array([episode.importance_weight for episode in per_episode], dtype=np.float64)
    returns = np.array([episode.discounted_return for episode in per_episode], dtype=np.float64)

    # Reuse per-episode OPE terms; bootstrap only resamples row indices.
    sample_indices = rng.integers(0, n_episodes, size=(n_resamples, n_episodes))
    wis_samples = np.empty(n_resamples, dtype=np.float64)
    for i, indices in enumerate(sample_indices):
        wis_samples[i] = _wis_from_weights_returns(weights[indices], returns[indices])

    mean, lower, upper = _percentile_ci(wis_samples, ci)
    return WISBootstrapCI(
        mean=mean,
        lower=lower,
        upper=upper,
        ci_level=ci,
        n_resamples=n_resamples,
        n_episodes=n_episodes,
        ess=full_metrics.ess,
    )


__all__ = [
    "BootstrapCI",
    "WISBootstrapCI",
    "bootstrap_fqe",
    "bootstrap_wis",
]
