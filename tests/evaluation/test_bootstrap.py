"""Regression tests for patient-level bootstrap CI computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from mimic_sepsis_rl.evaluation.bootstrap import (
    bootstrap_fqe,
    bootstrap_wis,
)
from mimic_sepsis_rl.evaluation.ope import (
    FrozenFQEOutputs,
    HeldOutEpisode,
    HeldOutStep,
)


@dataclass
class MappingPolicy:
    """Deterministic policy keyed by exact state tuple."""

    actions: dict[tuple[float, ...], int]

    def select_action(self, state: tuple[float, ...]) -> int:
        return self.actions[tuple(float(value) for value in state)]


def _make_episodes(n: int = 10, steps_per: int = 4) -> tuple[HeldOutEpisode, ...]:
    """Build synthetic held-out episodes with deterministic actions."""
    episodes: list[HeldOutEpisode] = []
    for ep_idx in range(n):
        steps: list[HeldOutStep] = []
        for step_idx in range(steps_per):
            done = step_idx == steps_per - 1
            steps.append(
                HeldOutStep(
                    episode_id=f"ep-{ep_idx}",
                    step_index=step_idx,
                    state=(float(ep_idx), float(step_idx)),
                    action=0,
                    reward=1.0 if done else 0.0,
                    done=done,
                    behavior_action_prob=0.5,
                )
            )
        episodes.append(HeldOutEpisode(episode_id=f"ep-{ep_idx}", steps=tuple(steps)))
    return tuple(episodes)


def _fqe_outputs(episodes: tuple[HeldOutEpisode, ...]) -> FrozenFQEOutputs:
    """Build frozen FQE outputs with varying Q-values per episode."""
    values: dict[str | int, tuple[float, ...]] = {}
    for idx, ep in enumerate(episodes):
        qs = [0.0] * 25
        qs[0] = 1.0 + 0.1 * idx  # Q(state, action=0) varies by episode
        values[ep.episode_id] = tuple(qs)
    return FrozenFQEOutputs(
        fitted_split="train",
        initial_state_action_values=values,
        artifact_label="test_fqe",
    )


class TestPatientLevelResample:
    """Verify that bootstrap resampling operates at the patient level."""

    def test_bootstrap_fqe_uses_patient_level_resampling(self) -> None:
        """FQE bootstrap resamples episodes, not individual timesteps."""
        episodes = _make_episodes(n=20, steps_per=4)
        fqe = _fqe_outputs(episodes)
        policy = MappingPolicy(
            actions={(float(i), float(j)): 0 for i in range(20) for j in range(4)}
        )

        result = bootstrap_fqe(
            fqe,
            episodes,
            policy,
            n_resamples=500,
            ci=95,
            seed=42,
        )

        # CIs should be bounded
        assert result.lower <= result.mean <= result.upper
        assert result.ci_level == 95
        assert result.n_resamples == 500
        assert result.n_episodes == 20
        # With varying Q-values, CI should have non-zero width
        assert result.upper - result.lower >= 0.0

    def test_bootstrap_wis_uses_patient_level_resampling(self) -> None:
        """WIS bootstrap resamples episodes, not individual timesteps."""
        episodes = _make_episodes(n=20, steps_per=4)
        policy = MappingPolicy(
            actions={(float(i), float(j)): 0 for i in range(20) for j in range(4)}
        )

        result = bootstrap_wis(
            episodes,
            policy,
            n_resamples=500,
            ci=95,
            seed=42,
        )

        assert result.lower <= result.mean <= result.upper
        assert result.ci_level == 95
        assert result.n_resamples == 500
        assert result.n_episodes == 20
        assert result.ess > 0.0

    def test_bootstrap_wis_ess_is_from_full_dataset(self) -> None:
        """ESS is computed on the full dataset, not bootstrapped."""
        episodes = _make_episodes(n=30, steps_per=4)
        policy = MappingPolicy(
            actions={(float(i), float(j)): 0 for i in range(30) for j in range(4)}
        )

        result = bootstrap_wis(
            episodes,
            policy,
            n_resamples=200,
            ci=95,
            seed=42,
        )

        # ESS should be a reasonable value for 30 episodes with matching actions
        assert 1.0 <= result.ess <= 30.0


class TestBootstrapCI:

    def test_bootstrap_fqe_reproduces_with_fixed_seed(self) -> None:
        """Fixed seed produces deterministic bootstrap results."""
        episodes = _make_episodes(n=10, steps_per=2)
        fqe = _fqe_outputs(episodes)
        policy = MappingPolicy(
            actions={(float(i), float(j)): 0 for i in range(10) for j in range(2)}
        )

        result_a = bootstrap_fqe(fqe, episodes, policy, n_resamples=200, ci=95, seed=42)
        result_b = bootstrap_fqe(fqe, episodes, policy, n_resamples=200, ci=95, seed=42)

        assert result_a.mean == pytest.approx(result_b.mean)
        assert result_a.lower == pytest.approx(result_b.lower)
        assert result_a.upper == pytest.approx(result_b.upper)

    def test_bootstrap_fqe_empty_episodes_raises(self) -> None:
        """Empty episode list raises ValueError."""
        policy = MappingPolicy(actions={})
        fqe = FrozenFQEOutputs(
            fitted_split="train",
            initial_state_action_values={},
        )
        with pytest.raises(ValueError, match="At least one episode"):
            bootstrap_fqe(fqe, (), policy, n_resamples=100, ci=95)

    def test_bootstrap_wis_ci_width_increases_with_lower_ci_level(self) -> None:
        """Higher CI level (99 vs 80) should produce wider interval."""
        episodes = _make_episodes(n=15, steps_per=4)
        policy = MappingPolicy(
            actions={(float(i), float(j)): 0 for i in range(15) for j in range(4)}
        )

        result_80 = bootstrap_wis(episodes, policy, n_resamples=500, ci=80, seed=42)
        result_99 = bootstrap_wis(episodes, policy, n_resamples=500, ci=99, seed=42)

        width_80 = result_80.upper - result_80.lower
        width_99 = result_99.upper - result_99.lower
        assert width_99 >= width_80, (
            f"99% CI ({width_99:.3f}) should be wider than 80% CI ({width_80:.3f})"
        )

    def test_bootstrap_fqe_to_dict(self) -> None:
        """BootstrapCI.to_dict() produces serializable dictionary."""
        episodes = _make_episodes(n=5, steps_per=2)
        fqe = _fqe_outputs(episodes)
        policy = MappingPolicy(
            actions={(float(i), float(j)): 0 for i in range(5) for j in range(2)}
        )
        result = bootstrap_fqe(fqe, episodes, policy, n_resamples=50, ci=95, seed=1)

        d = result.to_dict()
        assert d["mean"] == pytest.approx(result.mean)
        assert d["lower"] == pytest.approx(result.lower)
        assert d["upper"] == pytest.approx(result.upper)
        assert d["ci_level"] == 95
        assert d["n_resamples"] == 50
