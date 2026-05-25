"""Regression tests for IQL Stage 1 Final-6 selection."""

from __future__ import annotations

import pytest

from scripts.evaluate_iql_sweep import select_final6_configs


def _row(
    config_id: str,
    *,
    reward_variant: str = "sparse",
    lr_regime: str = "baseline",
    iql_setting: str = "baseline",
    fqe_mean: float = 1.0,
    fqe_95ci_lower: float = 0.8,
    wis_mean: float = 0.9,
    ess: float = 80.0,
    support_mass: float = 0.9,
    low_support_rate: float = 0.1,
    clinician_agreement: float = 0.35,
    action_entropy: float = 2.0,
    severe_safety_flags: int = 0,
    minor_safety_flags: int = 0,
) -> dict[str, object]:
    return {
        "config_id": config_id,
        "reward_variant": reward_variant,
        "lr_regime": lr_regime,
        "iql_setting": iql_setting,
        "fqe_mean": fqe_mean,
        "fqe_95ci_lower": fqe_95ci_lower,
        "wis_mean": wis_mean,
        "ess": ess,
        "support_mass": support_mass,
        "low_support_rate": low_support_rate,
        "clinician_agreement": clinician_agreement,
        "action_entropy": action_entropy,
        "severe_safety_flags": severe_safety_flags,
        "minor_safety_flags": minor_safety_flags,
    }


def test_final6_selection_rejects_high_fqe_low_support_candidate() -> None:
    candidates = [
        _row("unsafe_high_fqe", fqe_mean=10.0, support_mass=0.5, low_support_rate=0.4),
        _row("iql_sparse_baseline_safe", iql_setting="safe", fqe_mean=2.0),
        _row("iql_sofa_shaped_baseline_safe", reward_variant="sofa_shaped", iql_setting="safe", fqe_mean=1.9),
        _row("sparse_top", fqe_mean=2.4, wis_mean=2.0),
        _row("sofa_top", reward_variant="sofa_shaped", fqe_mean=2.3, wis_mean=2.1),
        _row("stable", fqe_mean=2.1, ess=150.0, support_mass=0.98, low_support_rate=0.02),
        _row("diverse", reward_variant="sofa_shaped", lr_regime="conservative", fqe_mean=2.05),
    ]

    result = select_final6_configs(candidates, clinician_fqe_lower=0.0, min_entropy=1.0)

    selected_ids = [row["config_id"] for row in result.selected]
    assert "unsafe_high_fqe" not in selected_ids
    assert len(selected_ids) == 6
    assert all(row["passed_hard_gate"] is True for row in result.selected)
    assert any(row["selection_slot"] == "safety_support" for row in result.selected)


def test_final6_selection_requires_at_least_one_gated_candidate() -> None:
    with pytest.raises(ValueError, match="No Stage 1 candidates passed"):
        select_final6_configs(
            [_row("bad", fqe_mean=3.0, ess=10.0, support_mass=0.4, low_support_rate=0.5)],
            clinician_fqe_lower=0.0,
            min_entropy=1.0,
        )
