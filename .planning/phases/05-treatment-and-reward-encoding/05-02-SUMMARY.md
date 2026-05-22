---
phase: 05-treatment-and-reward-encoding
plan: 02
status: completed
backfilled: 2026-05-22
requirements-completed: [RWD-01]
requires:
  - phase: 05-treatment-and-reward-encoding
    provides: "Episode-aligned treatment/action contract from 05-01"
key-files:
  created:
    - src/mimic_sepsis_rl/mdp/rewards.py
    - src/mimic_sepsis_rl/mdp/reward_models.py
    - tests/mdp/test_rewards.py
    - docs/reward_spec.md
provides:
  - "Terminal 90-day mortality reward contract"
  - "Configurable intermediate SOFA/lactate/MAP shaping surface"
  - "Reward diagnostics and regression coverage"
---

# Plan 05-02 Summary: Reward Contract and Diagnostics

**Status:** ✅ Complete — summary backfilled against the current repository state on 2026-05-22.

## What Was Built

- `src/mimic_sepsis_rl/mdp/rewards.py` computes terminal mortality rewards and optional intermediate shaping terms over episode transitions.
- `src/mimic_sepsis_rl/mdp/reward_models.py` defines typed reward configuration, per-step outputs, summaries, and diagnostics metadata.
- `docs/reward_spec.md` records the reward formula, sparse/full shaping modes, clinical rationale, and diagnostics expected before training.
- `tests/mdp/test_rewards.py` covers terminal outcome handling, SOFA delta shaping, lactate/MAP edge cases, reward summaries, and deterministic behavior.

## Verified Artifacts

| Path | Lines | Purpose |
|------|------:|---------|
| `src/mimic_sepsis_rl/mdp/rewards.py` | 515 | Reward calculation and diagnostics |
| `src/mimic_sepsis_rl/mdp/reward_models.py` | 164 | Typed reward configs and outputs |
| `docs/reward_spec.md` | 177 | Research-facing reward definition |
| `tests/mdp/test_rewards.py` | 498 | Terminal and shaping reward coverage |

## Verification Results

Refreshed as part of the planning-doc reconciliation:

```text
uv run pytest -q tests/data/test_cohort_spec.py tests/data/test_cohort_audit.py tests/data/test_onset_assignment.py tests/data/test_episode_grid.py tests/mdp/test_action_bins.py tests/mdp/test_rewards.py
149 passed in 0.38s
```

## Decisions Captured

- Terminal reward encodes 90-day outcome, while intermediate shaping is configurable and versioned for later ablations.
- Reward diagnostics are part of the contract so training does not consume opaque reward columns.
- The reward definition is documented separately from implementation, making methods reporting and future review easier.

## Next Phase Readiness

Phase 6 can combine state vectors, action IDs, reward outputs, and episode boundaries into replay-ready transition datasets and baseline benchmarks.
