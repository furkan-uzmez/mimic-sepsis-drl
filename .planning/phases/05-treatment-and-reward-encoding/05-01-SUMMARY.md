---
phase: 05-treatment-and-reward-encoding
plan: 01
status: completed
backfilled: 2026-05-22
requirements-completed: [ACT-01, ACT-02]
requires:
  - phase: 04-state-representation-pipeline
    provides: "Episode-aligned state and split contracts from Phase 4"
key-files:
  created:
    - src/mimic_sepsis_rl/mdp/actions/vasopressors.py
    - src/mimic_sepsis_rl/mdp/actions/fluids.py
    - src/mimic_sepsis_rl/mdp/actions/bins.py
    - tests/mdp/test_action_bins.py
    - docs/action_mapping.md
provides:
  - "Norepinephrine-equivalent vasopressor standardization"
  - "4-hour IV fluid aggregation"
  - "Train-only 5x5 treatment binning and 25-action mapping contract"
---

# Plan 05-01 Summary: Treatment Standardization and 25-Action Map

**Status:** ✅ Complete — summary backfilled against the current repository state on 2026-05-22.

## What Was Built

- `src/mimic_sepsis_rl/mdp/actions/vasopressors.py` standardizes vasopressor administrations into norepinephrine-equivalent dose surfaces aligned to episode windows.
- `src/mimic_sepsis_rl/mdp/actions/fluids.py` aggregates IV fluid exposure over the 4-hour step grid.
- `src/mimic_sepsis_rl/mdp/actions/bins.py` learns train-only treatment thresholds, preserves explicit zero-dose bins, and combines vasopressor/fluid bins into the fixed 25-action map.
- `docs/action_mapping.md` documents action-id decoding and train-only threshold policy for methods reporting and clinical review.
- `tests/mdp/test_action_bins.py` verifies binning, zero-dose handling, leakage-sensitive threshold fitting, and action-id stability.

## Verified Artifacts

| Path | Lines | Purpose |
|------|------:|---------|
| `src/mimic_sepsis_rl/mdp/actions/vasopressors.py` | 344 | Vasopressor dose standardization |
| `src/mimic_sepsis_rl/mdp/actions/fluids.py` | 277 | 4-hour IV fluid aggregation |
| `src/mimic_sepsis_rl/mdp/actions/bins.py` | 416 | Train-only action bin learning and mapping |
| `docs/action_mapping.md` | 137 | Human-readable action contract |
| `tests/mdp/test_action_bins.py` | 538 | Action-bin regression coverage |

## Verification Results

Refreshed as part of the planning-doc reconciliation:

```text
uv run pytest -q tests/data/test_cohort_spec.py tests/data/test_cohort_audit.py tests/data/test_onset_assignment.py tests/data/test_episode_grid.py tests/mdp/test_action_bins.py tests/mdp/test_rewards.py
149 passed in 0.38s
```

## Decisions Captured

- The action space is discrete: a 5x5 vasopressor-by-fluid grid with 25 total actions.
- Treatment thresholds are learned from train-split data only and then frozen for validation/test and downstream RL runs.
- Action IDs remain decodable back into treatment bins for safety review, heatmaps, and reporting.

## Next Phase Readiness

Plan 05-02 can compute terminal and shaped rewards against the same episode/action contract, and Phase 6 can export replay transitions without redefining treatment semantics.
