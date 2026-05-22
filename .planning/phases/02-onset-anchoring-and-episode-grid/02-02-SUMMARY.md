---
phase: 02-onset-anchoring-and-episode-grid
plan: 02
status: completed
backfilled: 2026-05-22
requirements-completed: [COH-03]
requires:
  - phase: 02-onset-anchoring-and-episode-grid
    provides: "Selected onset or unusable episode outputs from 02-01"
key-files:
  created:
    - src/mimic_sepsis_rl/data/episodes.py
    - src/mimic_sepsis_rl/data/episode_models.py
    - src/mimic_sepsis_rl/cli/build_episode_grid.py
    - tests/data/test_episode_grid.py
provides:
  - "Deterministic 4-hour episode window materialization"
  - "Typed episode-step records with stable boundaries and truncation metadata"
  - "CLI and tests for episode grid generation"
---

# Plan 02-02 Summary: Deterministic Episode Grid

**Status:** ✅ Complete — summary backfilled against the current repository state on 2026-05-22.

## What Was Built

- `src/mimic_sepsis_rl/data/episodes.py` builds deterministic episode-step windows from onset-relative bounds and records truncation/unusable reasons.
- `src/mimic_sepsis_rl/data/episode_models.py` defines typed episode grid and step metadata used by downstream feature, action, and reward builders.
- `src/mimic_sepsis_rl/cli/build_episode_grid.py` exposes an invokable dry-run path for validating window materialization.
- `tests/data/test_episode_grid.py` verifies step counts, start/end alignment, truncation behavior, unusable episodes, and deterministic output.

## Verified Artifacts

| Path | Lines | Purpose |
|------|------:|---------|
| `src/mimic_sepsis_rl/data/episodes.py` | 300 | Episode window generation logic |
| `src/mimic_sepsis_rl/data/episode_models.py` | 136 | Typed episode-step records |
| `src/mimic_sepsis_rl/cli/build_episode_grid.py` | 198 | CLI entrypoint for episode grids |
| `tests/data/test_episode_grid.py` | 354 | Boundary, truncation, and determinism tests |

## Verification Results

Refreshed as part of the planning-doc reconciliation:

```text
uv run pytest -q tests/data/test_cohort_spec.py tests/data/test_cohort_audit.py tests/data/test_onset_assignment.py tests/data/test_episode_grid.py tests/mdp/test_action_bins.py tests/mdp/test_rewards.py
149 passed in 0.38s
```

## Decisions Captured

- The project uses deterministic 4-hour windows anchored to the selected Sepsis-3 onset timestamp.
- Truncated and unusable episodes stay inspectable through explicit metadata.
- Stable step indices and window boundaries become the shared time axis for split, feature, action, reward, and transition construction.

## Next Phase Readiness

Phase 3 can create patient-level split manifests against deterministic episode outputs, and Phase 4 can build leakage-safe state vectors on the fixed time grid.
