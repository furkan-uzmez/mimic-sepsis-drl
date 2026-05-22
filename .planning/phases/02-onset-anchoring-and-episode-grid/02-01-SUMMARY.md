---
phase: 02-onset-anchoring-and-episode-grid
plan: 01
status: completed
backfilled: 2026-05-22
requirements-completed: [COH-02]
requires:
  - phase: 01-cohort-definition
    provides: "Auditable cohort outputs from Phase 1"
key-files:
  created:
    - src/mimic_sepsis_rl/data/onset.py
    - src/mimic_sepsis_rl/data/onset_models.py
    - configs/onset/default.yaml
    - tests/data/test_onset_assignment.py
provides:
  - "One selected sepsis onset or explicit unusable marker per ICU episode"
  - "Typed onset candidate, rejection, and assignment outputs"
  - "Configurable Sepsis-3 onset operationalization with ambiguity tests"
---

# Plan 02-01 Summary: Sepsis-3 Onset Assignment

**Status:** ✅ Complete — summary backfilled against the current repository state on 2026-05-22.

## What Was Built

- `src/mimic_sepsis_rl/data/onset.py` implements the onset assignment pipeline, candidate selection/rejection, dry-run surface, and one-onset-or-unusable contract.
- `src/mimic_sepsis_rl/data/onset_models.py` defines typed onset candidates, selected onset records, rejected candidates, and unusable episode markers.
- `configs/onset/default.yaml` records operational onset parameters outside code.
- `tests/data/test_onset_assignment.py` covers straightforward selection, ambiguous candidates, no-valid-onset cases, and the invariant that a usable ICU episode never receives multiple selected onset timestamps.

## Verified Artifacts

| Path | Lines | Purpose |
|------|------:|---------|
| `src/mimic_sepsis_rl/data/onset.py` | 641 | Sepsis onset assignment pipeline |
| `src/mimic_sepsis_rl/data/onset_models.py` | 128 | Typed onset assignment outputs |
| `configs/onset/default.yaml` | 57 | Default onset parameters |
| `tests/data/test_onset_assignment.py` | 387 | Onset uniqueness and unusable-case coverage |

## Verification Results

Refreshed as part of the planning-doc reconciliation:

```text
uv run pytest -q tests/data/test_cohort_spec.py tests/data/test_cohort_audit.py tests/data/test_onset_assignment.py tests/data/test_episode_grid.py tests/mdp/test_action_bins.py tests/mdp/test_rewards.py
149 passed in 0.38s
```

## Decisions Captured

- Ambiguous onset evidence is surfaced instead of silently selecting an arbitrary timestamp.
- Unusable episodes remain explicit records so later phases do not lose auditability.
- Onset parameters are configurable and test-covered, keeping the Sepsis-3 operationalization reviewable.

## Next Phase Readiness

Plan 02-02 can materialize deterministic 4-hour windows around the selected onset timestamps and carry unusable/truncation metadata forward.
