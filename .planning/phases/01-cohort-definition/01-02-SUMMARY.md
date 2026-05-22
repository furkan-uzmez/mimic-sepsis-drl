---
phase: 01-cohort-definition
plan: 02
status: completed
backfilled: 2026-05-22
requirements-completed: [COH-01]
requires:
  - phase: 01-cohort-definition
    provides: "Cohort rule spec, config, CLI scaffold, and cohort-selection docs from 01-01"
key-files:
  created:
    - src/mimic_sepsis_rl/data/cohort/extract.py
    - src/mimic_sepsis_rl/data/cohort/audit.py
    - src/mimic_sepsis_rl/data/cohort/models.py
    - tests/data/test_cohort_audit.py
provides:
  - "Structured included/excluded cohort artifacts"
  - "Row-level and aggregate cohort audit reporting"
  - "Regression coverage for inclusion/exclusion traceability"
---

# Plan 01-02 Summary: Auditable Cohort Extraction and Exclusion Reporting

**Status:** ✅ Complete — summary backfilled against the current repository state on 2026-05-22.

## What Was Built

- `src/mimic_sepsis_rl/data/cohort/extract.py` orchestrates cohort extraction from source rows into stable included/excluded outputs.
- `src/mimic_sepsis_rl/data/cohort/models.py` defines typed cohort rows and extraction result structures that downstream phases can depend on.
- `src/mimic_sepsis_rl/data/cohort/audit.py` summarizes inclusion and exclusion decisions and preserves traceable rule-level reasons.
- `tests/data/test_cohort_audit.py` verifies excluded rows carry reasons, included rows remain traceable, and missing-rule situations are surfaced.

## Verified Artifacts

| Path | Lines | Purpose |
|------|------:|---------|
| `src/mimic_sepsis_rl/data/cohort/extract.py` | 297 | Cohort extraction orchestration |
| `src/mimic_sepsis_rl/data/cohort/audit.py` | 173 | Inclusion/exclusion audit summaries |
| `src/mimic_sepsis_rl/data/cohort/models.py` | 67 | Typed cohort output models |
| `tests/data/test_cohort_audit.py` | 367 | Audit completeness and traceability tests |

## Verification Results

Refreshed as part of the planning-doc reconciliation:

```text
uv run pytest -q tests/data/test_cohort_spec.py tests/data/test_cohort_audit.py tests/data/test_onset_assignment.py tests/data/test_episode_grid.py tests/mdp/test_action_bins.py tests/mdp/test_rewards.py
149 passed in 0.38s
```

## Decisions Captured

- Cohort extraction emits auditable artifacts rather than opaque side effects.
- Included and excluded outputs share stable identifiers so later onset/window work can trace every stay back to its cohort decision.
- Exclusion reasons are first-class audit data, not post-hoc prose.

## Next Phase Readiness

Phase 2 can consume the auditable cohort output and assign sepsis onset anchors only for usable ICU episodes while preserving unusable-case reasoning.
