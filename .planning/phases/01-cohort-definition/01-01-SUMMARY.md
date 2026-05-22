---
phase: 01-cohort-definition
plan: 01
status: completed
backfilled: 2026-05-22
requirements-completed: [COH-01]
key-files:
  created:
    - src/mimic_sepsis_rl/data/cohort/spec.py
    - src/mimic_sepsis_rl/cli/build_cohort.py
    - configs/cohort/default.yaml
    - docs/cohort_selection.md
    - tests/data/test_cohort_spec.py
provides:
  - "Executable adult ICU Sepsis-3 cohort rule contract"
  - "Dry-run cohort CLI scaffold backed by default YAML config"
  - "Research-facing cohort selection documentation and rule-completeness tests"
---

# Plan 01-01 Summary: Cohort Rules, CLI Scaffold, Config, and Docs

**Status:** ✅ Complete — summary backfilled against the current repository state on 2026-05-22.

## What Was Built

- `src/mimic_sepsis_rl/data/cohort/spec.py` defines the versioned cohort specification, inclusion/exclusion rule objects, identifier columns, and config-driven rule loading surface.
- `configs/cohort/default.yaml` externalizes the adult ICU Sepsis-3 cohort parameters so the definition is not hidden in notebooks or ad hoc scripts.
- `src/mimic_sepsis_rl/cli/build_cohort.py` provides an invokable dry-run cohort builder entrypoint for checking config/spec wiring before real MIMIC extraction.
- `docs/cohort_selection.md` records the study-population rules in methods-section language.
- `tests/data/test_cohort_spec.py` protects adult-only gating, ICU stay scope, required rule categories, and config binding.

## Verified Artifacts

| Path | Lines | Purpose |
|------|------:|---------|
| `src/mimic_sepsis_rl/data/cohort/spec.py` | 181 | Cohort rule and config contract |
| `src/mimic_sepsis_rl/cli/build_cohort.py` | 208 | Cohort builder CLI scaffold |
| `configs/cohort/default.yaml` | 49 | Default executable cohort config |
| `docs/cohort_selection.md` | 99 | Research-facing cohort definition |
| `tests/data/test_cohort_spec.py` | 194 | Rule-completeness regression tests |

## Verification Results

Refreshed as part of the planning-doc reconciliation:

```text
uv run pytest -q tests/data/test_cohort_spec.py tests/data/test_cohort_audit.py tests/data/test_onset_assignment.py tests/data/test_episode_grid.py tests/mdp/test_action_bins.py tests/mdp/test_rewards.py
149 passed in 0.38s
```

## Decisions Captured

- Cohort eligibility is a typed, versioned contract rather than a notebook-only convention.
- The dry-run CLI lets later phases validate config/spec wiring without requiring local MIMIC data.
- The written cohort-selection doc mirrors executable rule names so reporting and code stay aligned.

## Next Phase Readiness

Plan 01-02 can build auditable extraction outputs on top of the stable cohort spec, identifier columns, and CLI/config entrypoint created here.
