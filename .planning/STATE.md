# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-28)

**Core value:** Klinik olarak makul, veri sizintisina dayanikli ve yeniden uretilebilir bir offline RL benchmark'i olusturmak.
**Current focus:** All 9 phases and 17 plans are complete. The project is a finished research benchmark — remaining work is limited to commit hygiene, LaTeX deliverables, and optional Graphify updates.

## Current Position

Phase: 9 of 9 (Evaluation, Safety, and Reproducible Package)
Plan: 17 of 17 completed
Status: **COMPLETE** — All roadmap phases implemented, tested, documented, and planning docs reconciled.
Last activity: 2026-05-22 — `.planning` doc reconciliation + `.gitignore` hygiene committed

Progress: [██████████] 100% (9 of 9 phases complete)

## Repository Snapshot

**Codebase:**
- Source modules: 53 Python files under `src/mimic_sepsis_rl/`
- Test files: 28 Python files under `tests/`
- Total source LOC: ~13,400
- Python version: 3.12, managed with `uv`

**Package structure:**
```
src/mimic_sepsis_rl/
  data/         — cohort extraction, onset assignment, episodes, splits
  mdp/          — feature dictionary, extractors, preprocessing, actions, rewards
  datasets/     — transitions, replay buffer
  training/     — device abstraction, config, CQL, BCQ, IQL, registry, experiment runner, comparison
  evaluation/   — OPE (WIS/FQE/ESS), safety checks, ablation registry
  reporting/    — offline RL artifact generation, reproducibility bundle
  baselines/    — clinician, no-treatment, behavior cloning
  cli/          — build_cohort, build_episode_grid, build_transitions
```

**Key dependencies:** PyTorch, d3rlpy, Polars, PyArrow, scikit-learn, Hydra, MLflow, matplotlib, seaborn

**Configs:** `configs/{cohort,onset,splits,features,training}/` — 9 YAML files (including runtime.mps.yaml, runtime.cuda.yaml, cql.yaml, bcq.yaml, iql.yaml)

**Docs:** 10 markdown docs under `docs/` + CQL run report assets under `docs/assets/cql-run/`

**Trained artifacts:**
- CQL checkpoints: 3 epochs (160/180/200) with manifests under `checkpoints/cql/`
- IQL checkpoints: 3 epochs (160/180/200) with manifests under `checkpoints/iql/`
- Run logs: `runs/cql/`, `runs/iql/iql_baseline/`

**Data:** Processed Parquet artifacts under `data/processed/` (cohort, onset, episodes — gitignored)

**External deliverable:** `proje_onerisi_iql/` — Turkish IQL project proposal (LaTeX source + PDF)

## Performance Metrics

**Velocity:**
- Total plans completed: 17/17
- Completed phases: 9/9
- Planning doc coverage: 100% — every plan has a SUMMARY

**By Phase:**

| Phase | Plans | Total | Status |
|-------|-------|-------|--------|
| 1 | 2 | 2 | ✅ Complete |
| 2 | 2 | 2 | ✅ Complete |
| 3 | 1 | 1 | ✅ Complete |
| 4 | 2 | 2 | ✅ Complete |
| 5 | 2 | 2 | ✅ Complete |
| 6 | 2 | 2 | ✅ Complete |
| 7 | 2 | 2 | ✅ Complete |
| 8 | 2 | 2 | ✅ Complete |
| 9 | 2 | 2 | ✅ Complete |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table. Key decisions across the completed project:

- Phase 1 locks the adult ICU Sepsis-3 cohort before onset logic or MDP construction.
- Phase 2 assigns one selected onset or explicit unusable status, then materializes deterministic 4-hour episode windows.
- Phase 3 owns patient-level split manifests so all later scalers, bins, and transforms stay train-only.
- Phase 4 defines the state feature contract and train-only preprocessing path.
- Phase 5 uses a discrete 25-action vasopressor-by-IV-fluid grid and a versioned reward contract.
- The main training path stays device-agnostic so the same code runs on Apple Silicon `MPS`, NVIDIA `CUDA`, or CPU fallback.
- Custom PyTorch trainers are used instead of JAX; PyTorch remains the implementation framework for CQL, BCQ, and IQL.
- CQL is the reference algorithm; BCQ and IQL reuse `common.py`, `device.py`, and `TrainingConfig` without forking the runtime surface.
- BCQ and IQL emit the same checkpoint-manifest and JSONL curve envelope as CQL, so comparison tooling stays algorithm-agnostic.
- Evaluation claims remain retrospective research claims; OPE is not evidence of bedside efficacy.
- Comparison and reporting artifacts should flag dataset-contract drift instead of silently merging incompatible runs.

### Phase Completion Summary

**Phase 1 — Cohort Definition ✅**
- `src/mimic_sepsis_rl/data/cohort/spec.py`, `extract.py`, `audit.py`, `models.py`
- `src/mimic_sepsis_rl/cli/build_cohort.py`
- `configs/cohort/default.yaml`
- `docs/cohort_selection.md`
- Summaries: `01-01-SUMMARY.md`, `01-02-SUMMARY.md`

**Phase 2 — Onset Anchoring and Episode Grid ✅**
- `src/mimic_sepsis_rl/data/onset.py`, `onset_models.py`, `episodes.py`, `episode_models.py`
- `src/mimic_sepsis_rl/cli/build_episode_grid.py`
- `configs/onset/default.yaml`
- Summaries: `02-01-SUMMARY.md`, `02-02-SUMMARY.md`

**Phase 3 — Split Manifests and Leakage Boundaries ✅**
- `src/mimic_sepsis_rl/data/splits.py`
- `configs/splits/default.yaml`
- `docs/leakage_boundaries.md`
- Summary: `03-01-SUMMARY.md`

**Phase 4 — State Representation Pipeline ✅**
- `src/mimic_sepsis_rl/mdp/features/dictionary.py`, `extractors.py`, `builder.py`
- `configs/features/default.yaml`
- `docs/feature_dictionary.md`
- Summaries: `04-01-SUMMARY.md`, `04-02-SUMMARY.md`

**Phase 5 — Treatment and Reward Encoding ✅**
- `src/mimic_sepsis_rl/mdp/actions/vasopressors.py`, `fluids.py`, `bins.py`
- `src/mimic_sepsis_rl/mdp/rewards.py`, `reward_models.py`
- `docs/action_mapping.md`, `docs/reward_spec.md`
- Summaries: `05-01-SUMMARY.md`, `05-02-SUMMARY.md`

**Phase 6 — Transition Dataset and Baseline Benchmarks ✅**
- `src/mimic_sepsis_rl/datasets/transitions.py`, `replay_buffer.py`
- `src/mimic_sepsis_rl/cli/build_transitions.py`
- `src/mimic_sepsis_rl/baselines/*`
- `docs/baseline_benchmarks.md`
- Summaries: `06-01-SUMMARY.md`, `06-02-SUMMARY.md`

**Phase 7 — CQL Reference Training ✅**
- `src/mimic_sepsis_rl/training/device.py`, `config.py`, `common.py`, `cql.py`
- `configs/training/runtime.mps.yaml`, `runtime.cuda.yaml`, `cql.yaml`
- `docs/cql_training.md`, `docs/cql_run_report.md`, `docs/assets/cql-run/*`
- Summaries: `07-01-SUMMARY.md`, `07-02-SUMMARY.md`

**Phase 8 — Comparative Offline RL Experiments ✅**
- `src/mimic_sepsis_rl/training/registry.py`, `experiment_runner.py`, `bcq.py`, `iql.py`, `comparison.py`
- `configs/training/bcq.yaml`, `iql.yaml`
- `docs/model_comparison.md`
- Summaries: `08-01-SUMMARY.md`, `08-02-SUMMARY.md`

**Phase 9 — Evaluation, Safety, and Reproducible Package ✅**
- `src/mimic_sepsis_rl/evaluation/ope.py`, `safety.py`, `ablations.py`
- `src/mimic_sepsis_rl/reporting/package.py`, `offline_rl.py`
- `docs/evaluation_protocol.md`, `docs/reproducibility.md`
- Summaries: `09-01-SUMMARY.md`, `09-02-SUMMARY.md`

### Current Workspace Notes

- Working tree is clean (nothing uncommitted).
- `.gitignore` excludes `graphify-out/`, `commit_history/`, `proje_onerisi_iql/`, `.agent/`, `/data/`.
- `proje_onerisi_iql/` contains a Turkish IQL project proposal (LaTeX source + PDF + aux files), gitignored.
- `.planning/` has a SUMMARY file for every plan in `ROADMAP.md` (18 summaries across 9 phases).
- CQL and IQL training runs and checkpoints are committed; BCQ checkpoints have not been committed separately.

### Git History (recent)

```
48b8765 chore(gitignore): exclude local-generated directories
200c6c9 docs(planning): reconcile ROADMAP and STATE with completed implementation
5368f71 docs(planning): backfill Phase 1, 2, 5 plan summaries
428eeee feat(runs): add iql training runs logs and metrics
8c455f4 feat(checkpoints): add iql model checkpoints
b847ba6 test(reporting): cover offline RL artifact generation
5a95226 feat(training): publish report artifacts from offline RL trainers
7848517 feat(reporting): add offline RL reporting artifact bundle
c7f7a4f artifacts: add CQL run outputs and checkpoints
```

### Pending Todos

- (Optional) Commit BCQ training checkpoints if they exist locally.
- (Optional) Rerun `graphify update .` and `graphify export obsidian` if Graphify is part of the final deliverable workflow.
- (Optional) Clean up LaTeX auxiliary files in `proje_onerisi_iql/` if the PDF is final.

### Blockers/Concerns

- Retrospective OPE/safety outputs remain research evidence only; they must not be described as prospective bedside efficacy.
- Full clinical validity still depends on the real MIMIC-IV data extraction and externally reviewed cohort/onset assumptions.
- No BCQ training runs/checkpoints found in repo — only CQL and IQL have committed artifacts.

## Session Continuity

Last session: 2026-05-22 15:01 +03
Stopped at: All `.planning` docs reconciled; `.gitignore` hygiene committed; clean working tree
Resume file: None
