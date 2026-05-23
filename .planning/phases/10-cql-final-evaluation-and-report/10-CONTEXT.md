# Phase 10: CQL Final Evaluation and Report — Context

**Gathered:** 2026-05-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Run a multi-seed CQL hyperparameter sweep (reward shaping ablation), evaluate all checkpoints on the held-out validation/test splits with bootstrap confidence intervals, produce support/plausibility diagnostics, and generate publication-quality figures and tables for the project report. This phase is **CQL-only** — no BCQ, IQL, or algorithm comparison.

**In scope:**
- Multi-seed CQL training (5 seeds × 2 reward variants = 10 runs)
- Baseline evaluation (clinician, no-treatment, behavior cloning)
- Patient-level bootstrap confidence intervals for FQE/WIS
- Support diagnostics: behavior-support mass, low-support action rate
- Clinician agreement: exact-bin + adjacent-bin
- Learned-policy action heatmap vs clinician heatmap
- Reward decomposition plot (1 example patient)
- Cohort characteristics table, action distribution table
- Main results table (FQE ± CI, shaped vs sparse, CQL vs baselines)

**Out of scope:**
- BCQ / IQL / algorithm comparison (hoca sadece CQL istedi)
- Action grid ablation (5×5 is locked)
- DR/WDR estimator
- Subgroup analysis
- Delta-Q analysis
</domain>

<decisions>
## Implementation Decisions

### Reward ablation
- Two variants: **shaped** (SOFA-based intermediate + terminal survival) and **sparse** (terminal survival only)
- Single action grid: 5×5 bins
- All other contract elements frozen (state, split, episode rules)

### Learning rate sweep
- Three values: **1×10⁻⁴, 3×10⁻⁴, 1×10⁻³**
- Screened in Stage 1 with CQL alpha
- Selected because: lr is the most impactful hyperparameter after α_CQL

### CQL alpha sweep
- Four values: **0.05, 0.1, 0.5, 1.0**
- Included in Stage 1 broad screen (not deferred to Stage 2)
- Rationale: Sepsis CQL papers (Nambiar 2023, Offline Safe RL 2025) show best alpha often ≤ 0.1; 2025 paper found best α = 0.05
- Literature-backed range from sepsis offline RL studies

### Common evaluation reward
- **Critical:** FQE model selection uses a common terminal survival utility, NOT the training reward
- Shaped and sparse reward used only for training; evaluation always uses terminal ±15 survival/death
- Prevents unfair comparison between policies trained with different reward scales

### Multi-seed protocol (Tang & Wiens 2021)
- **Stage 1:** 1 seed (42) × 2 rewards × 3 lr × 4 alpha = 24 configs screened
- FQE ranking on validation set (common terminal utility) selects top 6
- **Stage 2:** 6 configs × 4 extra seeds (123, 456, 789, 1024) = 24 runs
- Total: 48 CQL training runs, ~6.5 hours
- **Best checkpoint:** Selected by validation FQE, NOT epoch 200 (prior run showed best at epoch 140)
- Same train/val/test split across all seeds
- Report mean ± 95% bootstrap CI per configuration

### Evaluation
- Primary metric: FQE with patient-level bootstrap CI
- Secondary: WIS + ESS for support awareness
- All evaluation on held-out test split only
- Baseline policies evaluated once (deterministic)

### Reporting
- 7 figures + 3 tables for project report
- Figures auto-generated via existing `reporting/offline_rl.py` + new diagnostic scripts
- Tables generated from Parquet data + evaluation outputs

### Claude's Discretion
- Exact bootstrap implementation (percentile vs BCa)
- Figure styling and layout
- Script organization (monolithic sweep vs individual scripts)
</decisions>

<specifics>
## Specific Ideas

- Reward decomposition plot picks 1 representative patient episode to show SOFA trajectory + reward signal
- Bootstrap uses 2000 resamples at patient level (not timestep level)
- Support threshold: behavior_prob < 0.01 and count < 5 → "low support"
- Adjacent-bin agreement: clinician action ±1 bin in either vasopressor or fluid dimension
- Cohort table stratified by 90-day mortality outcome (survivor vs non-survivor)
</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets (already implemented, just need to run)
- `src/mimic_sepsis_rl/training/cql.py` — CQL trainer with reporting artifact generation
- `src/mimic_sepsis_rl/evaluation/ope.py` — WIS, FQE, ESS (needs bootstrap wrapper)
- `src/mimic_sepsis_rl/evaluation/safety.py` — ClinicianAgreementSummary, ActionSupport, ActionHeatmap
- `src/mimic_sepsis_rl/reporting/offline_rl.py` — Plot generation (step_metrics, epoch_metrics, q_diagnostics, episode_reward_*, action_heatmap)
- `src/mimic_sepsis_rl/baselines/` — clinician, no-treatment, behavior_cloning
- `configs/training/cql.yaml` — CQL training config

### What Needs New Code
- `src/mimic_sepsis_rl/evaluation/bootstrap.py` — patient-level bootstrap CI (new module)
- `src/mimic_sepsis_rl/evaluation/safety.py` — `build_learned_policy_heatmap()` extension
- `scripts/run_cql_sweep.py` — orchestrates 10 CQL runs + baselines + evaluation
- `scripts/generate_report_figures.py` — collects all plots into final report bundle

### Established Patterns
- Training → reporting artifact generation is already wired in CQL trainer
- Safety module has `to_dict()` on all dataclasses for JSON export
- Configs use YAML with schema versioning
</code_context>

<deferred>
## Deferred Ideas

- Action grid ablation (3×3 vs 5×5) — not needed for CQL-only report
- DR/WDR estimator — overkill for single algorithm
- Subgroup analysis — separate project scope
- TRIPOD+AI checklist — report is not a journal submission
</deferred>

---

*Phase: 10-cql-final-evaluation-and-report*
*Context gathered: 2026-05-23*
