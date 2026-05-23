# Snakefile — MIMIC Sepsis Offline RL: Full Pipeline (Raw → Report)
#
# Usage:
#   snakemake -j1 all              # run everything sequentially
#   snakemake -j4 all              # run with 4 parallel jobs (safe: GPU training runs sequentially)
#   snakemake -j1 --dry-run        # see what would run
#   snakemake -j1 report           # only figures & tables
#   snakemake -j1 stage1_sweep     # only Stage 1 training + eval
#
# Server setup (one-time):
#   pip install snakemake
#   # Or: uv pip install snakemake

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MIMIC_RAW  = "data/raw/physionet.org/files/mimiciv/3.1"
COHORT     = "data/processed/cohort/cohort.parquet"
ONSET      = "data/processed/onset/onset_assignments.parquet"
EPISODES   = "data/processed/episodes/episodes.parquet"
STEPS      = "data/processed/episodes/episode_steps.parquet"
SPLIT_DIR  = "data/splits"
SPLIT_CFG  = "configs/splits/default.yaml"
ONSET_CFG  = "configs/onset/default.yaml"

SHAPED_REPLAY  = "data/replay/replay_train.parquet"
SPARSE_REPLAY  = "data/replay_sparse/replay_train.parquet"

STAGE1_MANIFEST  = "runs/cql_sweep/stage1_manifest.json"
STAGE1_EVAL      = "runs/cql_sweep/stage1_evaluation.json"
STAGE2_MANIFEST  = "runs/cql_sweep/stage2_manifest.json"
EVAL_SUMMARY     = "runs/cql_sweep/evaluation_summary.json"

# ---------------------------------------------------------------------------
# Rule: all — every output
# ---------------------------------------------------------------------------

rule all:
    input:
        EVAL_SUMMARY,
        expand("docs/assets/report/fig{n}.png", n=range(1, 9)),
        expand("docs/assets/report/table{n}.csv", n=range(1, 6)),

# ---------------------------------------------------------------------------
# Phase 1: Cohort extraction
# ---------------------------------------------------------------------------

rule cohort:
    """Extract Sepsis-3 cohort from MIMIC-IV raw tables."""
    input:
        raw = MIMIC_RAW,
    output:
        COHORT,
    shell:
        "python -m mimic_sepsis_rl.cli.build_cohort"

# ---------------------------------------------------------------------------
# Phase 2: Onset assignment
# ---------------------------------------------------------------------------

rule onset:
    """Assign Sepsis-3 onset times to cohort episodes."""
    input:
        cohort = COHORT,
        config = ONSET_CFG,
        raw    = MIMIC_RAW,
    output:
        ONSET,
    shell:
        "python -m mimic_sepsis_rl.data.onset"

# ---------------------------------------------------------------------------
# Phase 3: Episode grid construction
# ---------------------------------------------------------------------------

rule episodes:
    """Build 4-hour episode grids from onset assignments."""
    input:
        onset = ONSET,
        raw   = MIMIC_RAW,
    output:
        EPISODES,
        STEPS,
    shell:
        "python -m mimic_sepsis_rl.cli.build_episode_grid"

# ---------------------------------------------------------------------------
# Phase 4: Patient-level splits
# ---------------------------------------------------------------------------

rule splits:
    """Generate train/validation/test split manifests."""
    input:
        episodes = EPISODES,
        config   = SPLIT_CFG,
    output:
        f"{SPLIT_DIR}/train_manifest.parquet",
        f"{SPLIT_DIR}/validation_manifest.parquet",
        f"{SPLIT_DIR}/test_manifest.parquet",
    shell:
        "python -m mimic_sepsis_rl.data.splits --config {input.config}"

# ---------------------------------------------------------------------------
# Phase 5-6: Shaped replay buffer
# ---------------------------------------------------------------------------

rule shaped_replay:
    """Build shaped-reward replay buffers (train/val/test)."""
    input:
        cohort  = COHORT,
        onset   = ONSET,
        episodes = EPISODES,
        steps    = STEPS,
        splits   = f"{SPLIT_DIR}/train_manifest.parquet",
        raw      = MIMIC_RAW,
    output:
        SHAPED_REPLAY,
    shell:
        "python -m mimic_sepsis_rl.cli.build_transitions "
        "--reward-variant sofa_shaped "

# ---------------------------------------------------------------------------
# Phase 5-6: Sparse replay buffer
# ---------------------------------------------------------------------------

rule sparse_replay:
    """Build sparse-reward replay buffers (train/val/test)."""
    input:
        cohort  = COHORT,
        onset   = ONSET,
        episodes = EPISODES,
        steps    = STEPS,
        splits   = f"{SPLIT_DIR}/train_manifest.parquet",
        raw      = MIMIC_RAW,
    output:
        SPARSE_REPLAY,
    shell:
        "python -m mimic_sepsis_rl.cli.build_transitions "
        "--reward-variant sparse "
        "--output-dir data/replay_sparse/ "

# ---------------------------------------------------------------------------
# Phase 10: Stage 1 — Broad CQL sweep (24 runs, ~3h)
# ---------------------------------------------------------------------------

rule stage1_sweep:
    """Train 24 CQL configurations with seed=42 (broad screen)."""
    input:
        shaped = SHAPED_REPLAY,
        sparse = SPARSE_REPLAY,
    output:
        STAGE1_MANIFEST,
    shell:
        "python scripts/run_cql_sweep.py --stage 1"

# ---------------------------------------------------------------------------
# Phase 10: Stage 1 evaluation — rank by validation FQE
# ---------------------------------------------------------------------------

rule stage1_eval:
    """Evaluate Stage 1 checkpoints on validation split, rank by FQE."""
    input:
        manifest = STAGE1_MANIFEST,
    output:
        STAGE1_EVAL,
    shell:
        "python scripts/evaluate_cql_sweep.py --stage 1"

# ---------------------------------------------------------------------------
# Phase 10: Stage 2 — Multi-seed confirmation (24 runs, ~3h)
# ---------------------------------------------------------------------------

rule stage2_sweep:
    """Train top-6 configs with 4 extra seeds each."""
    input:
        eval_json = STAGE1_EVAL,
        shaped    = SHAPED_REPLAY,
        sparse    = SPARSE_REPLAY,
    output:
        STAGE2_MANIFEST,
    shell:
        "python scripts/run_cql_sweep.py "
        "--stage 2 "
        "--stage1-eval {input.eval_json} "

# ---------------------------------------------------------------------------
# Phase 10: Final evaluation — test split + bootstrap CIs
# ---------------------------------------------------------------------------

rule final_eval:
    """Evaluate all checkpoints on test split with bootstrap CIs."""
    input:
        manifest = STAGE2_MANIFEST,
    output:
        EVAL_SUMMARY,
    shell:
        "python scripts/evaluate_cql_sweep.py --stage final"

# ---------------------------------------------------------------------------
# Phase 10: Report — figures + tables
# ---------------------------------------------------------------------------

rule report:
    """Generate 8 figures, 5 tables, and draft report."""
    input:
        eval_summary = EVAL_SUMMARY,
    output:
        expand("docs/assets/report/fig{n}.png", n=range(1, 9)),
        expand("docs/assets/report/table{n}.csv", n=range(1, 6)),
    shell:
        "python scripts/generate_report_figures.py"


# ---------------------------------------------------------------------------
# Clean targets
# ---------------------------------------------------------------------------

rule clean_sweep:
    """Remove sweep outputs (checkpoints + runs) to free disk space."""
    shell:
        "rm -rf checkpoints/cql_sweep/ runs/cql_sweep/"

rule clean_replay:
    """Remove replay buffers."""
    shell:
        "rm -rf data/replay/ data/replay_sparse/"

rule clean_all:
    """Remove ALL generated data including processed artifacts."""
    shell:
        "rm -rf data/processed/ data/replay/ data/replay_sparse/ "
        "data/splits/ checkpoints/cql_sweep/ runs/cql_sweep/ "
        "docs/assets/report/"
