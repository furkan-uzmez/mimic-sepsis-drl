# IQL final sweep workflow (data checks -> grid -> Stage 1 -> Stage 2 -> report).
# Mock validation:
#   uv run snakemake -n --cores 1
#   uv run snakemake --cores 1 --config mock=true

MOCK = bool(config.get("mock", False))

SPLITS = ["train", "validation", "test"]
RESULT_ROOT = "results/iql_final"

COHORT = "data/processed/cohort/cohort.parquet"
ONSET = "data/processed/onset/onset_assignments.parquet"
EPISODES = "data/processed/episodes/episodes.parquet"
STEPS = "data/processed/episodes/episode_steps.parquet"
SPLIT_MANIFESTS = expand("data/splits/{split}_manifest.parquet", split=SPLITS)
SPLIT_SUMMARY = "data/splits/split_summary.json"
SOFA_REPLAY = expand("data/replay/replay_{split}.parquet", split=SPLITS)
SOFA_REPLAY_META = expand("data/replay/replay_{split}_meta.json", split=SPLITS)
SPARSE_REPLAY = expand("data/replay_sparse/replay_{split}.parquet", split=SPLITS)
SPARSE_REPLAY_META = expand("data/replay_sparse/replay_{split}_meta.json", split=SPLITS)
ACTION_BINS = "data/processed/actions/action_bins.json"
PREPROCESSING = "data/processed/features/state_vectors/preprocessing_artifacts.json"

PRESWEEP_AUDIT = f"{RESULT_ROOT}/audit/presweep_audit.json"
GRID_MANIFEST = f"{RESULT_ROOT}/grid/iql_grid_manifest.json"
STAGE1_MANIFEST = f"{RESULT_ROOT}/stage1/stage1_manifest.json"
VALIDATION_METRICS = f"{RESULT_ROOT}/stage1/validation_metrics.csv"
TOP5_CONFIGS = f"{RESULT_ROOT}/stage1/selection/top5_configs.csv"
FINAL6_CONFIGS = f"{RESULT_ROOT}/stage1/selection/final6_configs.json"
STAGE2_MANIFEST = f"{RESULT_ROOT}/stage2/finalists_manifest.json"
STAGE2_SUMMARY = f"{RESULT_ROOT}/stage2/stage2_summary.json"
STAGE2_SEED_SUMMARY = f"{RESULT_ROOT}/stage2/seed_summary.csv"
FINAL_REPORT = f"{RESULT_ROOT}/final_report.md"
FINAL_METRICS = f"{RESULT_ROOT}/final_metrics.json"
FINAL_COMPARISON = f"{RESULT_ROOT}/final_comparison.csv"
FINAL_FIGURES = [
    f"{RESULT_ROOT}/figures/fqe_vs_support.png",
    f"{RESULT_ROOT}/figures/seed_variance.png",
    f"{RESULT_ROOT}/figures/action_heatmap.png",
    f"{RESULT_ROOT}/figures/baseline_comparison.png",
    f"{RESULT_ROOT}/figures/bootstrap_ci.png",
]


rule all:
    input:
        FINAL_REPORT,
        FINAL_METRICS,
        FINAL_COMPARISON,
        STAGE2_SUMMARY,
        STAGE2_SEED_SUMMARY,
        FINAL_FIGURES,
        TOP5_CONFIGS,
        STAGE2_MANIFEST,
        PRESWEEP_AUDIT,


rule mock_pipeline_inputs:
    output:
        SOFA_REPLAY,
        SOFA_REPLAY_META,
        SPARSE_REPLAY,
        SPARSE_REPLAY_META,
        SPLIT_MANIFESTS,
        SPLIT_SUMMARY,
        ACTION_BINS,
        PREPROCESSING,
    shell:
        "uv run python scripts/mock_iql_workflow.py inputs"


rule sofa_replay:
    input:
        cohort=COHORT,
        onset=ONSET,
        episodes=EPISODES,
        steps=STEPS,
        splits=SPLIT_MANIFESTS,
    output:
        SOFA_REPLAY,
        SOFA_REPLAY_META,
    shell:
        "uv run python -m mimic_sepsis_rl.cli.build_transitions --reward-variant sofa_shaped"


rule sparse_replay:
    input:
        cohort=COHORT,
        onset=ONSET,
        episodes=EPISODES,
        steps=STEPS,
        splits=SPLIT_MANIFESTS,
    output:
        SPARSE_REPLAY,
        SPARSE_REPLAY_META,
    shell:
        "uv run python -m mimic_sepsis_rl.cli.build_transitions --reward-variant sparse --output-dir data/replay_sparse/"


rule presweep_audit:
    input:
        sofa=SOFA_REPLAY,
        sofa_meta=SOFA_REPLAY_META,
        sparse=SPARSE_REPLAY,
        sparse_meta=SPARSE_REPLAY_META,
        splits=SPLIT_MANIFESTS,
        split_summary=SPLIT_SUMMARY,
        action_bins=ACTION_BINS,
        preprocessing=PREPROCESSING,
    output:
        PRESWEEP_AUDIT,
    shell:
        "uv run python scripts/mock_iql_workflow.py audit --output {output}"
        if MOCK else
        "uv run python scripts/audit_iql_presweep.py --output {output}"


rule iql_grid_manifest:
    input:
        audit=PRESWEEP_AUDIT,
    output:
        GRID_MANIFEST,
    shell:
        "uv run python scripts/run_iql_sweep.py --stage 1 --output-root {RESULT_ROOT} --mock --dry-run"
        if MOCK else
        "uv run python scripts/run_iql_sweep.py --stage 1 --output-root {RESULT_ROOT} --dry-run"


rule iql_stage1_sweep:
    input:
        grid=GRID_MANIFEST,
        audit=PRESWEEP_AUDIT,
    output:
        manifest=STAGE1_MANIFEST,
        top5=TOP5_CONFIGS,
        final6=FINAL6_CONFIGS,
    shell:
        "uv run python scripts/run_iql_sweep.py --stage 1 --output-root {RESULT_ROOT} --mock --dry-run"
        if MOCK else
        "uv run python scripts/run_iql_sweep.py --stage 1 --output-root {RESULT_ROOT}"


rule iql_stage1_validation:
    input:
        manifest=STAGE1_MANIFEST,
        final6=FINAL6_CONFIGS,
    output:
        VALIDATION_METRICS,
    shell:
        "uv run python scripts/mock_iql_workflow.py validation --output {output}"
        if MOCK else
        "uv run python scripts/evaluate_iql_sweep.py --checkpoint-dir checkpoints/iql_final --output-dir {RESULT_ROOT}/stage1/evaluation"


rule iql_stage2_sweep:
    input:
        validation=VALIDATION_METRICS,
        final6=FINAL6_CONFIGS,
    output:
        manifest=STAGE2_MANIFEST,
    shell:
        "uv run python scripts/run_iql_sweep.py --stage 2 --output-root {RESULT_ROOT} --mock --dry-run"
        if MOCK else
        "uv run python scripts/run_iql_sweep.py --stage 2 --output-root {RESULT_ROOT}"


rule iql_stage2_summary:
    input:
        manifest=STAGE2_MANIFEST,
        stage1=STAGE1_MANIFEST,
    output:
        summary=STAGE2_SUMMARY,
        seed_summary=STAGE2_SEED_SUMMARY,
        metrics=FINAL_METRICS,
        comparison=FINAL_COMPARISON,
        report=FINAL_REPORT,
        figures=FINAL_FIGURES,
    shell:
        "uv run python scripts/build_iql_final_bundle.py --output-root {RESULT_ROOT} --stage2-manifest {input.manifest}"
