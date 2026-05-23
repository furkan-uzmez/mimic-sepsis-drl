#!/usr/bin/env python3
"""
CQL multi-seed sweep orchestrator with reward-shaping ablation.

Runs baselines (clinician, no-treatment, behaviour cloning) once, then
executes a full CQL sweep across seeds × reward variants.  Each CQL run
receives a temporary config YAML derived from the reference CQL config
with overridden seed, reward variant, and output directories.

Usage
-----
    uv run python scripts/run_cql_sweep.py
    uv run python scripts/run_cql_sweep.py --seeds 42,123,456,789,1024
    uv run python scripts/run_cql_sweep.py --rewards shaped,sparse

Reference
---------
- Task 3, Plan 10-01: Phase 10 CQL Final Evaluation and Report
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SWEEP_SCRIPT_VERSION: str = "1.0.0"

REFERENCE_CQL_CONFIG: str = "configs/training/cql.yaml"

# Reward variant name mapping (plan names → code enum values)
REWARD_VARIANT_MAP: dict[str, str] = {
    "shaped": "sofa_shaped",
    "sparse": "sparse",
}

# Seeds for the multi-seed CQL sweep
DEFAULT_SEEDS: tuple[int, ...] = (42, 123, 456, 789, 1024)
DEFAULT_REWARDS: tuple[str, ...] = ("shaped", "sparse")
DEFAULT_LEARNING_RATES: tuple[float, ...] = (1e-4, 3e-4, 1e-3)
DEFAULT_ALPHAS: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0)
DEFAULT_STAGE1_SEED: int = 42
STAGE2_EXTRA_SEEDS: tuple[int, ...] = (123, 456, 789, 1024)
STAGE2_TOP_N: int = 6

# Baseline labels and their config mapping
BASELINES: tuple[dict[str, str], ...] = (
    {"name": "clinician", "description": "Observed clinician action at each step"},
    {"name": "no_treatment", "description": "Zero-dose action (bin 0) at every step"},
    {
        "name": "bc",
        "description": "Behaviour cloning (supervised imitation of clinician actions)",
    },
)


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Outcome of a single CQL training run."""

    seed: int
    reward_variant: str
    learning_rate: float
    cql_alpha: float
    success: bool
    elapsed_sec: float
    output_dir: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SweepManifest:
    """Manifest recording all runs in the sweep."""

    version: str = SWEEP_SCRIPT_VERSION
    seeds: tuple[int, ...] = ()
    reward_variants: tuple[str, ...] = ()
    learning_rates: tuple[float, ...] = ()
    cql_alphas: tuple[float, ...] = ()
    stage: int = 1
    top_configs: list[dict[str, Any]] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    total_elapsed_sec: float = 0.0
    baselines: list[dict[str, Any]] = field(default_factory=list)
    runs: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_reference_config(config_path: str) -> dict[str, Any]:
    """Load the reference CQL YAML config."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Reference config not found: {config_path}")
    with path.open("r") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}.")
    return cfg

def _build_sweep_config(
    base: dict[str, Any],
    *,
    seed: int,
    reward_variant: str,
    learning_rate: float,
    cql_alpha: float,
) -> dict[str, Any]:
    """Build a deep-copied config with seed, reward, lr, alpha, and directory overrides."""
    import copy

    cfg = copy.deepcopy(base)

    # Resolve reward variant code name
    variant_code = REWARD_VARIANT_MAP.get(reward_variant, reward_variant)

    # --- Override runtime seed ---
    cfg.setdefault("runtime", {})
    cfg["runtime"]["seed"] = seed

    # --- Override learning rate and CQL alpha ---
    cfg.setdefault("extra", {})
    cfg["extra"]["lr"] = learning_rate
    cfg["extra"]["cql_alpha"] = cql_alpha

    # --- Override checkpoint/log directories ---
    lr_label = f"lr{learning_rate:.0e}".replace("e-0", "e-").replace("e-", "e-")
    alpha_label = f"a{str(cql_alpha).replace('.', 'p')}"
    run_name = f"cql_s{seed}_{variant_code}_{lr_label}_{alpha_label}"

    cfg.setdefault("checkpoint", {})
    cfg["checkpoint"]["checkpoint_dir"] = f"checkpoints/cql_sweep/{run_name}"
    cfg["checkpoint"]["save_every_n_epochs"] = 20
    cfg["checkpoint"]["keep_last_n"] = 0  # keep all for best-checkpoint selection

    cfg.setdefault("logging", {})
    cfg["logging"]["log_dir"] = "runs/cql_sweep"
    cfg["logging"]["experiment_name"] = run_name

    # --- Store reward variant as extra metadata ---
    cfg.setdefault("extra", {})
    cfg["extra"]["reward_variant"] = variant_code

    # --- Select dataset path per reward variant ---
    _DATASET_PATHS = {
        "shaped": {
            "dataset_path": "data/replay/replay_train.parquet",
            "dataset_meta_path": "data/replay/replay_train_meta.json",
        },
        "sparse": {
            "dataset_path": "data/replay_sparse/replay_train.parquet",
            "dataset_meta_path": "data/replay_sparse/replay_train_meta.json",
        },
    }
    paths = _DATASET_PATHS.get(reward_variant, _DATASET_PATHS["shaped"])
    cfg["dataset_path"] = paths["dataset_path"]
    cfg["dataset_meta_path"] = paths["dataset_meta_path"]

    return cfg


def _write_temp_config(cfg: dict[str, Any]) -> Path:
    """Write a temporary config YAML and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="cql_sweep_",
        delete=False,
    )
    yaml.safe_dump(cfg, tmp, default_flow_style=False)
    tmp.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Run execution
# ---------------------------------------------------------------------------


def _run_command(cmd: list[str], timeout_sec: int = 7200) -> subprocess.CompletedProcess:
    """Run a subprocess and return its result."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )


def run_cql_training(
    config_path: Path | str,
    *,
    dry_run: bool = False,
) -> subprocess.CompletedProcess:
    """Execute one CQL training run via the CLI entry point."""
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "mimic_sepsis_rl.training.cql",
        "--config",
        str(config_path),
    ]
    if dry_run:
        cmd.append("--dry-run")
    return _run_command(cmd)


# ---------------------------------------------------------------------------
# Sweep orchestrator
# ---------------------------------------------------------------------------


def run_baselines(output_dir: str) -> list[dict[str, Any]]:
    """Compute baseline metrics (stub — real baselines from Phase 6)."""
    results: list[dict[str, Any]] = []
    for baseline in BASELINES:
        logger.info("Baseline: %s (%s)", baseline["name"], baseline["description"])
        results.append(
            {
                "name": baseline["name"],
                "description": baseline["description"],
                "status": "available",  # baselines from Phase 6 are pre-computed
                "output_dir": str(Path(output_dir) / "baselines" / baseline["name"]),
            }
        )
    return results


def run_cql_sweep(
    seeds: tuple[int, ...],
    reward_variants: tuple[str, ...],
    learning_rates: tuple[float, ...],
    cql_alphas: tuple[float, ...],
    *,
    stage: int = 1,
    stage1_eval_path: str | None = None,
    reference_config_path: str = REFERENCE_CQL_CONFIG,
    dry_run: bool = False,
) -> tuple[list[RunResult], SweepManifest]:
    """Two-stage offline hyperparameter selection protocol.

    Stage 1: 1 seed × 2 rewards × 3 lr × 3 alpha = 18 configs screened.
    Stage 2: Reads top-N configs from stage 1 evaluation, adds extra seeds (24 runs).

    Reference: Tang & Wiens 2021 two-stage model selection for healthcare RL.
    """
    base_cfg = _load_reference_config(reference_config_path)
    results: list[RunResult] = []

    if stage == 1:
        return _run_stage1(
            base_cfg, reward_variants, learning_rates, cql_alphas, dry_run
        )
    elif stage == 2:
        if not stage1_eval_path:
            raise ValueError("--stage1-eval required for stage 2")
        return _run_stage2(
            base_cfg, results, stage1_eval_path, dry_run
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")


def _run_stage1(
    base_cfg: dict[str, Any],
    reward_variants: tuple[str, ...],
    learning_rates: tuple[float, ...],
    cql_alphas: tuple[float, ...],
    dry_run: bool,
) -> tuple[list[RunResult], SweepManifest]:
    """Stage 1: Broad single-seed screen over all (reward, lr, alpha) combos."""
    results: list[RunResult] = []
    manifest = SweepManifest(
        seeds=(DEFAULT_STAGE1_SEED,),
        reward_variants=reward_variants,
        learning_rates=learning_rates,
        cql_alphas=cql_alphas,
        stage=1,
        start_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    total = len(reward_variants) * len(learning_rates) * len(cql_alphas)
    completed = 0
    failed = 0
    sweep_start = time.monotonic()

    logger.info(
        "=== Stage 1: Broad screen === %d rewards × %d lr × %d alpha = %d configs",
        len(reward_variants), len(learning_rates), len(cql_alphas), total,
    )

    for reward in reward_variants:
        for lr in learning_rates:
            for alpha in cql_alphas:
                seed = DEFAULT_STAGE1_SEED
                label = f"reward={reward}, lr={lr:.0e}, α={alpha}"
                run_start = time.monotonic()
                temp_config_path: Path | None = None

                try:
                    sweep_cfg = _build_sweep_config(
                        base_cfg,
                        seed=seed,
                        reward_variant=reward,
                        learning_rate=lr,
                        cql_alpha=alpha,
                    )
                    temp_config_path = _write_temp_config(sweep_cfg)
                    logger.info("[%d/%d] %s", completed + failed + 1, total, label)
                    proc = run_cql_training(temp_config_path, dry_run=dry_run)
                    elapsed = time.monotonic() - run_start

                    result = RunResult(
                        seed=seed,
                        reward_variant=reward,
                        learning_rate=lr,
                        cql_alpha=alpha,
                        success=proc.returncode == 0,
                        elapsed_sec=elapsed,
                        output_dir=str(Path("runs/cql_sweep") / f"cql_s{seed}_{REWARD_VARIANT_MAP.get(reward, reward)}"),
                    )
                    results.append(result)
                    if proc.returncode == 0:
                        completed += 1
                        logger.info("  ✓ %s (%.1fs)", label, elapsed)
                    else:
                        failed += 1
                        result.error_message = proc.stderr[:500] if proc.stderr else "Unknown error"
                        logger.error("  ✗ %s failed (rc=%d)", label, proc.returncode)
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - run_start
                    failed += 1
                    results.append(RunResult(
                        seed=seed, reward_variant=reward, learning_rate=lr,
                        cql_alpha=alpha, success=False, elapsed_sec=elapsed,
                        error_message="Timeout expired",
                    ))
                    logger.error("  ✗ %s timed out", label)
                except Exception as exc:
                    elapsed = time.monotonic() - run_start
                    failed += 1
                    results.append(RunResult(
                        seed=seed, reward_variant=reward, learning_rate=lr,
                        cql_alpha=alpha, success=False, elapsed_sec=elapsed,
                        error_message=str(exc),
                    ))
                    logger.error("  ✗ %s error: %s", label, exc)
                finally:
                    if temp_config_path is not None and temp_config_path.exists():
                        try:
                            temp_config_path.unlink()
                        except OSError:
                            pass

    manifest.end_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest.total_elapsed_sec = time.monotonic() - sweep_start
    manifest.runs = [r.to_dict() for r in results]
    manifest.summary = {"total": total, "completed": completed, "failed": failed}

    output_dir = Path("runs/cql_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "stage1_manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2))
    logger.info("Stage 1 manifest → %s (%d/%d done)", manifest_path, completed, total)

    return results, manifest


def _run_stage2(
    base_cfg: dict[str, Any],
    _results: list[RunResult],
    eval_path: str,
    dry_run: bool,
) -> tuple[list[RunResult], SweepManifest]:
    """Stage 2: Add 4 extra seeds to top-N configs from stage 1 evaluation."""
    eval_data = json.loads(Path(eval_path).read_text())

    # Try multiple key formats
    top_configs = eval_data.get("top_configs") or eval_data.get("top_configs", [])
    if not top_configs:
        rankings = eval_data.get("rankings") or eval_data.get("ranking", [])
        top_configs = rankings[:STAGE2_TOP_N]

    if not top_configs:
        raise ValueError(f"No top_configs/rankings found in {eval_path}")

    results: list[RunResult] = []
    manifest = SweepManifest(
        seeds=STAGE2_EXTRA_SEEDS,
        stage=2,
        top_configs=top_configs[:STAGE2_TOP_N],
        start_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    total = sum(len(STAGE2_EXTRA_SEEDS) for _ in top_configs[:STAGE2_TOP_N])
    completed = 0
    failed = 0
    sweep_start = time.monotonic()

    logger.info("=== Stage 2: Multi-seed confirmation === %d configs × %d seeds = %d runs",
                min(STAGE2_TOP_N, len(top_configs)), len(STAGE2_EXTRA_SEEDS), total)

    for i, config in enumerate(top_configs[:STAGE2_TOP_N]):
        reward = config.get("reward_variant", config.get("reward", "shaped"))
        lr = float(config.get("learning_rate", config.get("lr", 3e-4)))
        alpha = float(config.get("cql_alpha", config.get("alpha", 1.0)))

        for seed in STAGE2_EXTRA_SEEDS:
            label = f"#{i+1} reward={reward}, lr={lr:.0e}, α={alpha}, seed={seed}"
            run_start = time.monotonic()
            temp_config_path: Path | None = None

            try:
                sweep_cfg = _build_sweep_config(
                    base_cfg, seed=seed, reward_variant=reward,
                    learning_rate=lr, cql_alpha=alpha,
                )
                temp_config_path = _write_temp_config(sweep_cfg)
                logger.info("[%d/%d] %s", completed + failed + 1, total, label)
                proc = run_cql_training(temp_config_path, dry_run=dry_run)
                elapsed = time.monotonic() - run_start

                result = RunResult(
                    seed=seed, reward_variant=reward, learning_rate=lr,
                    cql_alpha=alpha, success=proc.returncode == 0,
                    elapsed_sec=elapsed,
                    output_dir=str(Path("runs/cql_sweep") / f"cql_s{seed}_{REWARD_VARIANT_MAP.get(reward, reward)}"),
                )
                results.append(result)
                if proc.returncode == 0:
                    completed += 1
                    logger.info("  ✓ %s (%.1fs)", label, elapsed)
                else:
                    failed += 1
                    result.error_message = proc.stderr[:500] if proc.stderr else "Unknown error"
                    logger.error("  ✗ %s failed", label)
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - run_start
                failed += 1
                results.append(RunResult(
                    seed=seed, reward_variant=reward, learning_rate=lr,
                    cql_alpha=alpha, success=False, elapsed_sec=elapsed,
                    error_message="Timeout expired",
                ))
            except Exception as exc:
                elapsed = time.monotonic() - run_start
                failed += 1
                results.append(RunResult(
                    seed=seed, reward_variant=reward, learning_rate=lr,
                    cql_alpha=alpha, success=False, elapsed_sec=elapsed,
                    error_message=str(exc),
                ))
            finally:
                if temp_config_path is not None and temp_config_path.exists():
                    try:
                        temp_config_path.unlink()
                    except OSError:
                        pass

    manifest.end_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest.total_elapsed_sec = time.monotonic() - sweep_start
    manifest.runs = [r.to_dict() for r in results]
    manifest.summary = {"total": total, "completed": completed, "failed": failed}

    output_dir = Path("runs/cql_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "stage2_manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2))
    logger.info("Stage 2 manifest → %s (%d/%d done)", manifest_path, completed, total)

    return results, manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Entry point for the CQL sweep orchestrator."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        prog="run_cql_sweep",
        description="Run multi-seed CQL sweep with reward shaping ablation.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated seeds (default: %(default)s).",
    )
    parser.add_argument(
        "--rewards",
        type=str,
        default=",".join(DEFAULT_REWARDS),
        help="Comma-separated reward variants: shaped, sparse (default: %(default)s).",
    )
    parser.add_argument(
        "--learning-rates",
        type=str,
        default=",".join(str(lr) for lr in DEFAULT_LEARNING_RATES),
        help="Comma-separated learning rates (default: %(default)s).",
    )
    parser.add_argument(
        "--cql-alphas",
        type=str,
        default=",".join(str(a) for a in DEFAULT_ALPHAS),
        help="Comma-separated CQL alpha values (default: %(default)s).",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        choices=[1, 2],
        help="Stage 1 (broad screen) or Stage 2 (multi-seed confirmation, default: 1).",
    )
    parser.add_argument(
        "--stage1-eval",
        type=str,
        default=None,
        help="Path to stage 1 evaluation JSON (required for --stage 2).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=REFERENCE_CQL_CONFIG,
        help="Path to reference CQL config YAML (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run mode: validate configs but do not train.",
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default="runs/baselines",
        help="Directory for baseline outputs (default: %(default)s).",
    )
    args = parser.parse_args(argv)

    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    rewards = tuple(r.strip() for r in args.rewards.split(",") if r.strip())
    learning_rates = tuple(
        float(x.strip()) for x in args.learning_rates.split(",") if x.strip()
    )
    cql_alphas = tuple(
        float(x.strip()) for x in args.cql_alphas.split(",") if x.strip()
    )

    # Validate reward variants
    for reward in rewards:
        if reward not in REWARD_VARIANT_MAP:
            valid = ", ".join(REWARD_VARIANT_MAP)
            logger.error(
                "Unknown reward variant '%s'. Valid: %s", reward, valid
            )
            sys.exit(1)

    logger.info("CQL Sweep Orchestrator v%s", SWEEP_SCRIPT_VERSION)
    logger.info("Stage: %d", args.stage)
    logger.info("Rewards: %s", rewards)
    logger.info("Learning rates: %s", learning_rates)
    logger.info("CQL alphas: %s", cql_alphas)
    logger.info("Dry-run: %s", args.dry_run)

    # --- Baselines (stage 1 only) ---
    if args.stage == 1:
        logger.info("--- Baselines ---")
        run_baselines(args.baseline_dir)

    # --- CQL Sweep ---
    logger.info("--- CQL Sweep ---")
    results, manifest = run_cql_sweep(
        seeds,
        rewards,
        learning_rates,
        cql_alphas,
        stage=args.stage,
        stage1_eval_path=args.stage1_eval,
        reference_config_path=args.config,
        dry_run=args.dry_run,
    )

    # --- Summary ---
    sweep_summary = manifest.summary
    logger.info("=" * 60)
    logger.info("Sweep complete: %d/%d runs succeeded",
                sweep_summary.get("completed", 0), sweep_summary.get("total", 0))
    if sweep_summary["failed"] > 0:
        logger.warning(
            "Failed seeds: %s",
            sweep_summary.get("seeds_failed", []),
        )
    logger.info("Total elapsed: %.1fs", manifest.total_elapsed_sec)
    logger.info("Manifest: runs/cql_sweep/sweep_manifest.json")

    sys.exit(0 if sweep_summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
