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
    seed: int,
    reward_variant: str,
) -> dict[str, Any]:
    """Build a deep-copied config with seed, reward, and directory overrides."""
    import copy

    cfg = copy.deepcopy(base)

    # Resolve reward variant code name
    variant_code = REWARD_VARIANT_MAP.get(reward_variant, reward_variant)

    # --- Override runtime seed ---
    cfg.setdefault("runtime", {})
    cfg["runtime"]["seed"] = seed

    # --- Override checkpoint/log directories ---
    cfg.setdefault("checkpoint", {})
    cfg["checkpoint"]["checkpoint_dir"] = "checkpoints/cql_sweep"

    cfg.setdefault("logging", {})
    cfg["logging"]["log_dir"] = "runs/cql_sweep"
    cfg["logging"]["experiment_name"] = f"cql_s{seed}_{variant_code}"

    # --- Store reward variant as extra metadata ---
    cfg.setdefault("extra", {})
    cfg["extra"]["reward_variant"] = variant_code

    # --- Preserve dataset path from reference ---
    cfg.setdefault("dataset_path", "data/replay/replay_train.parquet")
    cfg.setdefault("dataset_meta_path", "data/replay/replay_train_meta.json")

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
    *,
    reference_config_path: str = REFERENCE_CQL_CONFIG,
    dry_run: bool = False,
) -> tuple[list[RunResult], SweepManifest]:
    """Run the full CQL sweep across all seeds and reward variants.

    For each (seed, reward) pair, a temporary config YAML is created
    and the CQL trainer is invoked as a subprocess.  Failures are logged
    and the sweep continues.
    """
    base_cfg = _load_reference_config(reference_config_path)
    results: list[RunResult] = []
    manifest = SweepManifest(
        seeds=seeds,
        reward_variants=reward_variants,
        start_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    total_runs = len(seeds) * len(reward_variants)
    completed = 0
    failed = 0
    sweep_start = time.monotonic()

    logger.info(
        "Starting CQL sweep: %d seeds × %d rewards = %d runs",
        len(seeds),
        len(reward_variants),
        total_runs,
    )

    for reward in reward_variants:
        for seed in seeds:
            label = f"seed={seed}, reward={reward}"
            run_start = time.monotonic()
            temp_config_path: Path | None = None

            try:
                sweep_cfg = _build_sweep_config(base_cfg, seed, reward)
                temp_config_path = _write_temp_config(sweep_cfg)

                logger.info(
                    "[%d/%d] Running CQL %s … config=%s",
                    completed + failed + 1,
                    total_runs,
                    label,
                    temp_config_path,
                )

                proc = run_cql_training(temp_config_path, dry_run=dry_run)
                elapsed = time.monotonic() - run_start

                if proc.returncode == 0:
                    logger.info(
                        "  ✓ %s completed in %.1fs", label, elapsed
                    )
                    results.append(
                        RunResult(
                            seed=seed,
                            reward_variant=reward,
                            success=True,
                            elapsed_sec=elapsed,
                            output_dir=str(
                                Path("runs/cql_sweep")
                                / f"cql_s{seed}_{REWARD_VARIANT_MAP.get(reward, reward)}"
                            ),
                        )
                    )
                    completed += 1
                else:
                    logger.error(
                        "  ✗ %s failed (rc=%d): %s",
                        label,
                        proc.returncode,
                        proc.stderr[:500] if proc.stderr else "(no stderr)",
                    )
                    results.append(
                        RunResult(
                            seed=seed,
                            reward_variant=reward,
                            success=False,
                            elapsed_sec=elapsed,
                            error_message=(
                                proc.stderr[:1000] if proc.stderr else "Unknown error"
                            ),
                        )
                    )
                    failed += 1

            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - run_start
                logger.error("  ✗ %s timed out after %.0fs", label, elapsed)
                results.append(
                    RunResult(
                        seed=seed,
                        reward_variant=reward,
                        success=False,
                        elapsed_sec=elapsed,
                        error_message="Timeout expired",
                    )
                )
                failed += 1

            except Exception as exc:
                elapsed = time.monotonic() - run_start
                logger.error("  ✗ %s failed: %s", label, exc)
                results.append(
                    RunResult(
                        seed=seed,
                        reward_variant=reward,
                        success=False,
                        elapsed_sec=elapsed,
                        error_message=str(exc),
                    )
                )
                failed += 1

            finally:
                # Clean up temporary config
                if temp_config_path is not None and temp_config_path.exists():
                    try:
                        temp_config_path.unlink()
                    except OSError:
                        pass

    manifest.end_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest.total_elapsed_sec = time.monotonic() - sweep_start
    manifest.runs = [result.to_dict() for result in results]
    manifest.summary = {
        "total_runs": total_runs,
        "completed": completed,
        "failed": failed,
        "seeds_completed": sorted(
            {r.seed for r in results if r.success}
        ),
        "seeds_failed": sorted({r.seed for r in results if not r.success}),
    }

    # Write manifest
    output_dir = Path("runs/cql_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "sweep_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {**asdict(manifest), "baselines": []},
            indent=2,
        )
    )
    logger.info("Sweep manifest written to %s", manifest_path)

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

    # Validate reward variants
    for reward in rewards:
        if reward not in REWARD_VARIANT_MAP:
            valid = ", ".join(REWARD_VARIANT_MAP)
            logger.error(
                "Unknown reward variant '%s'. Valid: %s", reward, valid
            )
            sys.exit(1)

    logger.info("CQL Sweep Orchestrator v%s", SWEEP_SCRIPT_VERSION)
    logger.info("Seeds: %s", seeds)
    logger.info("Rewards: %s", rewards)
    logger.info("Dry-run: %s", args.dry_run)

    # --- Baselines ---
    logger.info("--- Baselines ---")
    baseline_results = run_baselines(args.baseline_dir)

    # --- CQL Sweep ---
    logger.info("--- CQL Sweep ---")
    results, manifest = run_cql_sweep(
        seeds,
        rewards,
        reference_config_path=args.config,
        dry_run=args.dry_run,
    )

    # --- Summary ---
    sweep_summary = manifest.summary
    logger.info("=" * 60)
    logger.info("Sweep complete: %d/%d runs succeeded",
                sweep_summary["completed"], sweep_summary["total_runs"])
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
