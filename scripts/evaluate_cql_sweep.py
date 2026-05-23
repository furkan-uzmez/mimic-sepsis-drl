#!/usr/bin/env python3
"""
Evaluate all CQL checkpoint policies with bootstrap CIs and safety diagnostics.

Reads the sweep manifest from Task 3, loads each CQL checkpoint, evaluates
on held-out test episodes, computes bootstrap confidence intervals for FQE
and WIS, and runs safety/support diagnostics (clinician agreement, action
support analysis).  Aggregates results across seeds for each reward variant
and includes baseline policy comparisons.

Usage
-----
    uv run python scripts/evaluate_cql_sweep.py
    uv run python scripts/evaluate_cql_sweep.py --manifest runs/cql_sweep/sweep_manifest.json

Output
------
    runs/cql_sweep/evaluation_summary.json — aggregated OPE metrics with CIs
    runs/cql_sweep/evaluation/per_run/    — per-run detailed results

Reference
---------
- Task 4, Plan 10-01: Phase 10 CQL Final Evaluation and Report
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVAL_SCRIPT_VERSION: str = "1.0.0"
SWEEP_MANIFEST_PATH: str = "runs/cql_sweep/sweep_manifest.json"
EVAL_OUTPUT_DIR: str = "runs/cql_sweep/evaluation"
EVAL_SUMMARY_PATH: str = "runs/cql_sweep/evaluation_summary.json"

# Reward variant name mapping (plan → code)
REWARD_VARIANT_MAP: dict[str, str] = {
    "shaped": "sofa_shaped",
    "sparse": "sparse",
}

ACTION_LABELS: tuple[str, ...] = (
    "no_vaso", "vaso_Q1", "vaso_Q2", "vaso_Q3", "vaso_Q4",
)
FLUID_LABELS: tuple[str, ...] = (
    "no_fluid", "fluid_Q1", "fluid_Q2", "fluid_Q3", "fluid_Q4",
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class PerRunEval:
    """Evaluation metrics for one (seed, reward) run."""

    seed: int
    reward_variant: str
    fqe_mean: float
    fqe_lower: float
    fqe_upper: float
    wis_mean: float
    wis_lower: float
    wis_upper: float
    ess: float
    n_episodes: int
    clinician_agreement_rate: float
    adjacent_agreement_rate: float
    low_support_action_rate: float
    mean_behavior_return: float
    status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AggregatedResult:
    """Aggregated metrics across seeds for one reward variant."""

    reward_variant: str
    n_seeds: int
    n_episodes: int
    fqe_mean: float
    fqe_sem: float
    wis_mean: float
    wis_sem: float
    ess_mean: float
    clinician_agreement_mean: float
    adjacent_agreement_mean: float
    low_support_mean: float
    per_seed: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationSummary:
    """Top-level evaluation summary for the entire sweep."""

    version: str = EVAL_SCRIPT_VERSION
    reward_variants: dict[str, AggregatedResult] = field(default_factory=dict)
    baselines: dict[str, dict[str, Any]] = field(default_factory=dict)
    n_total_runs: int = 0
    n_evaluated: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "reward_variants": {
                k: v.to_dict() for k, v in self.reward_variants.items()
            },
            "baselines": self.baselines,
            "n_total_runs": self.n_total_runs,
            "n_evaluated": self.n_evaluated,
        }


# ---------------------------------------------------------------------------
# Per-run evaluation
# ---------------------------------------------------------------------------


def load_cql_policy_from_checkpoint(
    checkpoint_path: str,
) -> Any | None:
    """Load a CQL policy from a checkpoint directory.

    Returns None if the checkpoint cannot be loaded.
    """
    try:
        from mimic_sepsis_rl.training.cql import load_cql_policy

        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            logger.warning("Checkpoint not found: %s", checkpoint_path)
            return None
        return load_cql_policy(ckpt)
    except ImportError as exc:
        logger.warning("Cannot load CQL policy: %s", exc)
        return None
    except Exception as exc:
        logger.warning("Error loading checkpoint %s: %s", checkpoint_path, exc)
        return None


def load_held_out_episodes(
    held_out_path: str | None = None,
) -> list[Any]:
    """Load held-out test episodes.

    This is a stub that returns synthetic episodes for offline evaluation.
    In production, episodes are loaded from the Phase 6 split manifests.
    """
    # In production, this would load real held-out episodes from
    # data/processed/ episodes filtered to the test split.
    logger.warning(
        "Held-out episode loading stub: real episodes require Phase 6 "
        "split manifests and transition data."
    )
    return []


def evaluate_single_run(
    seed: int,
    reward_variant: str,
    manifest_entry: dict[str, Any],
) -> PerRunEval:
    """Evaluate one CQL run from its sweep manifest entry.

    When real checkpoints and episodes are available, this calls the
    full OPE + bootstrap pipeline.  Currently returns stub values.
    """
    variant_code = REWARD_VARIANT_MAP.get(reward_variant, reward_variant)

    logger.info(
        "Evaluating seed=%d, reward=%s …", seed, variant_code
    )

    # In production, this would:
    # 1. Load checkpoint via load_cql_policy_from_checkpoint()
    # 2. Load held-out episodes via load_held_out_episodes()
    # 3. Compute FQE + WIS via evaluate_policy_run()
    # 4. Compute bootstrap CIs via bootstrap_fqe() / bootstrap_wis()
    # 5. Compute safety diagnostics via build_safety_review()

    success = manifest_entry.get("success", False)
    if not success:
        return PerRunEval(
            seed=seed,
            reward_variant=variant_code,
            fqe_mean=float("nan"),
            fqe_lower=float("nan"),
            fqe_upper=float("nan"),
            wis_mean=float("nan"),
            wis_lower=float("nan"),
            wis_upper=float("nan"),
            ess=float("nan"),
            n_episodes=0,
            clinician_agreement_rate=float("nan"),
            adjacent_agreement_rate=float("nan"),
            low_support_action_rate=float("nan"),
            mean_behavior_return=float("nan"),
            status="failed_training",
        )

    # Stub values — replace with real evaluation when data is available
    return PerRunEval(
        seed=seed,
        reward_variant=variant_code,
        fqe_mean=0.0,
        fqe_lower=0.0,
        fqe_upper=0.0,
        wis_mean=0.0,
        wis_lower=0.0,
        wis_upper=0.0,
        ess=0.0,
        n_episodes=0,
        clinician_agreement_rate=0.0,
        adjacent_agreement_rate=0.0,
        low_support_action_rate=0.0,
        mean_behavior_return=0.0,
        status="stub",
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_across_seeds(
    per_run_results: list[PerRunEval],
    reward_variant: str,
) -> AggregatedResult:
    """Aggregate per-seed results into mean ± SEM for one reward variant."""
    valid = [r for r in per_run_results if r.status == "ok"]

    result = AggregatedResult(
        reward_variant=reward_variant,
        n_seeds=len(valid),
        n_episodes=0,
        fqe_mean=0.0,
        fqe_sem=0.0,
        wis_mean=0.0,
        wis_sem=0.0,
        ess_mean=0.0,
        clinician_agreement_mean=0.0,
        adjacent_agreement_mean=0.0,
        low_support_mean=0.0,
    )

    if not valid:
        return result

    n = len(valid)

    def _mean_sem(values: list[float]) -> tuple[float, float]:
        m = statistics.mean(values)
        s = statistics.stdev(values) / (n**0.5) if n > 1 else 0.0
        return m, s

    result.fqe_mean, result.fqe_sem = _mean_sem([r.fqe_mean for r in valid])
    result.wis_mean, result.wis_sem = _mean_sem([r.wis_mean for r in valid])
    result.ess_mean = statistics.mean(r.ess for r in valid)
    result.clinician_agreement_mean = statistics.mean(
        r.clinician_agreement_rate for r in valid
    )
    result.adjacent_agreement_mean = statistics.mean(
        r.adjacent_agreement_rate for r in valid
    )
    result.low_support_mean = statistics.mean(
        r.low_support_action_rate for r in valid
    )
    result.n_episodes = valid[0].n_episodes if valid else 0
    result.per_seed = [r.to_dict() for r in valid]

    return result


def evaluate_baselines() -> dict[str, dict[str, Any]]:
    """Evaluate baseline policies (clinician, no-treatment, BC).

    Baseline policies are deterministic and evaluated once with the
    same OPE pipeline as the CQL policies.
    """
    baselines: dict[str, dict[str, Any]] = {}

    for name in ("clinician", "no_treatment", "bc"):
        baselines[name] = {
            "status": "available",
            "fqe": None,
            "wis": None,
            "ess": None,
            "n_episodes": 0,
            "mean_behavior_return": None,
            "note": "Phase 6 baseline — full evaluation requires held-out episode re-run.",
        }

    return baselines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Entry point for the CQL sweep evaluation script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        prog="evaluate_cql_sweep",
        description="Evaluate CQL checkpoints with bootstrap CIs and safety diagnostics.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=SWEEP_MANIFEST_PATH,
        help="Path to sweep manifest JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=EVAL_SUMMARY_PATH,
        help="Output path for evaluation summary JSON (default: %(default)s).",
    )
    args = parser.parse_args(argv)

    logger.info("CQL Sweep Evaluator v%s", EVAL_SCRIPT_VERSION)

    # Load sweep manifest (if it exists)
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.warning(
            "Sweep manifest not found at %s. Creating stub evaluation summary.",
            args.manifest,
        )
        summary = EvaluationSummary()
        summary.baselines = evaluate_baselines()

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary.to_dict(), indent=2))
        logger.info("Stub evaluation summary written to %s", args.output)

        print(json.dumps(summary.to_dict(), indent=2))
        sys.exit(0)

    # Load manifest
    manifest = json.loads(manifest_path.read_text())
    runs = manifest.get("runs", [])
    summary = EvaluationSummary(n_total_runs=len(runs))

    # Group runs by reward variant
    by_reward: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        reward = run.get("reward_variant", "unknown")
        by_reward.setdefault(reward, []).append(run)

    # Evaluate each run
    all_per_run: list[PerRunEval] = []
    for reward, run_entries in sorted(by_reward.items()):
        per_run: list[PerRunEval] = []
        for entry in run_entries:
            seed = entry.get("seed", 0)
            result = evaluate_single_run(seed, reward, entry)
            per_run.append(result)
            all_per_run.append(result)

        aggregated = aggregate_across_seeds(per_run, reward)
        summary.reward_variants[reward] = aggregated

    summary.n_evaluated = sum(
        1 for r in all_per_run if r.status == "ok"
    )
    summary.baselines = evaluate_baselines()

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary.to_dict(), indent=2))
    logger.info("Evaluation summary written to %s", args.output)

    # Print summary
    print(json.dumps(summary.to_dict(), indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
