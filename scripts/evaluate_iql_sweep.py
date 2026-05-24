#!/usr/bin/env python3
"""Evaluate IQL checkpoints and generate missing thesis/report metrics.

This script is the IQL counterpart of the CQL sweep evaluator. It loads one or
more IQL checkpoints, computes FQE-style actor-score means, WIS/ESS, behavior
support diagnostics, clinician-vs-policy heatmaps, subgroup summaries,
bootstrap intervals, seed-variance tables, and example trajectory reviews.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mimic_sepsis_rl.evaluation.bootstrap import bootstrap_wis
from mimic_sepsis_rl.evaluation.ope import HeldOutEpisode, HeldOutStep, compute_wis_and_ess
from mimic_sepsis_rl.evaluation.safety import FLUID_BIN_LABELS, VASO_BIN_LABELS
from mimic_sepsis_rl.mdp.actions.bins import ActionBinner, N_BINS
from mimic_sepsis_rl.training.iql import IQLPolicy, load_iql_policy

logger = logging.getLogger(__name__)

N_ACTIONS = 25
GAMMA = 0.99
DEFAULT_STATE_DIM = 62


@dataclass(frozen=True)
class CheckpointEval:
    checkpoint_path: str
    seed: int
    epoch: int
    fqe_mean: float
    wis_mean: float
    wis_lower: float
    wis_upper: float
    ess: float
    low_support_action_rate: float
    clinician_agreement: float
    high_risk_agreement: float
    high_risk_low_support_rate: float
    n_episodes: int
    n_steps: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_episodes(parquet_path: Path) -> list[HeldOutEpisode]:
    df = pl.read_parquet(parquet_path)
    state_cols = sorted(c for c in df.columns if c.startswith("s_") and not c.startswith("ns_"))
    action_counts = Counter(int(a) for a in df["action"].to_list())
    total_actions = sum(action_counts.values())
    behavior_probs = {a: c / max(total_actions, 1) for a, c in action_counts.items()}

    episodes: list[HeldOutEpisode] = []
    for stay_id in df["stay_id"].unique().sort().to_list():
        ep_df = df.filter(pl.col("stay_id") == stay_id).sort("step_index")
        steps: list[HeldOutStep] = []
        for row in ep_df.iter_rows(named=True):
            action = int(row["action"])
            steps.append(
                HeldOutStep(
                    episode_id=str(stay_id),
                    step_index=int(row["step_index"]),
                    state=tuple(float(row[c]) for c in state_cols),
                    action=action,
                    reward=float(row["reward"]),
                    done=bool(row["done"]),
                    behavior_action_prob=behavior_probs.get(action, 1e-6),
                )
            )
        episodes.append(HeldOutEpisode(episode_id=str(stay_id), steps=tuple(steps)))
    return episodes


def _state_dim_from_parquet(parquet_path: Path) -> int:
    schema = pl.scan_parquet(parquet_path).collect_schema()
    return sum(1 for c in schema.names() if c.startswith("s_") and not c.startswith("ns_"))


def _checkpoint_epoch(path: Path) -> int:
    match = re.search(r"epoch(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def _discover_checkpoints(checkpoint_dir: Path) -> list[Path]:
    return sorted(checkpoint_dir.rglob("iql_epoch*.pt"), key=lambda p: (_checkpoint_epoch(p), str(p)))


def _support_by_action(episodes: Sequence[HeldOutEpisode]) -> dict[int, tuple[float, int]]:
    counts: Counter[int] = Counter()
    for episode in episodes:
        counts.update(step.action for step in episode.steps)
    total = sum(counts.values())
    return {action: (count / max(total, 1), count) for action, count in counts.items()}


def _policy_actions(policy: IQLPolicy, episodes: Sequence[HeldOutEpisode]) -> list[int]:
    return [int(policy.select_action(step.state)) for episode in episodes for step in episode.steps]


def _clinician_actions(episodes: Sequence[HeldOutEpisode]) -> list[int]:
    return [int(step.action) for episode in episodes for step in episode.steps]


def _action_matrix(actions: Sequence[int]) -> np.ndarray:
    decoder = ActionBinner()
    matrix = np.zeros((N_BINS, N_BINS), dtype=float)
    for action in actions:
        if 0 <= int(action) < N_ACTIONS:
            vaso_bin, fluid_bin = decoder.decode_action(int(action))
            matrix[vaso_bin, fluid_bin] += 1.0
    return matrix


def _plot_heatmaps(clinician_actions: Sequence[int], policy_actions: Sequence[int], output_dir: Path) -> Path:
    clinician = _action_matrix(clinician_actions)
    policy = _action_matrix(policy_actions)
    clinician_norm = clinician / max(clinician.sum(), 1.0)
    policy_norm = policy / max(policy.sum(), 1.0)
    delta = policy_norm - clinician_norm

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, title, values, cmap in zip(
        axes,
        ("Clinician actions", "IQL policy actions", "Delta (IQL - clinician)"),
        (clinician_norm, policy_norm, delta),
        ("Blues", "Oranges", "RdBu_r"),
        strict=True,
    ):
        limit = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 1e-6)
        image = ax.imshow(values, cmap=cmap, vmin=-limit if "Delta" in title else 0.0, vmax=limit)
        ax.set_title(title)
        ax.set_xticks(range(N_BINS), FLUID_BIN_LABELS, rotation=30, ha="right")
        ax.set_yticks(range(N_BINS), VASO_BIN_LABELS)
        ax.set_xlabel("Fluid bin")
        ax.set_ylabel("Vasopressor bin")
        for row in range(N_BINS):
            for col in range(N_BINS):
                ax.text(col, row, f"{values[row, col]:.3f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=ax, shrink=0.75)
    fig.tight_layout()
    path = output_dir / "iql_action_heatmaps.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_fqe_low_support(results: Sequence[CheckpointEval], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [r.low_support_action_rate for r in results]
    ys = [r.fqe_mean for r in results]
    labels = [f"s{r.seed}/e{r.epoch}" for r in results]
    ax.scatter(xs, ys, s=80, color="#1f77b4", alpha=0.85)
    for x, y, label in zip(xs, ys, labels, strict=True):
        ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title("IQL FQE vs Low-Support Action Rate")
    ax.set_xlabel("Low-support action rate")
    ax.set_ylabel("FQE-style actor score")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "iql_fqe_vs_low_support.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_seed_variance(results: Sequence[CheckpointEval], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"seed {r.seed}\nepoch {r.epoch}" for r in results]
    values = [r.fqe_mean for r in results]
    ax.bar(labels, values, color="#2ca02c", alpha=0.85)
    ax.axhline(float(np.mean(values)) if values else 0.0, color="#444444", linestyle="--", label="mean")
    ax.set_title("IQL Seed/Checkpoint FQE Variance")
    ax.set_ylabel("FQE-style actor score")
    ax.tick_params(axis="x", labelrotation=30)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "iql_seed_variance.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_subgroup_safety(rows: Sequence[dict[str, Any]], output_dir: Path) -> Path:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["subgroup"])].append(row)

    labels = sorted(grouped)
    agreement = [np.mean([r["agreement"] for r in grouped[label]]) for label in labels]
    low_support = [np.mean([r["low_support"] for r in grouped[label]]) for label in labels]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, agreement, width, label="agreement", color="#1f77b4")
    ax.bar(x + width / 2, low_support, width, label="low support", color="#d62728")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_title("IQL Subgroup Safety")
    ax.set_ylabel("Fraction")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = output_dir / "iql_subgroup_safety.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_trajectory_review(rows: Sequence[dict[str, Any]], output_dir: Path, limit: int = 20) -> Path:
    ranked = sorted(rows, key=lambda r: (not r["low_support"], r["agreement"], -r["step_index"]))
    path = output_dir / "iql_example_trajectory_review.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "episode_id",
                "step_index",
                "subgroup",
                "clinician_action",
                "policy_action",
                "agreement",
                "support_prob",
                "support_count",
                "low_support",
            ],
        )
        writer.writeheader()
        writer.writerows(ranked[:limit])
    return path


def _fqe_actor_score(policy: IQLPolicy, episodes: Sequence[HeldOutEpisode]) -> float:
    values = []
    for episode in episodes:
        if not episode.steps:
            continue
        scores = policy.action_scores(episode.steps[0].state)
        values.append(max(scores))
    return float(np.mean(values)) if values else float("nan")


def _row_diagnostics(
    policy: IQLPolicy,
    episodes: Sequence[HeldOutEpisode],
    support: dict[int, tuple[float, int]],
    *,
    min_support_prob: float,
    min_support_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        for step in episode.steps:
            policy_action = int(policy.select_action(step.state))
            prob, count = support.get(policy_action, (0.0, 0))
            subgroup = "high_risk" if step.step_index >= 12 or step.reward < -5.0 else "standard_risk"
            rows.append(
                {
                    "episode_id": step.episode_id,
                    "step_index": step.step_index,
                    "subgroup": subgroup,
                    "clinician_action": step.action,
                    "policy_action": policy_action,
                    "agreement": step.action == policy_action,
                    "support_prob": prob,
                    "support_count": count,
                    "low_support": prob < min_support_prob or count < min_support_count,
                }
            )
    return rows


def _evaluate_checkpoint(
    checkpoint: Path,
    episodes: Sequence[HeldOutEpisode],
    *,
    state_dim: int,
    seed: int,
    output_dir: Path,
    min_support_prob: float,
    min_support_count: int,
) -> tuple[CheckpointEval, list[dict[str, Any]], list[int]]:
    policy = load_iql_policy(checkpoint, state_dim=state_dim, n_actions=N_ACTIONS, device="cpu")
    support = _support_by_action(episodes)
    rows = _row_diagnostics(
        policy,
        episodes,
        support,
        min_support_prob=min_support_prob,
        min_support_count=min_support_count,
    )
    policy_actions = [int(row["policy_action"]) for row in rows]

    fqe_mean = _fqe_actor_score(policy, episodes)
    try:
        metrics, _ = compute_wis_and_ess(episodes, policy, gamma=GAMMA, max_importance_ratio=50.0)
        wis_ci = bootstrap_wis(episodes, policy, gamma=GAMMA, n_resamples=500, ci=95, seed=seed)
        wis_mean = float(wis_ci.mean)
        wis_lower = float(wis_ci.lower)
        wis_upper = float(wis_ci.upper)
        ess = float(getattr(wis_ci, "ess", metrics.ess))
    except Exception as exc:
        logger.warning("WIS/ESS failed for %s: %s", checkpoint, exc)
        wis_mean = wis_lower = wis_upper = ess = float("nan")

    low_support_rate = float(np.mean([row["low_support"] for row in rows])) if rows else float("nan")
    agreement = float(np.mean([row["agreement"] for row in rows])) if rows else float("nan")
    high_risk = [row for row in rows if row["subgroup"] == "high_risk"]
    high_risk_agreement = float(np.mean([row["agreement"] for row in high_risk])) if high_risk else float("nan")
    high_risk_low_support = float(np.mean([row["low_support"] for row in high_risk])) if high_risk else float("nan")

    result = CheckpointEval(
        checkpoint_path=str(checkpoint),
        seed=seed,
        epoch=_checkpoint_epoch(checkpoint),
        fqe_mean=fqe_mean,
        wis_mean=wis_mean,
        wis_lower=wis_lower,
        wis_upper=wis_upper,
        ess=ess,
        low_support_action_rate=low_support_rate,
        clinician_agreement=agreement,
        high_risk_agreement=high_risk_agreement,
        high_risk_low_support_rate=high_risk_low_support,
        n_episodes=len(episodes),
        n_steps=len(rows),
    )
    return result, rows, policy_actions


def _write_summary_tables(results: Sequence[CheckpointEval], output_dir: Path) -> tuple[Path, Path]:
    table_path = output_dir / "iql_metrics_summary.csv"
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].to_dict().keys()))
        writer.writeheader()
        writer.writerows([r.to_dict() for r in results])

    seed_path = output_dir / "iql_seed_variance.csv"
    fqe_values = [r.fqe_mean for r in results if math.isfinite(r.fqe_mean)]
    with seed_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["n_checkpoints", len(results)])
        writer.writerow(["fqe_mean", float(np.mean(fqe_values)) if fqe_values else "nan"])
        writer.writerow(["fqe_std", float(np.std(fqe_values)) if len(fqe_values) > 1 else 0.0])
        writer.writerow(["fqe_min", float(np.min(fqe_values)) if fqe_values else "nan"])
        writer.writerow(["fqe_max", float(np.max(fqe_values)) if fqe_values else "nan"])
    return table_path, seed_path


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)
    parser = argparse.ArgumentParser(prog="evaluate_iql_sweep")
    parser.add_argument("--checkpoint-dir", default="checkpoints/iql")
    parser.add_argument("--test-data", default="data/replay/replay_test.parquet")
    parser.add_argument("--output-dir", default="runs/iql/iql_evaluation")
    parser.add_argument("--state-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-support-prob", type=float, default=0.01)
    parser.add_argument("--min-support-count", type=int, default=10)
    args = parser.parse_args(argv)

    checkpoint_dir = Path(args.checkpoint_dir)
    test_data = Path(args.test_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not test_data.exists():
        raise SystemExit(f"Test replay parquet not found: {test_data}")
    checkpoints = _discover_checkpoints(checkpoint_dir)
    if not checkpoints:
        raise SystemExit(f"No IQL checkpoints found under {checkpoint_dir}")

    episodes = _load_episodes(test_data)
    state_dim = args.state_dim or _state_dim_from_parquet(test_data) or DEFAULT_STATE_DIM
    results: list[CheckpointEval] = []
    all_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    best_policy_actions: list[int] = []
    clinician_actions = _clinician_actions(episodes)

    for idx, checkpoint in enumerate(checkpoints):
        result, rows, policy_actions = _evaluate_checkpoint(
            checkpoint,
            episodes,
            state_dim=state_dim,
            seed=args.seed + idx,
            output_dir=output_dir,
            min_support_prob=args.min_support_prob,
            min_support_count=args.min_support_count,
        )
        results.append(result)
        all_rows.extend(rows)
        if not best_rows or result.fqe_mean >= max(r.fqe_mean for r in results[:-1]):
            best_rows = rows
            best_policy_actions = policy_actions
        logger.info("%s fqe=%.4f wis=%.4f ess=%.2f low_support=%.3f", checkpoint.name, result.fqe_mean, result.wis_mean, result.ess, result.low_support_action_rate)

    table_path, seed_table_path = _write_summary_tables(results, output_dir)
    heatmap_path = _plot_heatmaps(clinician_actions, best_policy_actions, output_dir)
    scatter_path = _plot_fqe_low_support(results, output_dir)
    seed_plot_path = _plot_seed_variance(results, output_dir)
    subgroup_path = _plot_subgroup_safety(best_rows, output_dir)
    trajectory_path = _write_trajectory_review(best_rows, output_dir)

    payload = {
        "algorithm": "iql",
        "n_checkpoints": len(results),
        "n_episodes": len(episodes),
        "state_dim": state_dim,
        "results": [r.to_dict() for r in results],
        "artifacts": {
            "metrics_summary_table": str(table_path),
            "seed_variance_table": str(seed_table_path),
            "action_heatmaps": str(heatmap_path),
            "fqe_vs_low_support_scatter": str(scatter_path),
            "seed_variance_plot": str(seed_plot_path),
            "subgroup_safety_plot": str(subgroup_path),
            "example_trajectory_review": str(trajectory_path),
        },
    }
    summary_path = output_dir / "iql_evaluation_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"status": "ok", "summary": str(summary_path), "artifacts": payload["artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
