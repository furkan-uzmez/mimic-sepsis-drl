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
class MockIQLPolicy:
    """Deterministic mock policy used only for plot smoke tests."""

    n_actions: int = N_ACTIONS

    def select_action(self, state: Sequence[float]) -> int:
        signal = float(state[0]) + 0.5 * float(state[1]) - 0.25 * float(state[2])
        return int(abs(round(signal * 7.0))) % self.n_actions

    def action_scores(self, state: Sequence[float]) -> list[float]:
        center = self.select_action(state)
        return [float(2.0 - abs(action - center) * 0.15) for action in range(self.n_actions)]


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


def _make_mock_episodes(
    *,
    n_episodes: int,
    n_steps: int,
    state_dim: int,
    seed: int,
) -> list[HeldOutEpisode]:
    rng = np.random.default_rng(seed)
    actions = rng.choice(np.arange(N_ACTIONS), size=n_episodes * n_steps, replace=True)
    action_counts = Counter(int(action) for action in actions)
    total_actions = sum(action_counts.values())
    behavior_probs = {a: c / max(total_actions, 1) for a, c in action_counts.items()}

    episodes: list[HeldOutEpisode] = []
    action_idx = 0
    for episode_idx in range(n_episodes):
        severity = rng.normal(loc=0.0, scale=1.0)
        steps: list[HeldOutStep] = []
        for step_idx in range(n_steps):
            drift = step_idx / max(n_steps - 1, 1)
            state = rng.normal(loc=severity + drift, scale=0.7, size=state_dim)
            action = int(actions[action_idx])
            action_idx += 1
            reward = float(1.5 - severity - 0.15 * step_idx + rng.normal(0.0, 0.5))
            if step_idx == n_steps - 1:
                reward += 8.0 if severity < 0.3 else -8.0
            steps.append(
                HeldOutStep(
                    episode_id=f"mock-{episode_idx:03d}",
                    step_index=step_idx,
                    state=tuple(float(value) for value in state),
                    action=action,
                    reward=reward,
                    done=step_idx == n_steps - 1,
                    behavior_action_prob=behavior_probs.get(action, 1e-6),
                )
            )
        episodes.append(HeldOutEpisode(episode_id=f"mock-{episode_idx:03d}", steps=tuple(steps)))
    return episodes


def _mock_checkpoint_results(
    episodes: Sequence[HeldOutEpisode],
    *,
    seed: int,
    n_mock_checkpoints: int,
    min_support_prob: float,
    min_support_count: int,
) -> tuple[list[CheckpointEval], list[dict[str, Any]], list[int]]:
    results: list[CheckpointEval] = []
    best_rows: list[dict[str, Any]] = []
    best_policy_actions: list[int] = []
    best_fqe = float("-inf")

    for idx in range(n_mock_checkpoints):
        policy = MockIQLPolicy()
        support = _support_by_action(episodes)
        rows = _row_diagnostics(
            policy,
            episodes,
            support,
            min_support_prob=min_support_prob,
            min_support_count=min_support_count,
        )
        policy_actions = [int(row["policy_action"]) for row in rows]
        fqe_mean = _fqe_actor_score(policy, episodes) + idx * 0.12
        low_support_rate = float(np.mean([row["low_support"] for row in rows])) if rows else float("nan")
        agreement = float(np.mean([row["agreement"] for row in rows])) if rows else float("nan")
        high_risk = [row for row in rows if row["subgroup"] == "high_risk"]
        high_risk_agreement = float(np.mean([row["agreement"] for row in high_risk])) if high_risk else float("nan")
        high_risk_low_support = float(np.mean([row["low_support"] for row in high_risk])) if high_risk else float("nan")
        result = CheckpointEval(
            checkpoint_path=f"mock://iql_seed{seed + idx}_epoch{160 + idx * 20}",
            seed=seed + idx,
            epoch=160 + idx * 20,
            fqe_mean=fqe_mean,
            wis_mean=float(np.mean([step.reward for ep in episodes for step in ep.steps])) + idx * 0.05,
            wis_lower=fqe_mean - 0.4,
            wis_upper=fqe_mean + 0.4,
            ess=max(1.0, len(episodes) * (0.65 - 0.05 * idx)),
            low_support_action_rate=low_support_rate,
            clinician_agreement=agreement,
            high_risk_agreement=high_risk_agreement,
            high_risk_low_support_rate=high_risk_low_support,
            n_episodes=len(episodes),
            n_steps=len(rows),
        )
        results.append(result)
        if fqe_mean > best_fqe:
            best_fqe = fqe_mean
            best_rows = rows
            best_policy_actions = policy_actions
    return results, best_rows, best_policy_actions


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


def _policy_actions(policy: Any, episodes: Sequence[HeldOutEpisode]) -> list[int]:
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


def _plot_bootstrap_ci(results: Sequence[CheckpointEval], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"s{r.seed}/e{r.epoch}" for r in results]
    y = [r.wis_mean for r in results]
    lower = [max(0.0, r.wis_mean - r.wis_lower) for r in results]
    upper = [max(0.0, r.wis_upper - r.wis_mean) for r in results]
    ax.errorbar(labels, y, yerr=[lower, upper], fmt="o", capsize=4, color="#1f77b4")
    ax.set_title("IQL WIS Bootstrap Confidence Intervals")
    ax.set_ylabel("WIS")
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = output_dir / "iql_bootstrap_ci.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_advantage_weight_histogram(rows: Sequence[dict[str, Any]], output_dir: Path) -> Path:
    support_probs = np.array([float(row["support_prob"]) for row in rows], dtype=float)
    weights = 1.0 / np.clip(support_probs, 1e-3, 1.0)
    clipped = np.clip(weights, 0.0, 100.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(clipped, bins=25, color="#ff7f0e", alpha=0.85)
    ax.axvline(100.0, color="#d62728", linestyle="--", label="clip cap")
    ax.set_title("Mock IQL Advantage-Weight Clipping Histogram")
    ax.set_xlabel("Clipped proxy advantage weight")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = output_dir / "iql_advantage_weight_histogram.png"
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


def _plot_episode_return_distribution(episodes: Sequence[HeldOutEpisode], output_dir: Path) -> Path:
    returns = [sum((GAMMA**idx) * step.reward for idx, step in enumerate(ep.steps)) for ep in episodes]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].hist(returns, bins=min(25, max(5, len(returns) // 2)), color="#4c78a8", alpha=0.85)
    axes[0].axvline(float(np.mean(returns)) if returns else 0.0, color="#f58518", linestyle="--", label="mean")
    axes[0].set_title("IQL Evaluation Episode Returns")
    axes[0].set_xlabel("Discounted clinician return")
    axes[0].set_ylabel("Episodes")
    axes[0].legend()

    axes[1].plot(np.sort(returns), color="#54a24b", linewidth=2)
    axes[1].set_title("Sorted Episode Return Profile")
    axes[1].set_xlabel("Episode rank")
    axes[1].set_ylabel("Discounted return")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "iql_episode_return_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_support_action_frequency(rows: Sequence[dict[str, Any]], output_dir: Path) -> Path:
    support = np.zeros(N_ACTIONS, dtype=float)
    support_count = np.zeros(N_ACTIONS, dtype=float)
    policy_count = np.zeros(N_ACTIONS, dtype=float)
    for row in rows:
        action = int(row["policy_action"])
        if 0 <= action < N_ACTIONS:
            support[action] = max(support[action], float(row["support_prob"]))
            support_count[action] = max(support_count[action], float(row["support_count"]))
            policy_count[action] += 1.0
    policy_freq = policy_count / max(policy_count.sum(), 1.0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    actions = np.arange(N_ACTIONS)
    colors = ["#d62728" if count <= 0 else "#1f77b4" for count in support_count]
    axes[0].bar(actions, support, color=colors, alpha=0.85)
    axes[0].set_title("Behavior Support Mass for IQL-Selected Actions")
    axes[0].set_ylabel("Support probability")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(actions, policy_freq, color="#ff7f0e", alpha=0.85)
    axes[1].set_title("IQL Policy Action Frequency")
    axes[1].set_xlabel("Action ID")
    axes[1].set_ylabel("Policy frequency")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = output_dir / "iql_support_action_frequency.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_metrics_correlation(results: Sequence[CheckpointEval], output_dir: Path) -> Path:
    metric_names = ["fqe_mean", "wis_mean", "ess", "low_support_action_rate", "clinician_agreement"]
    data = np.array([[float(getattr(result, name)) for name in metric_names] for result in results], dtype=float)
    corr = np.eye(len(metric_names), dtype=float)
    if len(results) >= 2:
        for row in range(len(metric_names)):
            for col in range(len(metric_names)):
                left = data[:, row]
                right = data[:, col]
                if np.std(left) > 0.0 and np.std(right) > 0.0:
                    corr[row, col] = float(np.corrcoef(left, right)[0, 1])

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    labels = [name.replace("_", "\n") for name in metric_names]
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("IQL Metric Correlation Matrix")
    for row in range(len(metric_names)):
        for col in range(len(metric_names)):
            ax.text(col, row, f"{corr[row, col]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = output_dir / "iql_metric_correlation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_example_trajectory(rows: Sequence[dict[str, Any]], output_dir: Path) -> Path:
    if not rows:
        selected: list[dict[str, Any]] = []
    else:
        episode_id = sorted(rows, key=lambda r: (not r["low_support"], r["agreement"], -r["step_index"]))[0]["episode_id"]
        selected = sorted((r for r in rows if r["episode_id"] == episode_id), key=lambda r: int(r["step_index"]))

    steps = [int(row["step_index"]) for row in selected]
    clinician = [int(row["clinician_action"]) for row in selected]
    policy = [int(row["policy_action"]) for row in selected]
    low_support = [bool(row["low_support"]) for row in selected]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.step(steps, clinician, where="mid", label="clinician", color="#1f77b4", linewidth=2)
    ax.step(steps, policy, where="mid", label="IQL policy", color="#ff7f0e", linewidth=2)
    if steps:
        ax.scatter([s for s, low in zip(steps, low_support, strict=True) if low], [p for p, low in zip(policy, low_support, strict=True) if low], color="#d62728", s=60, label="low support")
    ax.set_title("Example Trajectory: Clinician vs IQL Actions")
    ax.set_xlabel("Step index")
    ax.set_ylabel("Discrete action ID")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "iql_example_trajectory_plot.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _normalise_metric(values: Sequence[float], *, higher_is_better: bool = True) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return arr
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    lo = float(np.min(arr[finite]))
    hi = float(np.max(arr[finite]))
    if math.isclose(lo, hi):
        scaled = np.full_like(arr, 0.5)
    else:
        scaled = (arr - lo) / (hi - lo)
    scaled = np.nan_to_num(scaled, nan=0.0)
    return scaled if higher_is_better else 1.0 - scaled


def _plot_iql_training_diagnostics(output_dir: Path, *, seed: int, n_epochs: int = 200) -> Path:
    rng = np.random.default_rng(seed)
    epochs = np.arange(1, n_epochs + 1)

    critic_loss = 12.0 * np.exp(-epochs / 55.0) + 7.0 + rng.normal(0.0, 0.7, n_epochs)
    value_loss = 0.9 * np.exp(-epochs / 70.0) + 0.25 + rng.normal(0.0, 0.05, n_epochs)
    actor_loss = 2.0 + 5.5 * (1.0 - np.exp(-epochs / 65.0)) + rng.normal(0.0, 0.35, n_epochs)
    mean_q = 1.5 + 7.5 * (1.0 - np.exp(-epochs / 75.0)) + rng.normal(0.0, 0.25, n_epochs)
    mean_v = mean_q + 0.2 + rng.normal(0.0, 0.18, n_epochs)
    advantage = mean_q - mean_v
    adv_weight_mean = 1.8 + 4.0 * (1.0 - np.exp(-epochs / 80.0)) + rng.normal(0.0, 0.25, n_epochs)
    adv_clip_fraction = np.clip(0.02 + 0.18 * (1.0 - np.exp(-epochs / 90.0)) + rng.normal(0.0, 0.015, n_epochs), 0.0, 1.0)

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    panels = [
        (axes[0, 0], critic_loss, "Critic/Q Loss", "Loss", "#1f77b4"),
        (axes[0, 1], value_loss, "Value/Expectile Loss", "Loss", "#ff7f0e"),
        (axes[1, 0], actor_loss, "Actor Loss", "Loss", "#2ca02c"),
        (axes[2, 0], adv_weight_mean, "Advantage Weight Mean", "Weight", "#8c564b"),
        (axes[2, 1], adv_clip_fraction, "Advantage Clip Fraction", "Fraction", "#d62728"),
    ]
    for ax, values, title, ylabel, color in panels:
        ax.plot(epochs, values, color=color, alpha=0.35, linewidth=0.8)
        window = min(10, len(values))
        smooth = np.convolve(values, np.ones(window) / window, mode="valid")
        ax.plot(epochs[window - 1 :], smooth, color=color, linewidth=2)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    ax.plot(epochs, mean_q, color="#9467bd", alpha=0.75, linewidth=1.4, label="Mean Q")
    ax.plot(epochs, mean_v, color="#17becf", alpha=0.75, linewidth=1.4, label="Mean V")
    ax.plot(epochs, advantage, color="#7f7f7f", alpha=0.75, linewidth=1.2, label="Q - V advantage")
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=0.8)
    ax.set_title("Q/V/Advantage Diagnostics")
    ax.set_ylabel("Value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    for ax in axes[-1, :]:
        ax.set_xlabel("Epoch")
    fig.suptitle("IQL Training Diagnostics (mock)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = output_dir / "iql_training_diagnostics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_pareto_frontier(results: Sequence[CheckpointEval], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    xs = np.array([r.low_support_action_rate for r in results], dtype=float)
    ys = np.array([r.fqe_mean for r in results], dtype=float)
    ess = np.array([r.ess for r in results], dtype=float)
    agreement = np.array([r.clinician_agreement for r in results], dtype=float)
    sizes = 120.0 + 360.0 * _normalise_metric(agreement.tolist())

    scatter = ax.scatter(xs, ys, c=ess, s=sizes, cmap="viridis", alpha=0.85, edgecolor="#222222")
    for result, x, y in zip(results, xs, ys, strict=True):
        ax.annotate(f"s{result.seed}/e{result.epoch}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_title("IQL Pareto Frontier: Value vs Support Risk")
    ax.set_xlabel("Low-support action rate (lower is safer)")
    ax.set_ylabel("FQE-style actor score (higher is better)")
    ax.grid(True, alpha=0.3)
    fig.colorbar(scatter, ax=ax, shrink=0.82, label="ESS")
    fig.tight_layout()
    path = output_dir / "iql_pareto_frontier.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_ope_metric_ranking(results: Sequence[CheckpointEval], output_dir: Path) -> Path:
    labels = [f"s{r.seed}/e{r.epoch}" for r in results]
    metrics = {
        "FQE": _normalise_metric([r.fqe_mean for r in results]),
        "WIS": _normalise_metric([r.wis_mean for r in results]),
        "ESS": _normalise_metric([r.ess for r in results]),
        "Support": _normalise_metric([r.low_support_action_rate for r in results], higher_is_better=False),
        "Agreement": _normalise_metric([r.clinician_agreement for r in results]),
    }
    overall = np.mean(np.vstack(list(metrics.values())), axis=0) if results else np.array([])
    order = np.argsort(-overall) if len(overall) else np.array([], dtype=int)
    data = np.vstack([values[order] for values in metrics.values()]) if len(order) else np.zeros((len(metrics), 0))
    ordered_labels = [labels[idx] for idx in order]

    fig, ax = plt.subplots(figsize=(max(7, len(ordered_labels) * 1.2), 5.5))
    image = ax.imshow(data, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(ordered_labels)), ordered_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(metrics)), list(metrics.keys()))
    ax.set_title("IQL OPE Metric Ranking (Normalized)")
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            ax.text(col, row, f"{data[row, col]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.82, label="normalized score")
    fig.tight_layout()
    path = output_dir / "iql_ope_metric_ranking.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_action_deviation_severity(rows: Sequence[dict[str, Any]], output_dir: Path) -> Path:
    decoder = ActionBinner()
    vaso_deltas: list[int] = []
    fluid_deltas: list[int] = []
    severity_counts = Counter({"exact": 0, "adjacent": 0, "moderate": 0, "large": 0})
    for row in rows:
        clinician_vaso, clinician_fluid = decoder.decode_action(int(row["clinician_action"]))
        policy_vaso, policy_fluid = decoder.decode_action(int(row["policy_action"]))
        vaso_delta = int(policy_vaso - clinician_vaso)
        fluid_delta = int(policy_fluid - clinician_fluid)
        vaso_deltas.append(vaso_delta)
        fluid_deltas.append(fluid_delta)
        max_delta = max(abs(vaso_delta), abs(fluid_delta))
        if max_delta == 0:
            severity_counts["exact"] += 1
        elif max_delta == 1:
            severity_counts["adjacent"] += 1
        elif max_delta == 2:
            severity_counts["moderate"] += 1
        else:
            severity_counts["large"] += 1

    labels = ["exact", "adjacent", "moderate", "large"]
    total = max(sum(severity_counts.values()), 1)
    fractions = [severity_counts[label] / total for label in labels]
    delta_grid = np.zeros((2 * N_BINS - 1, 2 * N_BINS - 1), dtype=float)
    offset = N_BINS - 1
    for vaso_delta, fluid_delta in zip(vaso_deltas, fluid_deltas, strict=True):
        delta_grid[vaso_delta + offset, fluid_delta + offset] += 1.0
    delta_grid /= max(delta_grid.sum(), 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(labels, fractions, color=["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"], alpha=0.85)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Action Deviation Severity")
    axes[0].set_ylabel("Fraction of decisions")
    axes[0].grid(True, axis="y", alpha=0.3)

    image = axes[1].imshow(delta_grid, cmap="magma", origin="lower")
    tick_values = list(range(-offset, offset + 1))
    axes[1].set_xticks(range(len(tick_values)), tick_values)
    axes[1].set_yticks(range(len(tick_values)), tick_values)
    axes[1].set_title("Policy - Clinician Bin Delta")
    axes[1].set_xlabel("Fluid bin delta")
    axes[1].set_ylabel("Vasopressor bin delta")
    fig.colorbar(image, ax=axes[1], shrink=0.82, label="fraction")
    fig.tight_layout()
    path = output_dir / "iql_action_deviation_severity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_high_risk_action_heatmaps(rows: Sequence[dict[str, Any]], output_dir: Path) -> Path:
    high_risk_rows = [row for row in rows if row["subgroup"] == "high_risk"] or list(rows)
    clinician = _action_matrix([int(row["clinician_action"]) for row in high_risk_rows])
    policy = _action_matrix([int(row["policy_action"]) for row in high_risk_rows])
    clinician_norm = clinician / max(clinician.sum(), 1.0)
    policy_norm = policy / max(policy.sum(), 1.0)
    delta = policy_norm - clinician_norm

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, title, values, cmap in zip(
        axes,
        ("High-risk clinician", "High-risk IQL", "High-risk delta"),
        (clinician_norm, policy_norm, delta),
        ("Blues", "Oranges", "RdBu_r"),
        strict=True,
    ):
        limit = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 1e-6)
        image = ax.imshow(values, cmap=cmap, vmin=-limit if "delta" in title else 0.0, vmax=limit)
        ax.set_title(title)
        ax.set_xticks(range(N_BINS), FLUID_BIN_LABELS, rotation=30, ha="right")
        ax.set_yticks(range(N_BINS), VASO_BIN_LABELS)
        ax.set_xlabel("Fluid bin")
        ax.set_ylabel("Vasopressor bin")
        fig.colorbar(image, ax=ax, shrink=0.75)
    fig.tight_layout()
    path = output_dir / "iql_high_risk_action_heatmaps.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_reward_decomposition(episodes: Sequence[HeldOutEpisode], output_dir: Path) -> Path:
    episode = max(episodes, key=lambda ep: len(ep.steps)) if episodes else HeldOutEpisode(episode_id="empty", steps=tuple())
    hours = np.arange(len(episode.steps)) * 4
    rewards = np.array([step.reward for step in episode.steps], dtype=float)
    sofa_proxy = np.array([12.0 - float(step.state[0]) if step.state else 0.0 for step in episode.steps], dtype=float)
    sofa_proxy = np.clip(sofa_proxy, 0.0, 24.0)
    intermediate = rewards.copy()
    terminal = np.zeros_like(rewards)
    if len(rewards):
        terminal[-1] = rewards[-1]
        intermediate[-1] = 0.0

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(hours, sofa_proxy, "o-", color="#1f77b4", linewidth=2)
    axes[0].fill_between(hours, 0, sofa_proxy, color="#1f77b4", alpha=0.12)
    axes[0].set_title("Example Clinical Time Series (SOFA Proxy)")
    axes[0].set_ylabel("SOFA proxy")

    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in intermediate]
    axes[1].bar(hours, intermediate, width=3.0, color=colors, alpha=0.8)
    axes[1].axhline(0.0, color="#555555", linewidth=0.8)
    axes[1].set_title("Intermediate Reward Components")
    axes[1].set_ylabel("Reward")

    axes[2].plot(hours, np.cumsum(intermediate), color="#ff7f0e", linewidth=2, label="cumulative intermediate")
    if len(hours):
        axes[2].bar(hours[-1], terminal[-1], width=3.0, color="#9467bd", alpha=0.75, label="terminal")
    axes[2].axhline(0.0, color="#555555", linewidth=0.8)
    axes[2].set_title("Cumulative + Terminal Reward")
    axes[2].set_xlabel("Hours from episode start")
    axes[2].set_ylabel("Reward")
    axes[2].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "iql_reward_decomposition.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_cumulative_episode_rewards(episodes: Sequence[HeldOutEpisode], output_dir: Path) -> Path:
    returns = np.array([sum((GAMMA**idx) * step.reward for idx, step in enumerate(ep.steps)) for ep in episodes], dtype=float)
    order = np.arange(1, len(returns) + 1)
    cumulative = np.cumsum(returns)
    rolling_window = min(10, max(1, len(returns)))
    rolling = np.convolve(returns, np.ones(rolling_window) / rolling_window, mode="valid") if len(returns) else np.array([])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(order, cumulative, color="#1f77b4", linewidth=2)
    axes[0].set_title("Cumulative Episode Rewards")
    axes[0].set_xlabel("Episode index")
    axes[0].set_ylabel("Cumulative discounted return")
    axes[1].plot(order, returns, color="#bab0ac", alpha=0.55, label="episode return")
    if len(rolling):
        axes[1].plot(order[rolling_window - 1 :], rolling, color="#d62728", linewidth=2, label=f"rolling mean ({rolling_window})")
    axes[1].set_title("Episode Return Trend")
    axes[1].set_xlabel("Episode index")
    axes[1].set_ylabel("Discounted return")
    axes[1].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "iql_cumulative_episode_rewards.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_clinician_agreement(rows: Sequence[dict[str, Any]], output_dir: Path) -> Path:
    decoder = ActionBinner()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["subgroup"])].append(row)

    labels = sorted(grouped)
    exact_values: list[float] = []
    adjacent_values: list[float] = []
    for label in labels:
        exact = []
        adjacent = []
        for row in grouped[label]:
            clinician_vaso, clinician_fluid = decoder.decode_action(int(row["clinician_action"]))
            policy_vaso, policy_fluid = decoder.decode_action(int(row["policy_action"]))
            exact.append(int(row["clinician_action"]) == int(row["policy_action"]))
            adjacent.append(abs(clinician_vaso - policy_vaso) <= 1 and abs(clinician_fluid - policy_fluid) <= 1)
        exact_values.append(float(np.mean(exact)) if exact else 0.0)
        adjacent_values.append(float(np.mean(adjacent)) if adjacent else 0.0)

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, exact_values, width, label="exact match", color="#1f77b4", alpha=0.88)
    ax.bar(x + width / 2, adjacent_values, width, label="adjacent-bin match", color="#2ca02c", alpha=0.88)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("IQL Policy - Clinician Agreement")
    ax.set_ylabel("Agreement rate")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = output_dir / "iql_clinician_agreement.png"
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


def _fqe_actor_score(policy: Any, episodes: Sequence[HeldOutEpisode]) -> float:
    values = []
    for episode in episodes:
        if not episode.steps:
            continue
        scores = policy.action_scores(episode.steps[0].state)
        values.append(max(scores))
    return float(np.mean(values)) if values else float("nan")


def _row_diagnostics(
    policy: Any,
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
    parser.add_argument("--mock", action="store_true", help="Generate every artifact from deterministic mock data.")
    parser.add_argument("--mock-episodes", type=int, default=32)
    parser.add_argument("--mock-steps", type=int, default=18)
    parser.add_argument("--mock-checkpoints", type=int, default=4)
    args = parser.parse_args(argv)

    checkpoint_dir = Path(args.checkpoint_dir)
    test_data = Path(args.test_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dim = args.state_dim or DEFAULT_STATE_DIM
    all_rows: list[dict[str, Any]] = []

    if args.mock:
        episodes = _make_mock_episodes(
            n_episodes=args.mock_episodes,
            n_steps=args.mock_steps,
            state_dim=state_dim,
            seed=args.seed,
        )
        results, best_rows, best_policy_actions = _mock_checkpoint_results(
            episodes,
            seed=args.seed,
            n_mock_checkpoints=args.mock_checkpoints,
            min_support_prob=args.min_support_prob,
            min_support_count=args.min_support_count,
        )
        all_rows.extend(best_rows)
    else:
        if not test_data.exists():
            raise SystemExit(f"Test replay parquet not found: {test_data}")
        checkpoints = _discover_checkpoints(checkpoint_dir)
        if not checkpoints:
            raise SystemExit(f"No IQL checkpoints found under {checkpoint_dir}")

        episodes = _load_episodes(test_data)
        state_dim = args.state_dim or _state_dim_from_parquet(test_data) or DEFAULT_STATE_DIM
        results = []
        best_rows = []
        best_policy_actions = []

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

    clinician_actions = _clinician_actions(episodes)

    table_path, seed_table_path = _write_summary_tables(results, output_dir)
    heatmap_path = _plot_heatmaps(clinician_actions, best_policy_actions, output_dir)
    scatter_path = _plot_fqe_low_support(results, output_dir)
    seed_plot_path = _plot_seed_variance(results, output_dir)
    bootstrap_ci_path = _plot_bootstrap_ci(results, output_dir)
    subgroup_path = _plot_subgroup_safety(best_rows, output_dir)
    advantage_hist_path = _plot_advantage_weight_histogram(best_rows, output_dir)
    episode_return_path = _plot_episode_return_distribution(episodes, output_dir)
    support_frequency_path = _plot_support_action_frequency(best_rows, output_dir)
    metric_correlation_path = _plot_metrics_correlation(results, output_dir)
    trajectory_plot_path = _plot_example_trajectory(best_rows, output_dir)
    reward_decomposition_path = _plot_reward_decomposition(episodes, output_dir)
    cumulative_rewards_path = _plot_cumulative_episode_rewards(episodes, output_dir)
    clinician_agreement_path = _plot_clinician_agreement(best_rows, output_dir)
    pareto_path = _plot_pareto_frontier(results, output_dir)
    ope_ranking_path = _plot_ope_metric_ranking(results, output_dir)
    action_deviation_path = _plot_action_deviation_severity(best_rows, output_dir)
    high_risk_heatmap_path = _plot_high_risk_action_heatmaps(best_rows, output_dir)
    training_diagnostics_path = _plot_iql_training_diagnostics(output_dir, seed=args.seed)
    trajectory_path = _write_trajectory_review(best_rows, output_dir)

    payload = {
        "algorithm": "iql",
        "mock": bool(args.mock),
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
            "bootstrap_ci_plot": str(bootstrap_ci_path),
            "subgroup_safety_plot": str(subgroup_path),
            "advantage_weight_histogram": str(advantage_hist_path),
            "episode_return_distribution": str(episode_return_path),
            "support_action_frequency": str(support_frequency_path),
            "metric_correlation_matrix": str(metric_correlation_path),
            "example_trajectory_plot": str(trajectory_plot_path),
            "reward_decomposition_plot": str(reward_decomposition_path),
            "cumulative_episode_rewards": str(cumulative_rewards_path),
            "clinician_agreement_plot": str(clinician_agreement_path),
            "pareto_frontier_plot": str(pareto_path),
            "ope_metric_ranking_plot": str(ope_ranking_path),
            "action_deviation_severity_plot": str(action_deviation_path),
            "high_risk_action_heatmaps": str(high_risk_heatmap_path),
            "training_diagnostics_plot": str(training_diagnostics_path),
            "example_trajectory_review": str(trajectory_path),
        },
    }
    summary_path = output_dir / "iql_evaluation_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"status": "ok", "summary": str(summary_path), "artifacts": payload["artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
