#!/usr/bin/env python3
"""Compute reference-policy OPE metrics for the final IQL report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_iql_sweep import GAMMA, _load_episodes, _row_diagnostics, _support_by_action
from mimic_sepsis_rl.evaluation.ope import compute_wis_and_ess


class BehaviorReplayPolicy:
    def __init__(self, episodes):
        self.action_by_state = {}
        self.prob_by_state_action = {}
        for episode in episodes:
            for step in episode.steps:
                key = tuple(round(float(x), 8) for x in step.state)
                self.action_by_state[key] = step.action
                self.prob_by_state_action[(key, step.action)] = step.behavior_action_prob

    def select_action(self, state):
        return self.action_by_state[tuple(round(float(x), 8) for x in state)]

    def action_probability(self, state, action):
        key = tuple(round(float(x), 8) for x in state)
        return self.prob_by_state_action.get((key, int(action)), 0.0)


class NoTreatmentPolicy:
    def select_action(self, state):
        return 0

    def action_probability(self, state, action):
        return 1.0 if int(action) == 0 else 0.0


class StandardizedBCPolicy:
    def __init__(self, weights, biases, mean, std):
        self.weights = weights
        self.biases = biases
        self.mean = mean
        self.std = std

    def _probs(self, state):
        x = (np.asarray(state, dtype=np.float64) - self.mean) / self.std
        logits = x @ self.weights + self.biases
        logits -= logits.max()
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum()

    def select_action(self, state):
        return int(np.argmax(self._probs(state)))

    def action_probability(self, state, action):
        return float(self._probs(state)[int(action)])


def train_bc_policy(train_path: Path) -> StandardizedBCPolicy:
    train = pl.read_parquet(train_path)
    state_cols = sorted(c for c in train.columns if c.startswith("s_") and not c.startswith("ns_"))
    x = train.select(state_cols).to_numpy().astype(np.float64)
    y = train["action"].to_numpy().astype(np.int64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    x = (x - mean) / std

    n_rows, n_features = x.shape
    n_actions = 25
    rng = np.random.default_rng(42)
    weights = rng.normal(0.0, 0.01, size=(n_features, n_actions))
    biases = np.zeros(n_actions)

    lr = 0.2
    batch_size = 4096
    epochs = 25
    reg = 1e-4
    for _ in range(epochs):
        indices = rng.permutation(n_rows)
        for start in range(0, n_rows, batch_size):
            batch_idx = indices[start : start + batch_size]
            xb = x[batch_idx]
            yb = y[batch_idx]
            logits = xb @ weights + biases
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1.0
            weights -= lr * ((xb.T @ probs) / len(yb) + reg * weights)
            biases -= lr * probs.mean(axis=0)

    return StandardizedBCPolicy(weights, biases, mean, std)


def evaluate_policy(name, policy, episodes, support):
    metrics, _ = compute_wis_and_ess(episodes, policy, gamma=GAMMA, max_importance_ratio=50.0)
    rows = _row_diagnostics(policy, episodes, support, min_support_prob=0.001, min_support_count=25)
    return {
        "wis": float(metrics.wis),
        "ess": float(metrics.ess),
        "wis_weight_sum": float(metrics.wis_weight_sum),
        "wis_nonzero_episodes": int(metrics.wis_nonzero_episodes),
        "mean_behavior_return": float(metrics.mean_behavior_return),
        "clinician_agreement": float(np.mean([row["agreement"] for row in rows])),
        "support_mass": float(1.0 - np.mean([row["low_support"] for row in rows])),
        "low_support_rate": float(np.mean([row["low_support"] for row in rows])),
        "n_episodes": len(episodes),
        "n_steps": len(rows),
    }


def main() -> None:
    train_path = ROOT / "data/replay/transitions_train.parquet"
    test_path = ROOT / "data/replay/transitions_test.parquet"
    out_path = ROOT / "results/iql_final/baseline_ope_metrics.json"
    episodes = _load_episodes(test_path)
    support = _support_by_action(episodes)
    policies = {
        "clinician_behavior_replay": BehaviorReplayPolicy(episodes),
        "no_treatment": NoTreatmentPolicy(),
        "behavior_cloning_softmax": train_bc_policy(train_path),
    }
    results = {name: evaluate_policy(name, policy, episodes, support) for name, policy in policies.items()}

    bc = policies["behavior_cloning_softmax"]
    correct = 0
    total = 0
    for episode in episodes:
        for step in episode.steps:
            correct += int(bc.select_action(step.state) == step.action)
            total += 1
    results["behavior_cloning_softmax"]["test_action_accuracy"] = correct / total

    returns = [sum((GAMMA**idx) * step.reward for idx, step in enumerate(ep.steps)) for ep in episodes]
    results["clinician_observed_empirical_return"] = {
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "n_episodes": len(returns),
        "note": "Observed test return, not counterfactual FQE.",
    }
    results["_note"] = (
        "FQE for baseline policies requires a frozen action-value evaluator fitted off the test split. "
        "The existing final artifact reports IQL actor-logit scores, not reusable baseline FQE values."
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
