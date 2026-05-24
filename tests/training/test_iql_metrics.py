"""Regression tests for IQL diagnostics and checkpoint loading."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import torch

from mimic_sepsis_rl.training.common import ReplayDataset, TransitionBatch
from mimic_sepsis_rl.training.config import build_training_config
from mimic_sepsis_rl.training.iql import IQLTrainer, load_iql_policy


def _replay_path(tmp_path: Path) -> Path:
    rows = []
    for idx in range(8):
        rows.append(
            {
                "action": idx % 3,
                "reward": float(idx % 2),
                "done": idx == 7,
                "s_0": float(idx),
                "s_1": float(idx + 1),
                "ns_0": float(idx + 0.5),
                "ns_1": float(idx + 1.5),
            }
        )
    path = tmp_path / "replay.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


def _trainer(tmp_path: Path) -> IQLTrainer:
    cfg = build_training_config(
        algorithm="iql",
        device="cpu",
        dataset_path=_replay_path(tmp_path),
        n_epochs=1,
        batch_size=4,
        gamma=0.99,
        seed=42,
        log_dir=tmp_path / "runs",
        checkpoint_dir=tmp_path / "checkpoints",
        experiment_name="iql_metrics_test",
        extra={
            "policy_hidden_sizes": [8],
            "value_hidden_sizes": [8],
            "critic_hidden_sizes": [8],
            "max_adv_weight": 1.0,
        },
    )
    dataset = ReplayDataset(cfg.dataset_path, device=cfg.device, seed=cfg.runtime.seed)
    return IQLTrainer(cfg, dataset, n_actions=3)


def test_iql_training_step_logs_advantage_weight_diagnostics(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    generator = torch.Generator().manual_seed(7)
    batch = TransitionBatch(
        states=torch.randn(4, 2, generator=generator),
        actions=torch.as_tensor([0, 1, 2, 0], dtype=torch.int64),
        rewards=torch.full((4,), 1.0),
        next_states=torch.randn(4, 2, generator=generator),
        dones=torch.full((4,), 0.0),
    )

    metrics = trainer._training_step(batch)  # noqa: SLF001 - diagnostic contract regression

    assert "adv_weight_clip_fraction" in metrics
    assert "adv_weight_mean" in metrics
    assert "adv_weight_max_raw" in metrics
    assert 0.0 <= metrics["adv_weight_clip_fraction"] <= 1.0
    assert metrics["adv_weight_mean"] <= 1.0


def test_load_iql_policy_restores_actor_checkpoint(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    result = trainer.train()

    assert result.checkpoint_path is not None
    policy = load_iql_policy(
        result.checkpoint_path,
        state_dim=2,
        n_actions=3,
        hidden_sizes=[8],
        device="cpu",
    )

    action = policy.select_action((0.0, 1.0))
    assert 0 <= action < 3
    assert len(policy.action_scores((0.0, 1.0))) == 3


def test_iql_sweep_mock_mode_writes_all_metric_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "mock_iql_eval"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_iql_sweep.py",
            "--mock",
            "--mock-episodes",
            "8",
            "--mock-steps",
            "6",
            "--mock-checkpoints",
            "3",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    summary_path = Path(payload["summary"])
    summary = json.loads(summary_path.read_text())

    assert summary["mock"] is True
    assert summary["n_checkpoints"] == 3
    for artifact_path in summary["artifacts"].values():
        path = Path(artifact_path)
        assert path.exists(), artifact_path
        assert path.stat().st_size > 0, artifact_path
