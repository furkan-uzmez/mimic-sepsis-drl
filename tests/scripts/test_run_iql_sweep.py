import json
from pathlib import Path

import yaml

from scripts.run_iql_sweep import (
    DEFAULT_STAGE1_SEED,
    STAGE2_EXTRA_SEEDS,
    build_iql_grid,
    build_sweep_config,
    run_iql_sweep,
)


def test_build_iql_grid_has_18_named_configs() -> None:
    grid = build_iql_grid()

    assert len(grid) == 18
    assert len({spec.config_id for spec in grid}) == 18
    assert {spec.reward_variant for spec in grid} == {"sparse", "sofa_shaped"}
    assert {spec.lr_regime for spec in grid} == {"conservative", "baseline", "fast_value"}
    assert {spec.iql_setting for spec in grid} == {"safe", "baseline", "optimistic"}


def test_build_sweep_config_sets_dataset_and_iql_hyperparameters() -> None:
    base = yaml.safe_load(Path("configs/training/iql.yaml").read_text())
    spec = next(
        item
        for item in build_iql_grid()
        if item.reward_variant == "sparse"
        and item.lr_regime == "baseline"
        and item.iql_setting == "safe"
    )

    cfg = build_sweep_config(base, spec=spec, seed=123, output_root=Path("runs/iql_final"))

    assert cfg["runtime"]["seed"] == 123
    assert cfg["dataset_path"] == "data/replay_sparse/replay_train.parquet"
    assert cfg["dataset_meta_path"] == "data/replay_sparse/replay_train_meta.json"
    assert cfg["gamma"] == 0.99
    assert cfg["actor_lr"] == spec.actor_lr
    assert cfg["critic_lr"] == spec.critic_lr
    assert cfg["value_lr"] == spec.value_lr
    assert cfg["expectile"] == spec.expectile
    assert cfg["temperature"] == spec.temperature
    assert cfg["logging"]["experiment_name"] == f"{spec.config_id}_seed123"
    assert cfg["extra"]["iql_grid"]["config_id"] == spec.config_id


def test_mock_stage1_writes_deterministic_manifest(tmp_path: Path) -> None:
    output_root = tmp_path / "iql_final"

    _, manifest = run_iql_sweep(
        stage=1,
        output_root=output_root,
        mock=True,
        dry_run=True,
    )

    manifest_path = output_root / "stage1" / "stage1_manifest.json"
    grid_path = output_root / "grid" / "iql_grid_manifest.json"
    assert manifest_path.exists()
    assert grid_path.exists()
    assert manifest.summary == {"total": 18, "completed": 18, "failed": 0}
    assert {run["seed"] for run in manifest.runs} == {DEFAULT_STAGE1_SEED}
    assert all(run["success"] for run in manifest.runs)

    first_payload = json.loads(manifest_path.read_text())
    _, second_manifest = run_iql_sweep(
        stage=1,
        output_root=output_root,
        mock=True,
        dry_run=True,
    )
    second_payload = json.loads(manifest_path.read_text())

    assert first_payload == second_payload
    assert second_manifest.runs == manifest.runs


def test_mock_stage2_uses_finalists_and_extra_seeds(tmp_path: Path) -> None:
    output_root = tmp_path / "iql_final"
    run_iql_sweep(stage=1, output_root=output_root, mock=True, dry_run=True)

    _, manifest = run_iql_sweep(stage=2, output_root=output_root, mock=True, dry_run=True)

    assert manifest.summary == {"total": 12, "completed": 12, "failed": 0}
    assert {run["seed"] for run in manifest.runs} == set(STAGE2_EXTRA_SEEDS)
    assert len({run["config_id"] for run in manifest.runs}) == 6
    assert (output_root / "stage2" / "finalists_manifest.json").exists()
