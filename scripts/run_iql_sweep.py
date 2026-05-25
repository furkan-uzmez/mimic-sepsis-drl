#!/usr/bin/env python3
"""IQL final sweep runner and 18-config grid generator."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger(__name__)

REFERENCE_IQL_CONFIG = "configs/training/iql.yaml"
SWEEP_VERSION = "0.1.0"
DEFAULT_STAGE1_SEED = 42
STAGE2_EXTRA_SEEDS = (123, 456)
FINALIST_COUNT = 6

REWARD_DATASETS = {
    "sparse": {
        "dataset_path": "data/replay_sparse/replay_train.parquet",
        "dataset_meta_path": "data/replay_sparse/replay_train_meta.json",
    },
    "sofa_shaped": {
        "dataset_path": "data/replay/replay_train.parquet",
        "dataset_meta_path": "data/replay/replay_train_meta.json",
    },
}

LR_REGIMES = {
    "conservative": {"actor_lr": 5e-5, "critic_lr": 1e-4, "value_lr": 1e-4},
    "baseline": {"actor_lr": 1e-4, "critic_lr": 3e-4, "value_lr": 3e-4},
    "fast_value": {"actor_lr": 1e-4, "critic_lr": 3e-4, "value_lr": 1e-3},
}

IQL_SETTINGS = {
    "safe": {"expectile": 0.6, "temperature": 2.0},
    "baseline": {"expectile": 0.7, "temperature": 3.0},
    "optimistic": {"expectile": 0.8, "temperature": 5.0},
}


@dataclass(frozen=True)
class IQLGridSpec:
    """Single IQL sweep configuration."""

    config_id: str
    reward_variant: str
    lr_regime: str
    iql_setting: str
    actor_lr: float
    critic_lr: float
    value_lr: float
    expectile: float
    temperature: float
    gamma: float = 0.99

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IQLRunResult:
    config_id: str
    seed: int
    reward_variant: str
    lr_regime: str
    iql_setting: str
    success: bool
    output_dir: str
    config_path: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IQLSweepManifest:
    version: str = SWEEP_VERSION
    stage: int = 1
    seeds: tuple[int, ...] = ()
    grid_size: int = 0
    mock: bool = False
    dry_run: bool = False
    finalist_count: int = FINALIST_COUNT
    runs: list[dict[str, Any]] = field(default_factory=list)
    finalists: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)


def build_iql_grid() -> list[IQLGridSpec]:
    """Return the protocol grid: 2 rewards x 3 LR regimes x 3 IQL settings."""
    specs: list[IQLGridSpec] = []
    for reward_variant in ("sparse", "sofa_shaped"):
        for lr_name, lr_values in LR_REGIMES.items():
            for setting_name, setting_values in IQL_SETTINGS.items():
                config_id = f"iql_{reward_variant}_{lr_name}_{setting_name}"
                specs.append(
                    IQLGridSpec(
                        config_id=config_id,
                        reward_variant=reward_variant,
                        lr_regime=lr_name,
                        iql_setting=setting_name,
                        actor_lr=float(lr_values["actor_lr"]),
                        critic_lr=float(lr_values["critic_lr"]),
                        value_lr=float(lr_values["value_lr"]),
                        expectile=float(setting_values["expectile"]),
                        temperature=float(setting_values["temperature"]),
                    )
                )
    return specs


def load_reference_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping in {config_path}")
    return config


def build_sweep_config(
    base_config: dict[str, Any],
    *,
    spec: IQLGridSpec,
    seed: int,
    output_root: Path,
) -> dict[str, Any]:
    """Build a concrete training config for one grid spec and seed."""
    cfg = copy.deepcopy(base_config)
    run_name = f"{spec.config_id}_seed{seed}"
    dataset_paths = REWARD_DATASETS[spec.reward_variant]

    cfg.setdefault("runtime", {})["seed"] = seed
    cfg["dataset_path"] = dataset_paths["dataset_path"]
    cfg["dataset_meta_path"] = dataset_paths["dataset_meta_path"]
    cfg["gamma"] = spec.gamma
    cfg["actor_lr"] = spec.actor_lr
    cfg["critic_lr"] = spec.critic_lr
    cfg["value_lr"] = spec.value_lr
    cfg["expectile"] = spec.expectile
    cfg["temperature"] = spec.temperature

    cfg.setdefault("checkpoint", {})["checkpoint_dir"] = str(
        Path("checkpoints/iql_final") / run_name
    )
    cfg["checkpoint"]["save_every_n_epochs"] = 20
    cfg["checkpoint"]["keep_last_n"] = 0

    cfg.setdefault("logging", {})["log_dir"] = str(output_root / "logs")
    cfg["logging"]["experiment_name"] = run_name
    cfg.setdefault("extra", {})["iql_grid"] = spec.to_dict()
    return cfg


def write_grid_manifest(output_root: Path, grid: list[IQLGridSpec]) -> Path:
    path = output_root / "grid" / "iql_grid_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": SWEEP_VERSION,
        "grid_size": len(grid),
        "gamma_effective_horizon_steps": 100,
        "configs": [spec.to_dict() for spec in grid],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def select_mock_finalists(grid: list[IQLGridSpec]) -> list[IQLGridSpec]:
    """Deterministic protocol-shaped finalist set for mock DAG validation."""
    preferred_ids = [
        "iql_sparse_baseline_safe",
        "iql_sofa_shaped_baseline_safe",
        "iql_sparse_baseline_baseline",
        "iql_sofa_shaped_baseline_baseline",
        "iql_sparse_conservative_safe",
        "iql_sofa_shaped_fast_value_optimistic",
    ]
    by_id = {spec.config_id: spec for spec in grid}
    return [by_id[config_id] for config_id in preferred_ids]


def load_stage1_finalists(output_root: Path, grid: list[IQLGridSpec]) -> list[IQLGridSpec]:
    """Load real Stage 1 selected finalists when evaluation has produced them."""
    manifest_path = output_root / "stage1" / "selection" / "final6_manifest.json"
    if not manifest_path.exists():
        return select_mock_finalists(grid)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = payload.get("selected", [])
    by_id = {spec.config_id: spec for spec in grid}
    finalists = [by_id[str(row["config_id"])] for row in selected if str(row.get("config_id")) in by_id]
    return finalists[:FINALIST_COUNT] or select_mock_finalists(grid)


def write_mock_selection(output_root: Path, finalists: list[IQLGridSpec]) -> None:
    selection_dir = output_root / "stage1" / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    csv_lines = ["rank,config_id,reward_variant,lr_regime,iql_setting,selection_reason"]
    for rank, spec in enumerate(finalists, start=1):
        csv_lines.append(
            f"{rank},{spec.config_id},{spec.reward_variant},{spec.lr_regime},"
            f"{spec.iql_setting},mock_protocol_slot"
        )
    (selection_dir / "top5_configs.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    (selection_dir / "final6_configs.json").write_text(
        json.dumps([spec.to_dict() for spec in finalists], indent=2, sort_keys=True),
        encoding="utf-8",
    )


def execute_iql_training(config_path: Path, *, dry_run: bool) -> subprocess.CompletedProcess[str]:
    cmd = ["uv", "run", "python", "-m", "mimic_sepsis_rl.training.iql", "--config", str(config_path)]
    if dry_run:
        cmd.append("--dry-run")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=7200)


def run_single_config(
    base_config: dict[str, Any],
    *,
    spec: IQLGridSpec,
    seed: int,
    output_root: Path,
    mock: bool,
    dry_run: bool,
) -> IQLRunResult:
    cfg = build_sweep_config(base_config, spec=spec, seed=seed, output_root=output_root)
    run_dir = output_root / f"stage_seed_{seed}" / spec.config_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "training_config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")

    if mock:
        checkpoint_path = run_dir / "checkpoint.pt"
        checkpoint_path.write_text("mock iql checkpoint\n", encoding="utf-8")
        (run_dir / "metrics_summary.json").write_text(
            json.dumps(mock_metrics(spec, seed), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return IQLRunResult(
            config_id=spec.config_id,
            seed=seed,
            reward_variant=spec.reward_variant,
            lr_regime=spec.lr_regime,
            iql_setting=spec.iql_setting,
            success=True,
            output_dir=str(run_dir),
            config_path=str(config_path),
        )

    proc = execute_iql_training(config_path, dry_run=dry_run)
    return IQLRunResult(
        config_id=spec.config_id,
        seed=seed,
        reward_variant=spec.reward_variant,
        lr_regime=spec.lr_regime,
        iql_setting=spec.iql_setting,
        success=proc.returncode == 0,
        output_dir=str(run_dir),
        config_path=str(config_path),
        error_message=None if proc.returncode == 0 else (proc.stderr or proc.stdout)[-1000:],
    )


def mock_metrics(spec: IQLGridSpec, seed: int) -> dict[str, Any]:
    reward_bonus = 0.2 if spec.reward_variant == "sofa_shaped" else 0.0
    setting_bonus = {"safe": 0.1, "baseline": 0.2, "optimistic": 0.15}[spec.iql_setting]
    lr_bonus = {"conservative": 0.05, "baseline": 0.2, "fast_value": 0.1}[spec.lr_regime]
    fqe = 1.0 + reward_bonus + setting_bonus + lr_bonus + (seed % 100) * 0.001
    return {
        "fqe_mean": round(fqe, 6),
        "fqe_95ci_lower": round(fqe - 0.1, 6),
        "wis_mean": round(fqe - 0.05, 6),
        "ess": 80.0,
        "support_mass": 0.9,
        "low_support_rate": 0.1,
        "clinician_agreement": 0.35,
        "action_entropy": 2.2,
        "severe_safety_flags": 0,
    }


def write_manifest(path: Path, manifest: IQLSweepManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8")


def run_iql_sweep(
    *,
    stage: int,
    output_root: Path,
    reference_config: Path = Path(REFERENCE_IQL_CONFIG),
    mock: bool = False,
    dry_run: bool = False,
) -> tuple[list[IQLRunResult], IQLSweepManifest]:
    grid = build_iql_grid()
    write_grid_manifest(output_root, grid)
    base_config = load_reference_config(reference_config)

    if stage == 1:
        selected_grid = grid
        seeds = (DEFAULT_STAGE1_SEED,)
        manifest_path = output_root / "stage1" / "stage1_manifest.json"
        finalists = select_mock_finalists(grid)
        if mock:
            write_mock_selection(output_root, finalists)
    elif stage == 2:
        selected_grid = load_stage1_finalists(output_root, grid)
        seeds = STAGE2_EXTRA_SEEDS
        manifest_path = output_root / "stage2" / "finalists_manifest.json"
        finalists = selected_grid
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    results = [
        run_single_config(
            base_config,
            spec=spec,
            seed=seed,
            output_root=output_root,
            mock=mock,
            dry_run=dry_run,
        )
        for spec in selected_grid
        for seed in seeds
    ]
    completed = sum(1 for result in results if result.success)
    manifest = IQLSweepManifest(
        stage=stage,
        seeds=seeds,
        grid_size=len(grid),
        mock=mock,
        dry_run=dry_run,
        runs=[result.to_dict() for result in results],
        finalists=[spec.to_dict() for spec in finalists],
        summary={"total": len(results), "completed": completed, "failed": len(results) - completed},
    )
    write_manifest(manifest_path, manifest)
    return results, manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the IQL final sweep protocol.")
    parser.add_argument("--stage", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output-root", type=Path, default=Path("results/iql_final"))
    parser.add_argument("--reference-config", type=Path, default=Path(REFERENCE_IQL_CONFIG))
    parser.add_argument("--mock", action="store_true", help="Write deterministic mock artifacts without training.")
    parser.add_argument("--dry-run", action="store_true", help="Pass --dry-run to IQL training.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    _, manifest = run_iql_sweep(
        stage=args.stage,
        output_root=args.output_root,
        reference_config=args.reference_config,
        mock=args.mock,
        dry_run=args.dry_run,
    )
    print(json.dumps(asdict(manifest), indent=2, sort_keys=True))
    if manifest.summary.get("failed", 0):
        sys.exit(1)


if __name__ == "__main__":
    main()
