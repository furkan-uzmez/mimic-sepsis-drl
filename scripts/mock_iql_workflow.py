#!/usr/bin/env python3
"""Fast mock artifact writer for the IQL Snakemake DAG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def mock_inputs() -> None:
    for split in ("train", "validation", "test"):
        write_text(Path(f"data/replay/replay_{split}.parquet"), "mock sofa replay\n")
        write_text(Path(f"data/replay_sparse/replay_{split}.parquet"), "mock sparse replay\n")
        write_json(Path(f"data/replay/replay_{split}_meta.json"), {"split": split, "reward_variant": "sofa_shaped"})
        write_json(Path(f"data/replay_sparse/replay_{split}_meta.json"), {"split": split, "reward_variant": "sparse"})
        write_text(Path(f"data/splits/{split}_manifest.parquet"), "mock split\n")
    write_json(Path("data/splits/split_summary.json"), {"mock": True})
    write_json(Path("data/processed/actions/action_bins.json"), {"fit_split": "train"})
    write_json(
        Path("data/processed/features/state_vectors/preprocessing_artifacts.json"),
        {"fit_split": "train"},
    )


def mock_audit(output: Path) -> None:
    write_json(
        output,
        {
            "mock": True,
            "status": "pass",
            "checks": {
                "artifact_presence": {"status": "pass"},
                "shared_replay_contract": {"status": "pass"},
                "split_leakage": {"status": "pass"},
                "temporal_alignment": {"status": "pass"},
            },
        },
    )


def mock_validation(output: Path) -> None:
    rows = [
        "config_id,fqe_mean,wis_mean,ess,support_mass,low_support_rate,clinician_agreement,action_entropy,severe_safety_flags",
        "iql_sparse_baseline_safe,1.35,1.30,80,0.90,0.10,0.35,2.2,0",
        "iql_sofa_shaped_baseline_safe,1.55,1.50,82,0.91,0.09,0.36,2.1,0",
    ]
    write_text(output, "\n".join(rows) + "\n")


def mock_final(output_root: Path) -> None:
    write_json(
        output_root / "stage2" / "stage2_summary.json",
        {"mock": True, "n_finalists": 6, "n_seeds": 3},
    )
    write_json(
        output_root / "final_metrics.json",
        {
            "mock": True,
            "selected_model": "iql_sofa_shaped_baseline_safe",
            "fqe_mean": 1.55,
            "wis_mean": 1.50,
            "ess": 82.0,
            "support_mass": 0.91,
        },
    )
    write_text(
        output_root / "final_comparison.csv",
        "model,fqe_mean,wis_mean,ess,support_mass\nclinician,1.20,1.18,100,1.00\niql,1.55,1.50,82,0.91\n",
    )
    write_text(
        output_root / "final_report.md",
        "# Mock IQL Final Report\n\nMock Snakemake run completed without launching real training.\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inputs", "audit", "validation", "final"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results/iql_final"))
    args = parser.parse_args()

    if args.command == "inputs":
        mock_inputs()
    elif args.command == "audit":
        mock_audit(args.output or Path("results/iql_final/audit/presweep_audit.json"))
    elif args.command == "validation":
        mock_validation(args.output or Path("results/iql_final/stage1/validation_metrics.csv"))
    elif args.command == "final":
        mock_final(args.output_root)


if __name__ == "__main__":
    main()
