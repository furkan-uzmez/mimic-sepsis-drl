from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from scripts.audit_iql_presweep import main as audit_main


def _write_manifest(root: Path) -> None:
    split_dir = root / "data" / "splits"
    split_dir.mkdir(parents=True)
    frames = {
        "train": [1, 2],
        "validation": [3],
        "test": [4],
    }
    for split, subjects in frames.items():
        pl.DataFrame(
            {
                "subject_id": subjects,
                "split": [split] * len(subjects),
                "episode_keys": [[subject * 10] for subject in subjects],
                "n_episodes": [1] * len(subjects),
            }
        ).write_parquet(split_dir / f"{split}_manifest.parquet")
    (split_dir / "split_summary.json").write_text(
        json.dumps(
            {
                "spec_version": "1.0.0",
                "seed": 42,
                "source_episode_set": "synthetic",
                "has_leakage": False,
            }
        )
    )


def _write_replay_pair(root: Path, split: str, stay_id: int, subject_id: int) -> None:
    rows = [
        {
            "stay_id": stay_id,
            "subject_id": subject_id,
            "step_index": 0,
            "action": 1,
            "reward": 0.0,
            "done": False,
            "terminal_outcome": 0,
            "s_map": 70.0,
            "ns_map": 72.0,
        },
        {
            "stay_id": stay_id,
            "subject_id": subject_id,
            "step_index": 1,
            "action": 2,
            "reward": 15.0,
            "done": True,
            "terminal_outcome": 0,
            "s_map": 72.0,
            "ns_map": 72.0,
        },
    ]
    for replay_dir, reward_version in (("replay", "sofa_shaped:1.0.0"), ("replay_sparse", "sparse:1.0.0")):
        out_dir = root / "data" / replay_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows).write_parquet(out_dir / f"replay_{split}.parquet")
        (out_dir / f"replay_{split}_meta.json").write_text(
            json.dumps(
                {
                    "spec_version": "1.0.0",
                    "replay_buffer_version": "1.0.0",
                    "split_label": split,
                    "manifest_seed": 42,
                    "action_spec_version": "1.0.0",
                    "reward_spec_version": reward_version,
                    "feature_columns": ["map"],
                    "state_dim": 1,
                    "n_actions": 25,
                    "n_episodes": 1,
                    "n_transitions": 2,
                    "provenance": {
                        "split_manifest_dir": "data/splits",
                        "preprocessing_artifact": "data/processed/features/state_vectors/preprocessing_artifacts.json",
                        "action_bins_path": "data/processed/actions/action_bins.json",
                    },
                }
            )
        )


def _write_lineage(root: Path) -> None:
    action_dir = root / "data" / "processed" / "actions"
    action_dir.mkdir(parents=True)
    (action_dir / "action_bins.json").write_text(
        json.dumps({"spec_version": "1.0.0", "fit_split": "train", "manifest_seed": 42})
    )
    feature_dir = root / "data" / "processed" / "features" / "state_vectors"
    feature_dir.mkdir(parents=True)
    (feature_dir / "preprocessing_artifacts.json").write_text(
        json.dumps({"spec_version": "1.0.0", "fit_split": "train", "manifest_seed": 42})
    )


def test_presweep_audit_writes_passing_report(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_manifest(tmp_path)
    _write_lineage(tmp_path)
    for split, stay_id, subject_id in (("train", 10, 1), ("validation", 30, 3), ("test", 40, 4)):
        _write_replay_pair(tmp_path, split, stay_id, subject_id)

    exit_code = audit_main([])

    assert exit_code == 0
    report_path = tmp_path / "results" / "iql_final" / "audit" / "presweep_audit.json"
    report = json.loads(report_path.read_text())
    assert report["status"] == "pass"
    assert report["summary"]["failed_checks"] == 0
    assert report["checks"]["split_leakage"]["status"] == "pass"


def test_presweep_audit_fails_on_split_leakage(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_manifest(tmp_path)
    _write_lineage(tmp_path)
    for split, stay_id, subject_id in (("train", 10, 1), ("validation", 30, 1), ("test", 40, 4)):
        _write_replay_pair(tmp_path, split, stay_id, subject_id)

    exit_code = audit_main([])

    assert exit_code == 1
    report = json.loads((tmp_path / "results" / "iql_final" / "audit" / "presweep_audit.json").read_text())
    assert report["status"] == "fail"
    assert report["checks"]["split_leakage"]["status"] == "fail"
