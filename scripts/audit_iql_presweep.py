#!/usr/bin/env python
"""Pre-sweep data artifact audit for the final IQL workflow."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import polars as pl

SPLITS = ("train", "validation", "test")
DEFAULT_SOFA_REPLAY_DIR = Path("data/replay")
DEFAULT_SPARSE_REPLAY_DIR = Path("data/replay_sparse")
DEFAULT_SPLIT_DIR = Path("data/splits")
DEFAULT_OUTPUT_PATH = Path("results/iql_final/audit/presweep_audit.json")
DEFAULT_ACTION_BINS = Path("data/processed/actions/action_bins.json")
DEFAULT_PREPROCESSING = Path("data/processed/features/state_vectors/preprocessing_artifacts.json")
LEAKAGE_FEATURE_HINTS = (
    "mortality",
    "death",
    "deathtime",
    "discharge",
    "survival",
    "outcome",
    "future",
)
ALLOWED_NON_STATE_OUTCOME_COLUMNS = {"terminal_outcome"}


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    message: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit final IQL replay artifacts before sweeps.")
    parser.add_argument("--sofa-replay-dir", type=Path, default=DEFAULT_SOFA_REPLAY_DIR)
    parser.add_argument("--sparse-replay-dir", type=Path, default=DEFAULT_SPARSE_REPLAY_DIR)
    parser.add_argument("--split-manifest-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--action-bins", type=Path, default=DEFAULT_ACTION_BINS)
    parser.add_argument("--preprocessing-artifact", type=Path, default=DEFAULT_PREPROCESSING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_replay_bundle(replay_dir: Path) -> dict[str, dict[str, Any]]:
    bundle: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        data_path = replay_dir / f"replay_{split}.parquet"
        meta_path = replay_dir / f"replay_{split}_meta.json"
        if not data_path.exists() or not meta_path.exists():
            missing = [str(path) for path in (data_path, meta_path) if not path.exists()]
            raise FileNotFoundError(f"Missing replay artifacts for {split}: {missing}")
        bundle[split] = {
            "data_path": str(data_path),
            "meta_path": str(meta_path),
            "data": pl.read_parquet(data_path),
            "meta": _load_json(meta_path),
        }
    return bundle


def _read_manifests(split_dir: Path) -> dict[str, pl.DataFrame]:
    manifests: dict[str, pl.DataFrame] = {}
    for split in SPLITS:
        path = split_dir / f"{split}_manifest.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing split manifest: {path}")
        manifests[split] = pl.read_parquet(path)
    summary = split_dir / "split_summary.json"
    if not summary.exists():
        raise FileNotFoundError(f"Missing split summary: {summary}")
    return manifests


def _status(ok: bool) -> str:
    return "pass" if ok else "fail"


def _column_values(df: pl.DataFrame, column: str) -> set[Any]:
    if column not in df.columns:
        return set()
    return set(df.get_column(column).drop_nulls().to_list())


def _meta_lineage(meta: dict[str, Any], key: str) -> Any:
    return meta.get(key, meta.get("provenance", {}).get(key))


def _check_artifact_presence(
    sofa: dict[str, dict[str, Any]],
    sparse: dict[str, dict[str, Any]],
    manifests: dict[str, pl.DataFrame],
) -> CheckResult:
    details = {
        "sofa_replay": {split: {"rows": sofa[split]["data"].height, "path": sofa[split]["data_path"]} for split in SPLITS},
        "sparse_replay": {split: {"rows": sparse[split]["data"].height, "path": sparse[split]["data_path"]} for split in SPLITS},
        "manifests": {split: {"rows": manifests[split].height} for split in SPLITS},
    }
    ok = all(details[group][split]["rows"] > 0 for group in ("sofa_replay", "sparse_replay", "manifests") for split in SPLITS)
    return CheckResult("artifact_presence", _status(ok), "Required replay and split artifacts are present and non-empty." if ok else "At least one required artifact is empty.", details)


def _check_shared_contracts(sofa: dict[str, dict[str, Any]], sparse: dict[str, dict[str, Any]]) -> CheckResult:
    mismatches: list[dict[str, Any]] = []
    for split in SPLITS:
        sofa_meta = sofa[split]["meta"]
        sparse_meta = sparse[split]["meta"]
        for key in ("split_label", "manifest_seed", "action_spec_version", "state_dim", "n_actions", "feature_columns"):
            if sofa_meta.get(key) != sparse_meta.get(key):
                mismatches.append({"split": split, "field": key, "sofa": sofa_meta.get(key), "sparse": sparse_meta.get(key)})
        sofa_keys = sofa[split]["data"].select(["stay_id", "step_index"]).sort(["stay_id", "step_index"])
        sparse_keys = sparse[split]["data"].select(["stay_id", "step_index"]).sort(["stay_id", "step_index"])
        if not sofa_keys.equals(sparse_keys):
            mismatches.append({"split": split, "field": "transition_index", "message": "stay_id/step_index keys differ"})
        for col in ("terminal_outcome", "done"):
            if col in sofa[split]["data"].columns and col in sparse[split]["data"].columns:
                left = sofa[split]["data"].select(["stay_id", "step_index", col]).sort(["stay_id", "step_index"])
                right = sparse[split]["data"].select(["stay_id", "step_index", col]).sort(["stay_id", "step_index"])
                if not left.equals(right):
                    mismatches.append({"split": split, "field": col, "message": f"{col} differs between reward variants"})
    ok = not mismatches
    return CheckResult("shared_replay_contract", _status(ok), "Sparse and SOFA-shaped replay artifacts share split, action, preprocessing, indexing, and terminal contracts." if ok else "Sparse and SOFA-shaped replay contracts diverge.", {"mismatches": mismatches})


def _check_train_fit_artifacts(
    sofa: dict[str, dict[str, Any]],
    sparse: dict[str, dict[str, Any]],
    action_bins: Path,
    preprocessing: Path,
) -> CheckResult:
    issues: list[str] = []
    payloads: dict[str, Any] = {}
    for label, path in (("action_bins", action_bins), ("preprocessing", preprocessing)):
        if not path.exists():
            issues.append(f"Missing {label}: {path}")
            continue
        payload = _load_json(path)
        payloads[label] = payload
        fit_split = payload.get("fit_split", payload.get("fitted_on_split", payload.get("source_split")))
        if fit_split is not None and fit_split != "train":
            issues.append(f"{label} was not fit on train split: {fit_split}")
    for split in SPLITS:
        for variant, bundle in (("sofa", sofa), ("sparse", sparse)):
            meta = bundle[split]["meta"]
            action_path = _meta_lineage(meta, "action_bins_path")
            preprocessing_path = _meta_lineage(meta, "preprocessing_artifact")
            if action_path is not None and Path(action_path).name != action_bins.name:
                issues.append(f"{variant}/{split} points to unexpected action bins: {action_path}")
            if preprocessing_path is not None and Path(preprocessing_path).name != preprocessing.name:
                issues.append(f"{variant}/{split} points to unexpected preprocessing artifact: {preprocessing_path}")
    ok = not issues
    return CheckResult("train_only_fit_artifacts", _status(ok), "Action bins and preprocessing artifacts are train-fit lineage compatible." if ok else "Train-only fit lineage checks failed.", {"issues": issues, "artifacts": payloads})


def _check_temporal_alignment(bundle: dict[str, dict[str, Any]]) -> CheckResult:
    issues: list[dict[str, Any]] = []
    for split in SPLITS:
        df = bundle[split]["data"].sort(["stay_id", "step_index"])
        for stay_id in _column_values(df, "stay_id"):
            ep = df.filter(pl.col("stay_id") == stay_id).sort("step_index")
            steps = ep.get_column("step_index").to_list()
            expected = list(range(int(steps[0]), int(steps[0]) + len(steps))) if steps else []
            if steps != expected:
                issues.append({"split": split, "stay_id": stay_id, "issue": "non_contiguous_step_index"})
            done = ep.get_column("done").to_list() if "done" in ep.columns else []
            if done and (sum(bool(value) for value in done) != 1 or not bool(done[-1])):
                issues.append({"split": split, "stay_id": stay_id, "issue": "invalid_terminal_done_flags"})
            state_cols = [col for col in ep.columns if col.startswith("s_")]
            for row_index in range(max(0, ep.height - 1)):
                row = ep.row(row_index, named=True)
                next_row = ep.row(row_index + 1, named=True)
                for state_col in state_cols:
                    next_state_col = f"ns_{state_col[2:]}"
                    if next_state_col in ep.columns:
                        left = row[next_state_col]
                        right = next_row[state_col]
                        if not _numbers_equal(left, right):
                            issues.append({"split": split, "stay_id": stay_id, "step_index": row["step_index"], "issue": "next_state_mismatch", "feature": state_col[2:]})
                            break
    ok = not issues
    return CheckResult("temporal_alignment", _status(ok), "s_t, a_t, s_t+1, r_t, and terminal flags are aligned." if ok else "Temporal alignment violations found.", {"issues": issues[:50], "n_issues": len(issues)})


def _numbers_equal(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-8, abs_tol=1e-8)
    except (TypeError, ValueError):
        return left == right


def _check_split_leakage(manifests: dict[str, pl.DataFrame], bundle: dict[str, dict[str, Any]]) -> CheckResult:
    split_subjects: dict[str, set[Any]] = {split: _column_values(manifests[split], "subject_id") for split in SPLITS}
    for split in SPLITS:
        replay_subjects = _column_values(bundle[split]["data"], "subject_id")
        if replay_subjects:
            split_subjects[split] |= replay_subjects
    overlaps: list[dict[str, Any]] = []
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1 :]:
            shared = sorted(split_subjects[left] & split_subjects[right])
            if shared:
                overlaps.append({"left": left, "right": right, "subject_ids": shared[:20], "n_shared": len(shared)})
    ok = not overlaps
    return CheckResult("split_leakage", _status(ok), "No patient appears in more than one split." if ok else "Patient-level split leakage detected.", {"overlaps": overlaps})


def _check_outcome_leakage(bundle: dict[str, dict[str, Any]]) -> CheckResult:
    bad_columns: dict[str, list[str]] = {}
    for split in SPLITS:
        cols = []
        for col in bundle[split]["data"].columns:
            lower = col.lower()
            is_state_col = lower.startswith("s_") or lower.startswith("ns_")
            if is_state_col and any(hint in lower for hint in LEAKAGE_FEATURE_HINTS) and lower not in ALLOWED_NON_STATE_OUTCOME_COLUMNS:
                cols.append(col)
        if cols:
            bad_columns[split] = cols
    ok = not bad_columns
    return CheckResult("outcome_leakage", _status(ok), "No mortality/discharge/future outcome hints appear in state feature columns." if ok else "Potential outcome leakage columns found in state features.", {"columns": bad_columns})


def _write_report(path: Path, checks: Iterable[CheckResult]) -> dict[str, Any]:
    check_list = list(checks)
    failed = [check for check in check_list if check.status != "pass"]
    report = {
        "status": "pass" if not failed else "fail",
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "summary": {
            "total_checks": len(check_list),
            "failed_checks": len(failed),
            "failed_check_names": [check.name for check in failed],
        },
        "checks": {check.name: check.to_dict() for check in check_list},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    sofa = _read_replay_bundle(args.sofa_replay_dir)
    sparse = _read_replay_bundle(args.sparse_replay_dir)
    manifests = _read_manifests(args.split_manifest_dir)
    checks = [
        _check_artifact_presence(sofa, sparse, manifests),
        _check_shared_contracts(sofa, sparse),
        _check_train_fit_artifacts(sofa, sparse, args.action_bins, args.preprocessing_artifact),
        _check_temporal_alignment(sofa),
        _check_split_leakage(manifests, sofa),
        _check_outcome_leakage(sofa),
    ]
    return _write_report(args.output, checks)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_audit(args)
    except Exception as exc:  # noqa: BLE001 - CLI must persist a failure report for operators.
        report = _write_report(
            args.output,
            [CheckResult("audit_execution", "fail", str(exc), {"error_type": type(exc).__name__})],
        )
    print(json.dumps({"status": report["status"], "output": str(args.output)}, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
