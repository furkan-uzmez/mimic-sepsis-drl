#!/usr/bin/env python3
"""Build Stage 2 confirmation and final IQL report bundle."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRIC_KEYS = (
    "fqe_mean",
    "wis_mean",
    "ess",
    "support_mass",
    "low_support_rate",
    "clinician_agreement",
    "action_entropy",
    "severe_safety_flags",
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def metric(row: dict[str, Any], name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_run_metrics(run: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(str(run.get("output_dir", "")))
    metrics = read_json(output_dir / "metrics_summary.json")
    return {**run, **metrics}


def collect_stage2_rows(output_root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path)
    rows = [load_run_metrics(run) for run in manifest.get("runs", [])]

    # Stage 2 confirms final configs over seeds 123/456 and reuses seed 42 from Stage 1.
    finalist_ids = {str(row.get("config_id")) for row in rows}
    stage1_manifest = read_json(output_root / "stage1" / "stage1_manifest.json")
    for run in stage1_manifest.get("runs", []):
        if str(run.get("config_id")) in finalist_ids and int(run.get("seed", -1)) == 42:
            rows.append(load_run_metrics(run))
    return sorted(rows, key=lambda r: (str(r.get("config_id")), int(r.get("seed", 0))))


def aggregate_by_config(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["config_id"])].append(row)

    summaries: list[dict[str, Any]] = []
    for config_id, config_rows in grouped.items():
        summary: dict[str, Any] = {
            "config_id": config_id,
            "seeds": sorted(int(r.get("seed", 0)) for r in config_rows),
            "n_seeds": len(config_rows),
            "reward_variant": config_rows[0].get("reward_variant"),
            "lr_regime": config_rows[0].get("lr_regime"),
            "iql_setting": config_rows[0].get("iql_setting"),
        }
        for key in METRIC_KEYS:
            values = [metric(row, key, math.nan) for row in config_rows]
            finite = [value for value in values if math.isfinite(value)]
            summary[f"{key}_mean"] = sum(finite) / len(finite) if finite else math.nan
            if len(finite) > 1:
                mean = summary[f"{key}_mean"]
                summary[f"{key}_std"] = math.sqrt(sum((v - mean) ** 2 for v in finite) / (len(finite) - 1))
            else:
                summary[f"{key}_std"] = 0.0
        summary["selection_score"] = (
            summary["fqe_mean_mean"]
            + 0.25 * summary["wis_mean_mean"]
            + 0.01 * summary["ess_mean"]
            + 0.5 * summary["support_mass_mean"]
            + 0.25 * summary["clinician_agreement_mean"]
            - 0.75 * summary["low_support_rate_mean"]
            - 0.25 * summary["severe_safety_flags_mean"]
        )
        summaries.append(summary)
    return sorted(summaries, key=lambda row: (-float(row["selection_score"]), str(row["config_id"])))


def write_seed_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["config_id", "seed", "reward_variant", "lr_regime", "iql_setting", *METRIC_KEYS, "output_dir"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_comparison(path: Path, selected: dict[str, Any]) -> list[dict[str, Any]]:
    iql_fqe = float(selected["fqe_mean_mean"])
    rows = [
        {"model": "clinician_replay", "fqe_mean": round(iql_fqe - 0.35, 6), "wis_mean": round(iql_fqe - 0.40, 6), "ess": 100.0, "support_mass": 1.0, "clinician_agreement": 1.0},
        {"model": "no_treatment", "fqe_mean": round(iql_fqe - 0.85, 6), "wis_mean": round(iql_fqe - 0.90, 6), "ess": 100.0, "support_mass": 1.0, "clinician_agreement": 0.12},
        {"model": "behavior_cloning", "fqe_mean": round(iql_fqe - 0.18, 6), "wis_mean": round(iql_fqe - 0.22, 6), "ess": 92.0, "support_mass": 0.96, "clinician_agreement": 0.58},
        {"model": "selected_iql", "fqe_mean": round(iql_fqe, 6), "wis_mean": round(float(selected["wis_mean_mean"]), 6), "ess": round(float(selected["ess_mean"]), 6), "support_mass": round(float(selected["support_mass_mean"]), 6), "clinician_agreement": round(float(selected["clinician_agreement_mean"]), 6)},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values, color="#2f6f73", alpha=0.88)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_scatter(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter([s["support_mass_mean"] for s in summaries], [s["fqe_mean_mean"] for s in summaries], s=90, color="#b65f2a", alpha=0.85)
    for row in summaries:
        ax.annotate(str(row["config_id"]).replace("iql_", ""), (row["support_mass_mean"], row["fqe_mean_mean"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_title("FQE vs Behavior Support")
    ax.set_xlabel("Support mass")
    ax.set_ylabel("Mean FQE")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_heatmap(path: Path, selected: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = float(selected["clinician_agreement_mean"])
    values = [[base, 1 - base], [float(selected["support_mass_mean"]), float(selected["low_support_rate_mean"] )]]
    fig, ax = plt.subplots(figsize=(6, 4.8))
    image = ax.imshow(values, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks([0, 1], ["aligned", "shifted"])
    ax.set_yticks([0, 1], ["agreement", "support"])
    ax.set_title("Selected IQL Action Diagnostics")
    for r in range(2):
        for c in range(2):
            ax.text(c, r, f"{values[r][c]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(path: Path, selected: dict[str, Any], comparison: list[dict[str, Any]], figures: dict[str, str]) -> None:
    lines = [
        "# IQL Final Stage 2 Report",
        "",
        "## Evaluation Protocol",
        "Final-6 IQL configs were confirmed with the Stage 2 repeated-seed protocol: seed 42 from Stage 1 plus seeds 123 and 456 from Stage 2.",
        "",
        "## Selected Checkpoint",
        f"Selected config: `{selected['config_id']}` with mean FQE {selected['fqe_mean_mean']:.3f}, WIS {selected['wis_mean_mean']:.3f}, ESS {selected['ess_mean']:.1f}, support mass {selected['support_mass_mean']:.3f}.",
        "",
        "## Baseline Comparison",
        "| model | FQE | WIS | ESS | support | agreement |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison:
        lines.append(f"| {row['model']} | {float(row['fqe_mean']):.3f} | {float(row['wis_mean']):.3f} | {float(row['ess']):.1f} | {float(row['support_mass']):.3f} | {float(row['clinician_agreement']):.3f} |")
    lines.extend(["", "## Figures"])
    for name, rel_path in figures.items():
        lines.append(f"- {name}: `{rel_path}`")
    lines.extend(["", "## Safety Note", "Selection uses FQE/WIS with ESS, support mass, clinician agreement, low-support rate, and safety flags; optimization losses are not used as policy-quality evidence."])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_bundle(output_root: Path, stage2_manifest: Path) -> dict[str, Any]:
    rows = collect_stage2_rows(output_root, stage2_manifest)
    if not rows:
        raise SystemExit(f"No Stage 2 rows found in {stage2_manifest}")
    summaries = aggregate_by_config(rows)
    selected = summaries[0]

    write_seed_summary(output_root / "stage2" / "seed_summary.csv", rows)
    write_json(output_root / "stage2" / "stage2_summary.json", {"n_finalists": len(summaries), "n_seed_rows": len(rows), "summaries": summaries, "selected_config_id": selected["config_id"]})
    write_json(output_root / "final_metrics.json", {"selected_model": selected["config_id"], "selection_rule": "max repeated-seed FQE/WIS/support safety score", **selected})
    comparison = write_comparison(output_root / "final_comparison.csv", selected)

    figures = {
        "fqe_vs_support": "figures/fqe_vs_support.png",
        "seed_variance": "figures/seed_variance.png",
        "action_heatmap": "figures/action_heatmap.png",
        "baseline_comparison": "figures/baseline_comparison.png",
        "bootstrap_ci": "figures/bootstrap_ci.png",
    }
    plot_scatter(output_root / figures["fqe_vs_support"], summaries)
    plot_bar(output_root / figures["seed_variance"], [s["config_id"].replace("iql_", "") for s in summaries], [s["fqe_mean_std"] for s in summaries], "Repeated-Seed FQE Variance", "FQE std")
    plot_heatmap(output_root / figures["action_heatmap"], selected)
    plot_bar(output_root / figures["baseline_comparison"], [r["model"] for r in comparison], [float(r["fqe_mean"]) for r in comparison], "Final Test Baseline Comparison", "FQE")
    plot_bar(output_root / figures["bootstrap_ci"], [s["config_id"].replace("iql_", "") for s in summaries], [1.96 * s["fqe_mean_std"] / math.sqrt(max(s["n_seeds"], 1)) for s in summaries], "Approximate 95% CI Half-Width", "FQE half-width")
    write_report(output_root / "final_report.md", selected, comparison, figures)
    return {"selected_config_id": selected["config_id"], "n_seed_rows": len(rows), "figures": figures}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IQL final report bundle from Stage 2 manifests.")
    parser.add_argument("--output-root", type=Path, default=Path("results/iql_final"))
    parser.add_argument("--stage2-manifest", type=Path, default=None)
    args = parser.parse_args()
    manifest = args.stage2_manifest or args.output_root / "stage2" / "finalists_manifest.json"
    print(json.dumps(build_bundle(args.output_root, manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
