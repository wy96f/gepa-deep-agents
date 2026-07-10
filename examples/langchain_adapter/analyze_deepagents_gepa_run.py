"""Summarize a Deep Agents GEPA artifact run.

The optimizer can now persist enough evidence to answer two practical
questions after a run:

1. Did the candidate pool improve?
2. If not, was the bottleneck algorithmic, data/eval related, or runtime?
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = REPO_ROOT / "examples" / "langchain_adapter" / "runs"


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def resolve_run_dir(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    latest = DEFAULT_RUNS_DIR / "latest_run.txt"
    if not latest.exists():
        raise FileNotFoundError(f"No run dir provided and {latest} does not exist")
    return Path(latest.read_text(encoding="utf-8").strip()).expanduser().resolve()


def summarize_run(run_dir: Path) -> dict[str, Any]:
    summary = read_json(run_dir / "result_summary.json", {})
    rollouts = read_jsonl(run_dir / "agent_logs" / "rollouts.jsonl")
    proposals = read_jsonl(run_dir / "proposals" / "index.jsonl")
    rejected_proposals = read_jsonl(run_dir / "rejected_proposals" / "index.jsonl")
    rollout_details = read_rollout_details(run_dir, rollouts)

    scores = [float(row.get("score", 0.0)) for row in rollouts]
    errors = Counter(error for row in rollouts if (error := rollout_error(row)))
    boundary_failures = Counter(
        constraint.get("name")
        for detail in rollout_details
        for constraint in detail.get("constraints", [])
        if isinstance(constraint, dict)
        and not constraint.get("passed", True)
        and ":boundary:" in str(constraint.get("name", ""))
    )
    proposal_statuses = Counter(str(row.get("status", "unknown")) for row in proposals)
    proposed_components = Counter(
        component
        for row in proposals
        for component in row.get("components", [])
    )
    rejected_components = Counter(
        component
        for row in rejected_proposals
        for component in row.get("components", [])
    )
    failure_classes = Counter(
        failure_class
        for row in rejected_proposals
        for failure_class in row.get("failure_classifications", [])
    )
    proposal_artifacts = proposal_artifact_counts(run_dir, proposals, rejected_proposals)

    val_scores = summary.get("val_aggregate_scores") or []
    best_val_score = summary.get("best_val_score")
    baseline_score = val_scores[0] if val_scores else None
    improvement = None
    if baseline_score is not None and best_val_score is not None:
        improvement = float(best_val_score) - float(baseline_score)

    runtime_blocked = bool(rollouts) and sum(errors.values()) == len(rollouts)
    connection_blocked = runtime_blocked and all("APIConnectionError" in key or "ConnectError" in key for key in errors)

    return {
        "run_dir": str(run_dir),
        "best_val_score": best_val_score,
        "baseline_val_score": baseline_score,
        "improvement": improvement,
        "num_candidates": summary.get("num_candidates"),
        "total_metric_calls": summary.get("total_metric_calls"),
        "num_full_val_evals": summary.get("num_full_val_evals"),
        "rollout_count": len(rollouts),
        "rollout_score_mean": sum(scores) / len(scores) if scores else None,
        "rollout_score_min": min(scores) if scores else None,
        "rollout_score_max": max(scores) if scores else None,
        "runtime_errors": dict(errors),
        "boundary_failures": dict(boundary_failures.most_common()),
        "proposal_statuses": dict(proposal_statuses),
        "proposed_components": dict(proposed_components.most_common()),
        "rejected_components": dict(rejected_components.most_common()),
        "rejected_failure_classes": dict(failure_classes.most_common()),
        "rejected_proposal_count": len(rejected_proposals),
        **proposal_artifacts,
        "experiment_valid_for_effectiveness": not connection_blocked,
        "diagnosis": diagnose(
            improvement=improvement,
            connection_blocked=connection_blocked,
            rejected_proposals=rejected_proposals,
            proposed_components=proposed_components,
            failure_classes=failure_classes,
            boundary_failures=boundary_failures,
            proposal_artifacts=proposal_artifacts,
        ),
    }


def diagnose(
    *,
    improvement: float | None,
    connection_blocked: bool,
    rejected_proposals: list[dict[str, Any]],
    proposed_components: Counter[str],
    failure_classes: Counter[str],
    boundary_failures: Counter[str],
    proposal_artifacts: dict[str, int],
) -> list[str]:
    notes: list[str] = []
    if connection_blocked:
        notes.append(
            "The run is not valid for algorithm-effectiveness analysis: every rollout failed to connect to the local model."
        )
        notes.append("Run the same command from PyCharm or a non-sandboxed terminal, then analyze the new artifact dir.")
        return notes
    if improvement is None:
        notes.append("No baseline/best validation score pair was available; inspect result_summary.json.")
    elif improvement > 0:
        notes.append(f"Best candidate improved validation score by {improvement:.3f}.")
    elif improvement == 0:
        notes.append("Best candidate did not improve over baseline on validation.")
    else:
        notes.append(f"Best candidate regressed by {abs(improvement):.3f}; check accepted candidate lineage.")

    if rejected_proposals:
        notes.append(f"{len(rejected_proposals)} proposals were rejected at subsample acceptance.")
    if failure_classes:
        dominant_class, count = failure_classes.most_common(1)[0]
        notes.append(f"Dominant rejected failure class: {dominant_class} ({count}).")
    if boundary_failures:
        dominant_gate, count = boundary_failures.most_common(1)[0]
        notes.append(f"Boundary gate failures observed: {dominant_gate} ({count}).")
    if proposed_components:
        dominant_component, count = proposed_components.most_common(1)[0]
        notes.append(f"Most frequently selected component: {dominant_component} ({count}).")
    if proposal_artifacts.get("proposal_rationale_files", 0):
        notes.append(f"Proposal rationale files: {proposal_artifacts['proposal_rationale_files']}.")
    if proposal_artifacts.get("proposal_missing_rationale_files", 0):
        notes.append(
            f"Proposals with missing rationale markers: {proposal_artifacts['proposal_missing_rationale_files']}."
        )
    if proposal_artifacts.get("proposal_diff_files", 0):
        notes.append(f"Proposal diff files: {proposal_artifacts['proposal_diff_files']}.")
    if not rejected_proposals and improvement == 0:
        notes.append("If no proposal text was generated, inspect proposals/*/metadata.json for reflection/runtime failures.")
    return notes


def read_rollout_details(run_dir: Path, rollouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for row in rollouts:
        detail_file = row.get("detail_file")
        if not detail_file:
            continue
        detail = read_json(run_dir / "agent_logs" / str(detail_file), {})
        if isinstance(detail, dict):
            details.append(detail)
    return details


def proposal_artifact_counts(
    run_dir: Path,
    proposals: list[dict[str, Any]],
    rejected_proposals: list[dict[str, Any]],
) -> dict[str, int]:
    proposal_dirs = {run_dir / str(row.get("proposal_dir", "")) for row in proposals if row.get("proposal_dir")}
    rejected_dirs = {
        run_dir / str(row.get("proposal_dir", "")) for row in rejected_proposals if row.get("proposal_dir")
    }
    all_dirs = proposal_dirs | rejected_dirs
    return {
        "proposal_rationale_files": sum(1 for path in all_dirs if (path / "proposal_rationale.json").exists()),
        "proposal_missing_rationale_files": sum(
            1 for path in all_dirs if (path / "proposal_rationale_missing.json").exists()
        ),
        "proposal_diff_files": sum(
            1
            for path in all_dirs
            if (path / "diff_against_parent.patch").exists() or (path / "diff_against_seed.patch").exists()
        ),
    }


def rollout_error(row: dict[str, Any]) -> str:
    error = row.get("error")
    if error:
        return str(error)
    state = row.get("state")
    if isinstance(state, dict) and state.get("error"):
        return str(state["error"])
    return ""


def print_report(summary: dict[str, Any]) -> None:
    print(f"Run: {summary['run_dir']}")
    print(f"Best val score: {summary['best_val_score']}")
    print(f"Baseline val score: {summary['baseline_val_score']}")
    print(f"Improvement: {summary['improvement']}")
    print(f"Metric calls: {summary['total_metric_calls']}")
    print(f"Candidates: {summary['num_candidates']}")
    print(f"Rollouts: {summary['rollout_count']}")
    print(f"Runtime errors: {summary['runtime_errors']}")
    print(f"Boundary failures: {summary['boundary_failures']}")
    print(f"Proposal statuses: {summary['proposal_statuses']}")
    print(f"Rejected proposals: {summary['rejected_proposal_count']}")
    print(f"Proposal rationale files: {summary['proposal_rationale_files']}")
    print(f"Proposal missing rationale files: {summary['proposal_missing_rationale_files']}")
    print(f"Proposal diff files: {summary['proposal_diff_files']}")
    print("Diagnosis:")
    for note in summary["diagnosis"]:
        print(f"- {note}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", help="Run artifact directory. Defaults to runs/latest_run.txt.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a human-readable report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_run(resolve_run_dir(args.run_dir))
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print_report(summary)


if __name__ == "__main__":
    main()
