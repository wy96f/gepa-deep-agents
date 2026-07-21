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
from collections.abc import Callable
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
    proposal_events = read_jsonl(run_dir / "proposals" / "index.jsonl")
    rejected_proposal_events = read_jsonl(run_dir / "rejected_proposals" / "index.jsonl")
    reflection_errors = read_jsonl(run_dir / "reflection_errors" / "index.jsonl")
    proposal_reviews = read_jsonl(run_dir / "proposal_reviews" / "index.jsonl")
    proposals = latest_proposal_events(proposal_events)
    rejected_proposals = latest_proposal_events(rejected_proposal_events)
    optimization_rollouts = [
        row for row in rollouts if not str(row.get("evaluation_phase", "optimization")).startswith("final_test")
    ]
    rollout_details = read_rollout_details(run_dir, optimization_rollouts)

    scores = [float(row.get("score", 0.0)) for row in optimization_rollouts]
    errors = Counter(error for row in optimization_rollouts if (error := rollout_error(row)))
    boundary_failures: Counter[str] = Counter(
        str(constraint["name"])
        for detail in rollout_details
        for constraint in detail.get("constraints", [])
        if isinstance(constraint, dict)
        and constraint.get("name")
        and not constraint.get("passed", True)
        and ":boundary:" in str(constraint.get("name", ""))
    )
    hard_constraint_failures: Counter[str] = Counter(
        str(constraint["name"])
        for detail in rollout_details
        for constraint in detail.get("constraints", [])
        if isinstance(constraint, dict)
        and constraint.get("name")
        and not constraint.get("passed", True)
        and str(constraint.get("severity", "hard")) == "hard"
    )
    unloadable_skill_failures = Counter(
        {
            name: count
            for name, count in hard_constraint_failures.items()
            if str(name).endswith((":frontmatter", ":frontmatter_yaml", ":name_description"))
        }
    )
    runtime_skipped_count = sum(1 for detail in rollout_details if bool(detail.get("candidate_runtime_skipped", False)))
    tool_capability_gaps = Counter(
        gap for detail in rollout_details for gap in detail.get("fitness", {}).get("tool_capability_gaps", [])
    )
    tool_capability_gap_unique_inputs = unique_input_counts(
        rollout_details,
        lambda detail: detail.get("fitness", {}).get("tool_capability_gaps", []),
    )
    missed_supported_expectations = Counter(
        gap
        for detail in rollout_details
        for gap in detail.get("fitness", {}).get("tool_supported_missing_expectations", [])
    )
    missed_supported_expectation_unique_inputs = unique_input_counts(
        rollout_details,
        lambda detail: detail.get("fitness", {}).get("tool_supported_missing_expectations", []),
    )
    missing_trace_expectations = Counter(
        gap for detail in rollout_details for gap in detail.get("fitness", {}).get("missing_trace_expectations", [])
    )
    failed_tool_expectations = Counter(
        gap for detail in rollout_details for gap in detail.get("fitness", {}).get("failed_tool_expectations", [])
    )
    incomplete_tool_result_expectations = Counter(
        gap
        for detail in rollout_details
        for gap in detail.get("fitness", {}).get("incomplete_tool_result_expectations", [])
    )
    runtime_supported_missing_checkpoints = Counter(
        item
        for detail in rollout_details
        for item in detail.get("fitness", {}).get("runtime_supported_missing_checkpoints", [])
    )
    tool_actionable_missing_checkpoints = Counter(
        item
        for detail in rollout_details
        for item in detail.get("fitness", {}).get("tool_actionable_missing_checkpoints", [])
    )
    tool_blocked_missing_checkpoints = Counter(
        item
        for detail in rollout_details
        for item in detail.get("fitness", {}).get("tool_blocked_missing_checkpoints", [])
    )
    unmapped_missing_checkpoints = Counter(
        item for detail in rollout_details for item in detail.get("fitness", {}).get("unmapped_missing_checkpoints", [])
    )
    remediation_types = Counter(
        str(action.get("type"))
        for detail in rollout_details
        for action in detail.get("fitness", {}).get("remediation_actions", [])
        if action.get("type") and action.get("type") != "NO_ACTION"
    )
    remediation_owners = Counter(
        str(action.get("owner"))
        for detail in rollout_details
        for action in detail.get("fitness", {}).get("remediation_actions", [])
        if action.get("owner") and action.get("type") != "NO_ACTION"
    )
    proposal_statuses = Counter(str(row.get("status", "unknown")) for row in proposals)
    proposed_components = Counter(component for row in proposals for component in row.get("components", []))
    rejected_components = Counter(component for row in rejected_proposals for component in row.get("components", []))
    failure_classes = Counter(
        failure_class
        for row in rejected_proposals
        for failure_class in row.get("failure_classifications", [])
        if failure_class != "NO_FAILURE"
    )
    rollout_failure_classes = Counter(
        str(row.get("failure_classification")) for row in optimization_rollouts if row.get("failure_classification")
    )
    mutation_ineligible_count = sum(
        1 for detail in rollout_details if detail.get("fitness", {}).get("mutation_eligible") is False
    )
    deterministic_judge_disagreement_count = sum(
        1
        for detail in rollout_details
        if detail.get("expected")
        and float(detail.get("fitness", {}).get("hard", 0.0)) >= 1.0
        and float(detail.get("fitness", {}).get("judge_score", 1.0)) < 1.0
    )
    no_actionable_proposal_count = sum(1 for row in proposals if not row.get("components"))
    proposal_review_decisions = Counter(str(row.get("decision", "unknown")) for row in proposal_reviews)
    proposal_artifacts = proposal_artifact_counts(run_dir, proposals, rejected_proposals)
    final_test = summary.get("final_test") or read_json(run_dir / "final_test" / "summary.json")

    val_scores = summary.get("val_aggregate_scores") or []
    best_val_score = summary.get("best_val_score")
    baseline_score = val_scores[0] if val_scores else None
    improvement = None
    if baseline_score is not None and best_val_score is not None:
        improvement = float(best_val_score) - float(baseline_score)

    runtime_blocked = bool(optimization_rollouts) and sum(errors.values()) == len(optimization_rollouts)
    connection_blocked = runtime_blocked and all("APIConnectionError" in key or "ConnectError" in key for key in errors)

    return {
        "run_dir": str(run_dir),
        "best_val_score": best_val_score,
        "baseline_val_score": baseline_score,
        "improvement": improvement,
        "best_idx": summary.get("best_idx"),
        "gepa_best_idx": summary.get("gepa_best_idx", summary.get("best_idx")),
        "tie_break_applied": bool(summary.get("tie_break_applied", False)),
        "selection_policy": summary.get("selection_policy", "unknown"),
        "tied_best_indices": summary.get("tied_best_indices", []),
        "num_candidates": summary.get("num_candidates"),
        "total_metric_calls": summary.get("total_metric_calls"),
        "num_full_val_evals": summary.get("num_full_val_evals"),
        "rollout_count": len(optimization_rollouts),
        "final_test_rollout_count": len(rollouts) - len(optimization_rollouts),
        "rollout_score_mean": sum(scores) / len(scores) if scores else None,
        "rollout_score_min": min(scores) if scores else None,
        "rollout_score_max": max(scores) if scores else None,
        "runtime_errors": dict(errors),
        "boundary_failures": dict(boundary_failures.most_common()),
        "hard_constraint_failures": dict(hard_constraint_failures.most_common()),
        "unloadable_skill_failures": dict(unloadable_skill_failures.most_common()),
        "candidate_runtime_skipped_count": runtime_skipped_count,
        "tool_capability_gaps": dict(tool_capability_gaps.most_common()),
        "tool_capability_gap_unique_inputs": dict(tool_capability_gap_unique_inputs.most_common()),
        "missed_supported_expectations": dict(missed_supported_expectations.most_common()),
        "missed_supported_expectation_unique_inputs": dict(missed_supported_expectation_unique_inputs.most_common()),
        "missing_trace_expectations": dict(missing_trace_expectations.most_common()),
        "failed_tool_expectations": dict(failed_tool_expectations.most_common()),
        "incomplete_tool_result_expectations": dict(incomplete_tool_result_expectations.most_common()),
        "runtime_supported_missing_checkpoints": dict(runtime_supported_missing_checkpoints.most_common()),
        "tool_actionable_missing_checkpoints": dict(tool_actionable_missing_checkpoints.most_common()),
        "tool_blocked_missing_checkpoints": dict(tool_blocked_missing_checkpoints.most_common()),
        "unmapped_missing_checkpoints": dict(unmapped_missing_checkpoints.most_common()),
        "remediation_types": dict(remediation_types.most_common()),
        "remediation_owners": dict(remediation_owners.most_common()),
        "proposal_statuses": dict(proposal_statuses),
        "proposal_count": len(proposals),
        "proposal_event_count": len(proposal_events),
        "proposed_components": dict(proposed_components.most_common()),
        "rejected_components": dict(rejected_components.most_common()),
        "rejected_failure_classes": dict(failure_classes.most_common()),
        "rollout_failure_classes": dict(rollout_failure_classes.most_common()),
        "mutation_ineligible_count": mutation_ineligible_count,
        "deterministic_judge_disagreement_count": deterministic_judge_disagreement_count,
        "no_actionable_proposal_count": no_actionable_proposal_count,
        "rejected_proposal_count": len(rejected_proposals),
        "rejected_proposal_event_count": len(rejected_proposal_events),
        "reflection_error_count": len(reflection_errors),
        "reflection_errors": reflection_errors,
        "proposal_review_count": len(proposal_reviews),
        "proposal_review_decisions": dict(proposal_review_decisions),
        "final_test": final_test,
        **proposal_artifacts,
        "experiment_valid_for_effectiveness": not connection_blocked,
        "diagnosis": diagnose(
            improvement=improvement,
            connection_blocked=connection_blocked,
            rejected_proposals=rejected_proposals,
            reflection_errors=reflection_errors,
            proposal_review_decisions=proposal_review_decisions,
            proposal_statuses=proposal_statuses,
            proposed_components=proposed_components,
            failure_classes=failure_classes,
            rollout_failure_classes=rollout_failure_classes,
            boundary_failures=boundary_failures,
            unloadable_skill_failures=unloadable_skill_failures,
            runtime_skipped_count=runtime_skipped_count,
            deterministic_judge_disagreement_count=deterministic_judge_disagreement_count,
            no_actionable_proposal_count=no_actionable_proposal_count,
            tool_capability_gaps=tool_capability_gaps,
            tool_capability_gap_unique_inputs=tool_capability_gap_unique_inputs,
            missed_supported_expectations=missed_supported_expectations,
            missed_supported_expectation_unique_inputs=missed_supported_expectation_unique_inputs,
            failed_tool_expectations=failed_tool_expectations,
            incomplete_tool_result_expectations=incomplete_tool_result_expectations,
            remediation_types=remediation_types,
            proposal_artifacts=proposal_artifacts,
            selection_policy=str(summary.get("selection_policy", "unknown")),
            final_test=final_test if isinstance(final_test, dict) else None,
            val_scores=[float(score) for score in val_scores],
            tied_best_indices=list(summary.get("tied_best_indices", [])),
        ),
    }


def latest_proposal_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse the proposal event stream to one terminal/latest row per iteration."""
    latest: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for index, row in enumerate(rows):
        identity = str(row.get("iteration") if row.get("iteration") is not None else row.get("proposal_dir") or index)
        if identity not in latest:
            order.append(identity)
        latest[identity] = row
    return [latest[identity] for identity in order]


def unique_input_counts(
    details: list[dict[str, Any]],
    values: Callable[[dict[str, Any]], Any],
) -> Counter[str]:
    inputs_by_value: dict[str, set[str]] = {}
    for index, detail in enumerate(details):
        input_id = str(detail.get("input") or f"<rollout:{index}>")
        for value in values(detail) or []:
            inputs_by_value.setdefault(str(value), set()).add(input_id)
    return Counter({value: len(inputs) for value, inputs in inputs_by_value.items()})


def diagnose(
    *,
    improvement: float | None,
    connection_blocked: bool,
    rejected_proposals: list[dict[str, Any]],
    reflection_errors: list[dict[str, Any]],
    proposal_review_decisions: Counter[str],
    proposal_statuses: Counter[str],
    proposed_components: Counter[str],
    failure_classes: Counter[str],
    rollout_failure_classes: Counter[str],
    boundary_failures: Counter[str],
    unloadable_skill_failures: Counter[str],
    runtime_skipped_count: int,
    deterministic_judge_disagreement_count: int,
    no_actionable_proposal_count: int,
    tool_capability_gaps: Counter[str],
    tool_capability_gap_unique_inputs: Counter[str],
    missed_supported_expectations: Counter[str],
    missed_supported_expectation_unique_inputs: Counter[str],
    failed_tool_expectations: Counter[str],
    incomplete_tool_result_expectations: Counter[str],
    remediation_types: Counter[str],
    proposal_artifacts: dict[str, int],
    selection_policy: str,
    final_test: dict[str, Any] | None,
    val_scores: list[float],
    tied_best_indices: list[int],
) -> list[str]:
    notes: list[str] = []
    if connection_blocked:
        notes.append(
            "The run is not valid for algorithm-effectiveness analysis: every rollout failed to connect to the local model."
        )
        notes.append(
            "Run the same command from PyCharm or a non-sandboxed terminal, then analyze the new artifact dir."
        )
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
    if reflection_errors:
        dominant_error, count = Counter(
            str(item.get("error_type", "unknown")) for item in reflection_errors
        ).most_common(1)[0]
        notes.append(
            f"{len(reflection_errors)} reflection calls failed before proposal generation; "
            f"dominant error: {dominant_error} ({count}). Inspect reflection_errors/index.jsonl."
        )
    elif proposal_statuses.get("started", 0):
        notes.append(
            f"{proposal_statuses['started']} proposals stopped after reflection started without producing candidate "
            "text. This artifact predates reflection error capture; inspect the external debug log for the provider "
            "exception."
        )
    if proposal_review_decisions:
        decisions = ", ".join(f"{name}={count}" for name, count in proposal_review_decisions.most_common())
        notes.append(f"Pre-runtime proposal reviews: {decisions}.")
    if failure_classes:
        dominant_class, count = failure_classes.most_common(1)[0]
        notes.append(f"Dominant rejected failure class: {dominant_class} ({count}).")
    if rollout_failure_classes.get("INSUFFICIENT_RUNTIME_EVIDENCE", 0):
        count = rollout_failure_classes["INSUFFICIENT_RUNTIME_EVIDENCE"]
        notes.append(
            f"{count} rollout evaluations lacked a defensible link between evaluator-only facts and runtime evidence; "
            "they were excluded from text-component voting."
        )
    if boundary_failures:
        dominant_gate, count = boundary_failures.most_common(1)[0]
        notes.append(f"Boundary gate failures observed: {dominant_gate} ({count}).")
    if unloadable_skill_failures:
        dominant_gate, count = unloadable_skill_failures.most_common(1)[0]
        notes.append(
            f"Runtime-unloadable SKILL.md candidate observed: {dominant_gate} ({count}). "
            "Reject this candidate before creating the Deep Agent."
        )
    if runtime_skipped_count:
        notes.append(
            f"{runtime_skipped_count} candidate rollouts were skipped before agent execution because a critical "
            "deterministic constraint failed."
        )
    if deterministic_judge_disagreement_count:
        notes.append(
            f"{deterministic_judge_disagreement_count} rollout evaluations had an exact authoritative target match "
            "but a lower LLM-judge score. Treat deterministic correctness as the score anchor for these tasks."
        )
    if no_actionable_proposal_count:
        notes.append(
            f"{no_actionable_proposal_count} proposal iterations had no actionable text component. Their feedback is "
            "diagnostic evidence for a tool/data backlog, not a text mutation."
        )
    if tool_capability_gaps:
        dominant_gap, rollout_count = tool_capability_gaps.most_common(1)[0]
        unique_count = tool_capability_gap_unique_inputs.get(dominant_gap, 0)
        notes.append(
            f"External tool capability gap likely blocks improvement: {dominant_gap} "
            f"({unique_count} unique inputs, {rollout_count} rollout occurrences). "
            "Implement or connect a tool for this data source before expecting GEPA text edits to fix it."
        )
    if missed_supported_expectations:
        dominant_miss, rollout_count = missed_supported_expectations.most_common(1)[0]
        unique_count = missed_supported_expectation_unique_inputs.get(dominant_miss, 0)
        notes.append(
            f"Agent skipped an apparently available data-acquisition path: {dominant_miss} "
            f"({unique_count} unique inputs, {rollout_count} rollout occurrences); "
            "optimize skill/prompt/tool descriptions to call the existing tool more reliably."
        )
    if failed_tool_expectations:
        dominant_failure, count = failed_tool_expectations.most_common(1)[0]
        notes.append(
            f"A matching tool call failed for: {dominant_failure} ({count}). Inspect its arguments and result; "
            "fix invocation guidance only for argument errors, otherwise repair the tool or runtime."
        )
    if incomplete_tool_result_expectations:
        dominant_result, count = incomplete_tool_result_expectations.most_common(1)[0]
        notes.append(
            f"A matching tool returned without usable expected evidence: {dominant_result} ({count}). "
            "Review query construction, returned fields, and the advertised tool capability."
        )
    if remediation_types:
        dominant_remediation, count = remediation_types.most_common(1)[0]
        notes.append(f"Dominant recommended remediation: {dominant_remediation} ({count}).")
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
    if final_test is not None and float(final_test.get("improvement", 0.0)) < 0:
        notes.append(
            f"Held-out test regressed by {abs(float(final_test['improvement'])):.3f}. "
            "The test set is diagnostic only and must not select a candidate, but this is strong evidence against "
            "deploying a validation-tied proposal."
        )
    if final_test is not None:
        positive_diagnostics = [
            item for item in final_test.get("diagnostic_candidates", []) if float(item.get("delta_vs_seed", 0.0)) > 0.0
        ]
        if positive_diagnostics:
            strongest = max(positive_diagnostics, key=lambda item: float(item.get("delta_vs_seed", 0.0)))
            notes.append(
                f"Validation-tied candidate {strongest.get('candidate_idx')} improved held-out diagnostic mean by "
                f"{float(strongest.get('delta_vs_seed', 0.0)):.3f}. Do not select it using test data; broaden or "
                "stratify validation so the repaired behavior is represented before deployment."
            )
    if proposal_statuses.get("accepted", 0) and improvement == 0:
        if len(tied_best_indices) > 1 and selection_policy == "latest_accepted_on_validation_tie":
            notes.append(
                "An accepted candidate tied the baseline validation score and the legacy policy deployed the newest "
                "candidate. A tie is not improvement evidence; retain the incumbent and keep the proposal as an "
                "artifact for review."
            )
        elif len(tied_best_indices) > 1:
            notes.append(
                "An accepted candidate tied the baseline validation score. The incumbent remains the deployment "
                "candidate; inspect the proposal artifact and per-example deltas before broadening validation data."
            )
        elif len(val_scores) > 1:
            notes.append(
                "A proposal entered the candidate pool after subsample/Pareto acceptance but did not improve aggregate "
                "full validation. It remains useful as an artifact, not as the deployment candidate."
            )
    elif not rejected_proposals and not proposal_statuses:
        notes.append(
            "If no proposal text was generated, inspect proposals/*/metadata.json for reflection/runtime failures."
        )
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
    dirs_by_iteration: dict[str, Path] = {}
    for index, row in enumerate([*proposals, *rejected_proposals]):
        proposal_dir = row.get("proposal_dir")
        if not proposal_dir:
            continue
        identity = str(row.get("iteration") if row.get("iteration") is not None else index)
        candidate_dir = run_dir / artifact_relative_path(proposal_dir)
        current = dirs_by_iteration.get(identity)
        if current is None or str(candidate_dir).replace("\\", "/").find("/proposals/") >= 0:
            dirs_by_iteration[identity] = candidate_dir
    all_dirs = set(dirs_by_iteration.values())
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


def artifact_relative_path(value: Any) -> Path:
    return Path(str(value).replace("\\", "/"))


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
    print(
        "Candidate selection: "
        f"deployment={summary['best_idx']} gepa={summary['gepa_best_idx']} "
        f"tie_break={summary['tie_break_applied']} policy={summary['selection_policy']}"
    )
    print(f"Metric calls: {summary['total_metric_calls']}")
    print(f"Candidates: {summary['num_candidates']}")
    print(f"Rollouts: {summary['rollout_count']}")
    print(f"Final test rollouts: {summary['final_test_rollout_count']}")
    print(f"Runtime errors: {summary['runtime_errors']}")
    print(f"Boundary failures: {summary['boundary_failures']}")
    print(f"Hard constraint failures: {summary['hard_constraint_failures']}")
    print(f"Unloadable skill failures: {summary['unloadable_skill_failures']}")
    print(f"Runtime-skipped candidates: {summary['candidate_runtime_skipped_count']}")
    print(f"Tool capability gaps: {summary['tool_capability_gaps']}")
    print(f"Tool capability gaps by unique input: {summary['tool_capability_gap_unique_inputs']}")
    print(f"Missed supported expectations: {summary['missed_supported_expectations']}")
    print(f"Missed supported expectations by unique input: {summary['missed_supported_expectation_unique_inputs']}")
    print(f"Missing trace expectations: {summary['missing_trace_expectations']}")
    print(f"Failed tool expectations: {summary['failed_tool_expectations']}")
    print(f"Incomplete tool results: {summary['incomplete_tool_result_expectations']}")
    print(f"Remediation types: {summary['remediation_types']}")
    print(f"Remediation owners: {summary['remediation_owners']}")
    print(f"Mutation-ineligible rollouts: {summary['mutation_ineligible_count']}")
    print(f"Deterministic/judge disagreements: {summary['deterministic_judge_disagreement_count']}")
    print(f"No-actionable proposals: {summary['no_actionable_proposal_count']}")
    print(f"Proposal statuses: {summary['proposal_statuses']}")
    print(f"Proposals: {summary['proposal_count']} ({summary['proposal_event_count']} lifecycle events)")
    print(f"Rejected proposals: {summary['rejected_proposal_count']}")
    print(f"Reflection errors: {summary['reflection_error_count']}")
    print(f"Proposal reviews: {summary['proposal_review_count']} {summary['proposal_review_decisions']}")
    print(f"Proposal rationale files: {summary['proposal_rationale_files']}")
    print(f"Proposal missing rationale files: {summary['proposal_missing_rationale_files']}")
    print(f"Proposal diff files: {summary['proposal_diff_files']}")
    if summary.get("final_test"):
        final_test = summary["final_test"]
        print(
            "Final test: "
            f"seed={final_test.get('seed_mean')} best={final_test.get('best_mean')} "
            f"improvement={final_test.get('improvement')}"
        )
        if final_test.get("diagnostic_candidates"):
            print(f"Final test diagnostic candidates: {final_test['diagnostic_candidates']}")
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
