"""Artifact persistence for Deep Agents GEPA runs."""

from __future__ import annotations

import difflib
import json
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from examples.langchain_adapter.deepagents_gepa.framework import component_consumption


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _safe_name(name: str, limit: int = 180) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.=-]+", "__", name).strip("._")
    return (safe or "component")[:limit]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(_jsonable(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _message_to_dict(message: Any) -> dict[str, Any]:
    content = getattr(message, "content", message)
    return {
        "type": type(message).__name__,
        "name": getattr(message, "name", None),
        "tool_call_id": getattr(message, "tool_call_id", None),
        "status": getattr(message, "status", None),
        "content": content,
        "tool_calls": getattr(message, "tool_calls", None) or [],
        "additional_kwargs": getattr(message, "additional_kwargs", {}) or {},
    }


def _state_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    messages = state.get("messages") or []
    return {
        "messages": [_message_to_dict(message) for message in messages],
        "error": repr(state.get("error")) if state.get("error") is not None else None,
        "candidate_hash": state.get("candidate_hash"),
        "candidate_constraints": state.get("candidate_constraints", []),
        "candidate_metrics": state.get("candidate_metrics", {}),
        "fitness": state.get("fitness", {}),
        "judge_diagnostics": state.get("judge_diagnostics", {}),
        "available_tools": state.get("available_tools", []),
        "capability_tools": state.get("capability_tools", []),
        "trace_context_window_tokens": state.get("trace_context_window_tokens"),
        "trace_context_ratio": state.get("trace_context_ratio"),
        "evaluation_trace_mode": state.get("evaluation_trace_mode"),
        "evaluation_phase": state.get("evaluation_phase", "optimization"),
        "candidate_runtime_skipped": bool(state.get("candidate_runtime_skipped", False)),
        "candidate_runtime_skip_reason": state.get("candidate_runtime_skip_reason"),
    }


def _last_message_text(state: Mapping[str, Any]) -> str:
    messages = state.get("messages") or []
    if not messages:
        return ""
    content = getattr(messages[-1], "content", messages[-1])
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def _write_component_texts(base_dir: Path, candidate: Mapping[str, str]) -> list[dict[str, Any]]:
    manifest = []
    for key, text in candidate.items():
        filename = f"{_safe_name(key)}.txt"
        component_path = base_dir / "components" / filename
        component_path.parent.mkdir(parents=True, exist_ok=True)
        component_path.write_text(text, encoding="utf-8")
        manifest.append({"key": key, "file": f"components/{filename}", "chars": len(text)})
    return manifest


def _candidate_diff(
    candidate: Mapping[str, str],
    baseline: Mapping[str, str],
    *,
    baseline_label: str,
    candidate_label: str = "candidate",
) -> str:
    chunks: list[str] = []
    for key in sorted(set(candidate) | set(baseline)):
        old_text = str(baseline.get(key, ""))
        new_text = str(candidate.get(key, ""))
        if old_text == new_text:
            continue
        chunks.extend(
            difflib.unified_diff(
                old_text.splitlines(),
                new_text.splitlines(),
                fromfile=f"{baseline_label}/{key}",
                tofile=f"{candidate_label}/{key}",
                lineterm="",
            )
        )
        chunks.append("")
    return "\n".join(chunks).rstrip() + ("\n" if chunks else "")


def _material_new_instructions(
    parent_candidate: Mapping[str, str],
    new_instructions: Mapping[str, Any],
) -> dict[str, str]:
    """Keep only proposal values that materially change their parent text."""
    return {
        str(component): str(replacement)
        for component, replacement in new_instructions.items()
        if str(parent_candidate.get(str(component), "")).rstrip() != str(replacement).rstrip()
    }


def _write_candidate_diffs(
    base_dir: Path,
    candidate: Mapping[str, str],
    *,
    parent_candidate: Mapping[str, str] | None = None,
    seed_candidate: Mapping[str, str] | None = None,
) -> None:
    diff_dir = base_dir / "diffs"
    if parent_candidate:
        parent_diff = _candidate_diff(candidate, parent_candidate, baseline_label="parent")
        if parent_diff:
            diff_dir.mkdir(parents=True, exist_ok=True)
            (base_dir / "diff_against_parent.patch").write_text(parent_diff, encoding="utf-8")
            (diff_dir / "all__against_parent.patch").write_text(parent_diff, encoding="utf-8")
            _write_component_level_diffs(
                diff_dir,
                candidate,
                parent_candidate,
                baseline_label="parent",
                suffix="against_parent",
            )
    if seed_candidate:
        seed_diff = _candidate_diff(candidate, seed_candidate, baseline_label="seed")
        if seed_diff:
            diff_dir.mkdir(parents=True, exist_ok=True)
            (base_dir / "diff_against_seed.patch").write_text(seed_diff, encoding="utf-8")
            (diff_dir / "all__against_seed.patch").write_text(seed_diff, encoding="utf-8")
            _write_component_level_diffs(
                diff_dir,
                candidate,
                seed_candidate,
                baseline_label="seed",
                suffix="against_seed",
            )


def _write_component_level_diffs(
    diff_dir: Path,
    candidate: Mapping[str, str],
    baseline: Mapping[str, str],
    *,
    baseline_label: str,
    suffix: str,
) -> None:
    for key in sorted(set(candidate) | set(baseline)):
        old_text = str(baseline.get(key, ""))
        new_text = str(candidate.get(key, ""))
        if old_text == new_text:
            continue
        diff = "\n".join(
            difflib.unified_diff(
                old_text.splitlines(),
                new_text.splitlines(),
                fromfile=f"{baseline_label}/{key}",
                tofile=f"candidate/{key}",
                lineterm="",
            )
        )
        if diff:
            (diff_dir / f"{_safe_name(key)}__{suffix}.patch").write_text(diff + "\n", encoding="utf-8")


def _extract_proposal_rationale(raw_output: str) -> str:
    """Return the explicit proposal rationale before the final fenced replacement."""
    first_fence = raw_output.find("```")
    rationale = raw_output[:first_fence] if first_fence >= 0 else raw_output
    rationale = rationale.strip()
    if not rationale:
        return ""
    return rationale


def _write_proposal_rationales(base_dir: Path, raw_lm_outputs: Mapping[str, Any]) -> dict[str, str]:
    rationales: dict[str, str] = {}
    rationale_dir = base_dir / "proposal_rationale"
    for component, raw_output in raw_lm_outputs.items():
        rationale = _extract_proposal_rationale(str(raw_output))
        if not rationale:
            continue
        rationales[str(component)] = rationale
        rationale_dir.mkdir(parents=True, exist_ok=True)
        (rationale_dir / f"{_safe_name(str(component))}.txt").write_text(rationale, encoding="utf-8")
    if rationales:
        _write_json(base_dir / "proposal_rationale.json", rationales)
    return rationales


def _missing_rationale_components(raw_lm_outputs: Mapping[str, Any], rationales: Mapping[str, str]) -> list[str]:
    return sorted(str(component) for component in raw_lm_outputs if str(component) not in rationales)


def _write_missing_rationales(base_dir: Path, missing_rationales: Sequence[str]) -> None:
    if missing_rationales:
        _write_json(base_dir / "proposal_rationale_missing.json", list(missing_rationales))


def _failure_classes_from_trajectories(trajectories: Sequence[Any] | None) -> list[str]:
    classes: list[str] = []
    for trajectory in trajectories or []:
        if not isinstance(trajectory, Mapping):
            continue
        feedback = str(trajectory.get("feedback", ""))
        match = re.search(r"(?m)^-\s*failure_classification:\s*(\S+)", feedback)
        if match:
            classes.append(match.group(1))
    return classes


def _evaluation_payload(examples: Sequence[Mapping[str, Any]], evaluation: Any) -> dict[str, Any]:
    scores = [float(score) for score in list(getattr(evaluation, "scores", []) or [])]
    outputs = list(getattr(evaluation, "outputs", []) or [])
    rows = []
    for index, example in enumerate(examples):
        output = outputs[index] if index < len(outputs) else None
        state = output.get("state", {}) if isinstance(output, Mapping) else {}
        rows.append(
            {
                "input": example.get("input"),
                "score": scores[index] if index < len(scores) else None,
                "response": output.get("response") if isinstance(output, Mapping) else None,
                "fitness": state.get("fitness", {}) if isinstance(state, Mapping) else {},
                "candidate_hash": state.get("candidate_hash") if isinstance(state, Mapping) else None,
            }
        )
    return {"scores": scores, "mean": _score_mean(scores), "rows": rows}


def _score_mean(scores: Sequence[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0


def _metric_calls(evaluation: Any, default: int) -> int:
    calls = getattr(evaluation, "num_metric_calls", None)
    return default if calls is None else int(calls)


def _without_shared_rubric(
    examples: Sequence[Mapping[str, Any]],
    shared_rubric: str | None,
) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        row = dict(example)
        if shared_rubric and row.get("rubric") == shared_rubric:
            row.pop("rubric", None)
        rows.append(row)
    return rows


def _evaluation_cohort_rows(examples: Sequence[Mapping[str, Any]], evaluation: Any) -> dict[str, Any]:
    outputs = list(getattr(evaluation, "outputs", []) or [])
    scores = [float(score) for score in list(getattr(evaluation, "scores", []) or [])]
    cohorts: dict[str, list[dict[str, Any]]] = {}
    for index, example in enumerate(examples):
        output = outputs[index] if index < len(outputs) else None
        state = output.get("state", {}) if isinstance(output, Mapping) else {}
        fitness = state.get("fitness", {}) if isinstance(state, Mapping) else {}
        classification = str(fitness.get("failure_classification") or "UNKNOWN")
        if bool(fitness.get("mutation_eligible", False)):
            cohort = "text_actionable"
        elif classification in {"TOOL_CAPABILITY_GAP", "INSUFFICIENT_RUNTIME_EVIDENCE"}:
            cohort = "tool_blocked"
        elif classification == "NO_FAILURE":
            cohort = "satisfied"
        else:
            cohort = "other_diagnostic"
        cohorts.setdefault(cohort, []).append(
            {
                "input": example.get("input"),
                "score": scores[index] if index < len(scores) else None,
                "failure_classification": classification,
            }
        )
    return {
        name: {
            "count": len(rows),
            "mean": _score_mean([float(row["score"]) for row in rows if row["score"] is not None]),
            "rows": rows,
        }
        for name, rows in cohorts.items()
    }


class RunArtifactStore:
    """Persist run inputs, candidates, and final materialized artifacts."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._rollout_counter = 0
        self._reflection_error_counter = 0
        self._proposal_review_counter = 0
        self._seed_candidate: dict[str, str] = {}
        self._preflight_summary: dict[str, Any] | None = None
        self._budget_plan: dict[str, Any] | None = None
        self._final_test_metric_calls = 0
        self._shared_rubric: str | None = None

    @classmethod
    def create(cls, base_dir: str | Path, run_name: str | None = None) -> RunArtifactStore:
        base = Path(base_dir)
        name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        store = cls(base / name)
        base.mkdir(parents=True, exist_ok=True)
        (base / "latest_run.txt").write_text(str(store.run_dir.resolve()) + "\n", encoding="utf-8")
        return store

    def write_run_inputs(
        self,
        *,
        config_path: str | Path,
        config: Any,
        project: Any,
        train_set: Sequence[Mapping[str, Any]],
        val_set: Sequence[Mapping[str, Any]],
        test_set: Sequence[Mapping[str, Any]],
    ) -> None:
        config_path = Path(config_path)
        if config_path.exists():
            config_copy = self.run_dir / "config" / config_path.name
            config_copy.parent.mkdir(parents=True, exist_ok=True)
            config_copy.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        _write_json(self.run_dir / "config" / "resolved_config.json", config)
        _write_json(self.run_dir / "project" / "surface_manifest.json", getattr(project, "surfaces", {}))
        self._seed_candidate = dict(getattr(project, "candidate", {}) or {})
        _write_json(self.run_dir / "project" / "seed_candidate_keys.json", list(self._seed_candidate))
        shared_rubric = getattr(getattr(config, "dataset", None), "rubric", None)
        self._shared_rubric = str(shared_rubric) if shared_rubric else None
        if shared_rubric:
            rubric_path = self.run_dir / "datasets" / "rubric.md"
            rubric_path.parent.mkdir(parents=True, exist_ok=True)
            rubric_path.write_text(str(shared_rubric).rstrip() + "\n", encoding="utf-8")
        _write_jsonl(self.run_dir / "datasets" / "train.jsonl", _without_shared_rubric(train_set, shared_rubric))
        _write_jsonl(self.run_dir / "datasets" / "val.jsonl", _without_shared_rubric(val_set, shared_rubric))
        _write_jsonl(self.run_dir / "datasets" / "test.jsonl", _without_shared_rubric(test_set, shared_rubric))

    def write_budget_plan(self, plan: Mapping[str, Any]) -> None:
        self._budget_plan = dict(plan)
        _write_json(self.run_dir / "diagnostics" / "metric_budget_plan.json", self._budget_plan)

    def write_actionability_preflight(
        self,
        *,
        examples: Sequence[Mapping[str, Any]],
        evaluation: Any,
        partition: Any,
        optimization_examples: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        outputs = list(getattr(evaluation, "outputs", []) or [])
        trajectories = list(getattr(evaluation, "trajectories", []) or [])
        scores = [float(score) for score in list(getattr(evaluation, "scores", []) or [])]
        rows = []
        for index, example in enumerate(examples):
            output = outputs[index] if index < len(outputs) else None
            state = output.get("state", {}) if isinstance(output, Mapping) else {}
            fitness = state.get("fitness", {}) if isinstance(state, Mapping) else {}
            feedback = str(trajectories[index].get("feedback", "")) if index < len(trajectories) else ""
            component_match = re.search(r"(?m)^-\s*suggested_component:\s*(\S+)", feedback)
            rows.append(
                {
                    "index": index,
                    "input": example.get("input"),
                    "score": scores[index] if index < len(scores) else None,
                    "mutation_eligible": bool(fitness.get("mutation_eligible", False)),
                    "failure_classification": fitness.get("failure_classification"),
                    "suggested_component": component_match.group(1) if component_match else None,
                    "tool_capability_gaps": fitness.get("tool_capability_gaps", []),
                    "tool_data_coverage_gaps": fitness.get("tool_data_coverage_gaps", []),
                    "runtime_supported_missing_checkpoints": fitness.get("runtime_supported_missing_checkpoints", []),
                    "runtime_read_paths": fitness.get("runtime_read_paths", []),
                    "consumed_skill_components": fitness.get("consumed_skill_components", []),
                    "consumed_reference_components": fitness.get("consumed_reference_components", []),
                    "configured_skill_unread": bool(fitness.get("configured_skill_unread", False)),
                    "configured_resources_unread": bool(fitness.get("configured_resources_unread", False)),
                }
            )
        summary = {
            "evaluated_count": len(examples),
            "metric_calls": _metric_calls(evaluation, len(examples)),
            "actionable_indices": list(partition.actionable_indices),
            "regression_guard_indices": list(partition.regression_guard_indices),
            "tool_blocked_indices": list(partition.tool_blocked_indices),
            "satisfied_indices": list(partition.satisfied_indices),
            "other_indices": list(partition.other_indices),
            "optimization_indices": list(partition.optimization_indices),
            "fallback_to_unfiltered": bool(partition.fallback_to_unfiltered),
            "rows": rows,
        }
        self._preflight_summary = summary
        _write_json(self.run_dir / "diagnostics" / "actionability_preflight.json", summary)
        _write_jsonl(
            self.run_dir / "datasets" / "optimization_train.jsonl",
            _without_shared_rubric(optimization_examples, self._shared_rubric),
        )
        return summary

    def write_candidate(
        self,
        index: int,
        candidate: Mapping[str, str],
        *,
        label: str | None = None,
        parent_candidate: Mapping[str, str] | None = None,
    ) -> Path:
        candidate_dir = self.run_dir / "candidates" / f"{index:04d}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        _write_json(candidate_dir / "candidate.json", dict(candidate))
        manifest = _write_component_texts(candidate_dir, candidate)
        _write_json(candidate_dir / "manifest.json", manifest)
        _write_candidate_diffs(
            candidate_dir,
            candidate,
            parent_candidate=parent_candidate,
            seed_candidate=self._seed_candidate,
        )
        if label:
            label_dir = self.run_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            _write_json(label_dir / "candidate.json", dict(candidate))
            _write_json(label_dir / "manifest.json", _write_component_texts(label_dir, candidate))
            _write_candidate_diffs(
                label_dir,
                candidate,
                parent_candidate=parent_candidate,
                seed_candidate=self._seed_candidate,
            )
        return candidate_dir

    def write_agent_rollout(
        self,
        *,
        example: Mapping[str, Any],
        state: Mapping[str, Any],
        score: float,
        feedback: str,
    ) -> None:
        with self._lock:
            index = self._rollout_counter
            self._rollout_counter += 1
        record = {
            "index": index,
            "candidate_hash": state.get("candidate_hash"),
            "input": example.get("input"),
            "expected": example.get("expected") or example.get("answer"),
            "rubric": example.get("rubric"),
            "metadata": example.get("metadata", {}),
            "evaluation_phase": state.get("evaluation_phase", "optimization"),
            "score": score,
            "response": _last_message_text(state),
            "baseline_response": state.get("baseline_response", ""),
            "feedback": feedback,
            "fitness": state.get("fitness", {}),
            "constraints": state.get("candidate_constraints", []),
            "candidate_metrics": state.get("candidate_metrics", {}),
            "candidate_runtime_skipped": bool(state.get("candidate_runtime_skipped", False)),
            "candidate_runtime_skip_reason": state.get("candidate_runtime_skip_reason"),
            "state": _state_summary(state),
        }
        compact_record = {
            "index": index,
            "candidate_hash": record["candidate_hash"],
            "input": record["input"],
            "expected": record["expected"],
            "score": score,
            "response_preview": str(record["response"])[:500],
            "baseline_preview": str(record["baseline_response"])[:500],
            "feedback_preview": feedback[:500],
            "fitness": record["fitness"],
            "failure_classification": record["fitness"].get("failure_classification"),
            "remediation_type": record["fitness"].get("remediation_type"),
            "remediation_owner": record["fitness"].get("remediation_owner"),
            "remediation_actions": record["fitness"].get("remediation_actions", []),
            "tool_capability_gaps": record["fitness"].get("tool_capability_gaps", []),
            "tool_supported_missing_expectations": record["fitness"].get("tool_supported_missing_expectations", []),
            "skipped_supported_expectations": record["fitness"].get("skipped_supported_expectations", []),
            "failed_tool_expectations": record["fitness"].get("failed_tool_expectations", []),
            "incomplete_tool_result_expectations": record["fitness"].get("incomplete_tool_result_expectations", []),
            "evaluation_phase": record["evaluation_phase"],
            "error": record["state"]["error"],
            "candidate_runtime_skipped": record["candidate_runtime_skipped"],
            "detail_file": f"rollouts/{index:06d}.json",
        }
        with self._lock:
            _append_jsonl(self.run_dir / "agent_logs" / "rollouts.jsonl", compact_record)
            _write_json(self.run_dir / "agent_logs" / "rollouts" / f"{index:06d}.json", record)
            for action in record["fitness"].get("remediation_actions", []):
                if action.get("type") == "NO_ACTION":
                    continue
                _append_jsonl(
                    self.run_dir / "diagnostics" / "remediation_actions.jsonl",
                    {
                        "rollout_index": index,
                        "candidate_hash": record["candidate_hash"],
                        "input": record["input"],
                        "evaluation_phase": record["evaluation_phase"],
                        "score": score,
                        "failure_classification": record["fitness"].get("failure_classification"),
                        "type": action.get("type"),
                        "owner": action.get("owner"),
                        "targets": action.get("targets", []),
                        "reason": action.get("reason"),
                        "detail_file": f"agent_logs/rollouts/{index:06d}.json",
                    },
                )

    def create_callback(self) -> RunArtifactCallback:
        return RunArtifactCallback(self)

    def write_reflection_error(self, *, prompt: Any, error: Exception) -> None:
        prompt_text = str(prompt)
        component_match = re.search(r"Component boundary rules for `([^`]+)`", prompt_text)
        record = {
            "component": component_match.group(1) if component_match else None,
            "error_type": type(error).__name__,
            "error": str(error),
            "error_repr": repr(error),
            "prompt_chars": len(prompt_text),
        }
        with self._lock:
            index = self._reflection_error_counter
            self._reflection_error_counter += 1
            error_dir = self.run_dir / "reflection_errors"
            _write_json(error_dir / f"{index:06d}.json", record)
            (error_dir / f"{index:06d}.prompt.txt").write_text(prompt_text, encoding="utf-8")
            _append_jsonl(
                error_dir / "index.jsonl",
                {
                    "index": index,
                    **record,
                    "detail_file": f"{index:06d}.json",
                    "prompt_file": f"{index:06d}.prompt.txt",
                },
            )

    def write_proposal_review(
        self,
        *,
        prompt: str,
        original_response: str,
        decision: str,
        issues: Sequence[str],
        raw_review: str,
        reviewed_response: str | None,
        error: str | None = None,
        review_pass: int = 1,
    ) -> None:
        component_match = re.search(r"Component boundary rules for `([^`]+)`", prompt)
        with self._lock:
            index = self._proposal_review_counter
            self._proposal_review_counter += 1
        review_dir = self.run_dir / "proposal_reviews" / f"{index:06d}"
        review_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "index": index,
            "component": component_match.group(1) if component_match else None,
            "decision": decision,
            "review_pass": review_pass,
            "issues": list(issues),
            "error": error,
            "prompt_chars": len(prompt),
            "original_response_chars": len(original_response),
            "has_reviewed_response": reviewed_response is not None,
            "reviewed_response_chars": len(reviewed_response) if reviewed_response is not None else None,
        }
        _write_json(review_dir / "metadata.json", record)
        (review_dir / "reflection_prompt.txt").write_text(prompt, encoding="utf-8")
        (review_dir / "original_proposal.txt").write_text(original_response, encoding="utf-8")
        (review_dir / "raw_review.txt").write_text(raw_review, encoding="utf-8")
        if reviewed_response is not None:
            (review_dir / "reviewed_proposal.txt").write_text(reviewed_response, encoding="utf-8")
        with self._lock:
            _append_jsonl(
                self.run_dir / "proposal_reviews" / "index.jsonl",
                {
                    **record,
                    "detail_dir": review_dir.relative_to(self.run_dir).as_posix(),
                },
            )

    def write_proposal_snapshot(self, iteration: int, pending: Mapping[str, Any], status: str) -> None:
        proposal_dir = self.run_dir / "proposals" / f"{iteration:04d}"
        proposal_dir.mkdir(parents=True, exist_ok=True)
        candidate = dict(pending.get("candidate") or {})
        if candidate:
            _write_json(proposal_dir / "candidate.json", candidate)
            _write_json(proposal_dir / "manifest.json", _write_component_texts(proposal_dir, candidate))
            _write_candidate_diffs(
                proposal_dir,
                candidate,
                parent_candidate=dict(pending.get("parent_candidate") or {}),
                seed_candidate=self._seed_candidate,
            )

        prompts_dir = proposal_dir / "prompts"
        for component, prompt in dict(pending.get("prompts") or {}).items():
            prompts_dir.mkdir(parents=True, exist_ok=True)
            (prompts_dir / f"{_safe_name(str(component))}.txt").write_text(str(prompt), encoding="utf-8")

        raw_dir = proposal_dir / "raw_lm_outputs"
        raw_lm_outputs = dict(pending.get("raw_lm_outputs") or {})
        for component, raw_output in raw_lm_outputs.items():
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / f"{_safe_name(str(component))}.txt").write_text(str(raw_output), encoding="utf-8")
        rationales = _write_proposal_rationales(proposal_dir, raw_lm_outputs)
        missing_rationales = _missing_rationale_components(raw_lm_outputs, rationales)
        _write_missing_rationales(proposal_dir, missing_rationales)

        metadata = {
            key: value
            for key, value in pending.items()
            if key
            not in {
                "parent_candidate",
                "candidate",
                "prompts",
                "raw_lm_outputs",
                "raw_new_instructions",
                "reflective_dataset",
                "parent_outputs",
                "proposed_outputs",
                "parent_trajectories",
                "proposed_trajectories",
            }
        }
        metadata["status"] = status
        metadata["acceptance_scope"] = "candidate_pool_not_deployment" if status == "accepted" else None
        metadata["candidate_components"] = list(candidate)
        metadata["changed_components"] = sorted(dict(pending.get("new_instructions") or {}))
        metadata["component_consumption"] = dict(pending.get("component_consumption") or {})
        metadata["changed_but_unconsumed"] = list(pending.get("changed_but_unconsumed") or [])
        metadata["proposal_rationale"] = rationales
        metadata["missing_proposal_rationale"] = missing_rationales
        _write_json(proposal_dir / "metadata.json", metadata)
        _write_json(proposal_dir / "reflective_dataset.json", pending.get("reflective_dataset", {}))
        _write_json(proposal_dir / "new_instructions.json", pending.get("new_instructions", {}))
        _write_json(proposal_dir / "raw_new_instructions.json", pending.get("raw_new_instructions", {}))

        _append_jsonl(
            self.run_dir / "proposals" / "index.jsonl",
            {
                "iteration": iteration,
                "status": status,
                "acceptance_scope": "candidate_pool_not_deployment" if status == "accepted" else None,
                "components": pending.get("components", []),
                "changed_components": sorted(dict(pending.get("new_instructions") or {})),
                "semantic_noop_components": list(pending.get("semantic_noop_components") or []),
                "parent_scores": pending.get("parent_scores", []),
                "proposed_scores": pending.get("proposed_scores", []),
                "old_score": pending.get("old_score"),
                "new_score": pending.get("new_score"),
                "reason": pending.get("reason"),
                "failure_classifications": pending.get("failure_classifications", []),
                "component_consumption": pending.get("component_consumption", {}),
                "changed_but_unconsumed": pending.get("changed_but_unconsumed", []),
                "proposal_rationale_preview": {
                    component: rationale[:300] for component, rationale in rationales.items()
                },
                "missing_proposal_rationale": missing_rationales,
                "proposal_dir": proposal_dir.relative_to(self.run_dir).as_posix(),
            },
        )

    def write_rejected_proposal(self, iteration: int, pending: Mapping[str, Any]) -> None:
        rejected_dir = self.run_dir / "rejected_proposals" / f"{iteration:04d}"
        rejected_dir.mkdir(parents=True, exist_ok=True)
        candidate = dict(pending.get("candidate") or {})
        if candidate:
            _write_json(rejected_dir / "candidate.json", candidate)
            _write_json(rejected_dir / "manifest.json", _write_component_texts(rejected_dir, candidate))
            _write_candidate_diffs(
                rejected_dir,
                candidate,
                parent_candidate=dict(pending.get("parent_candidate") or {}),
                seed_candidate=self._seed_candidate,
            )
        raw_lm_outputs = dict(pending.get("raw_lm_outputs") or {})
        rationales = _write_proposal_rationales(rejected_dir, raw_lm_outputs)
        missing_rationales = _missing_rationale_components(raw_lm_outputs, rationales)
        _write_missing_rationales(rejected_dir, missing_rationales)
        metadata = dict(pending)
        metadata["proposal_rationale"] = rationales
        metadata["missing_proposal_rationale"] = missing_rationales
        _write_json(rejected_dir / "metadata.json", metadata)
        _append_jsonl(
            self.run_dir / "rejected_proposals" / "index.jsonl",
            {
                "iteration": iteration,
                "components": pending.get("components", []),
                "changed_components": sorted(dict(pending.get("new_instructions") or {})),
                "parent_scores": pending.get("parent_scores", []),
                "proposed_scores": pending.get("proposed_scores", []),
                "old_score": pending.get("old_score"),
                "new_score": pending.get("new_score"),
                "reason": pending.get("reason"),
                "failure_classifications": pending.get("failure_classifications", []),
                "proposal_rationale_preview": {
                    component: rationale[:300] for component, rationale in rationales.items()
                },
                "missing_proposal_rationale": missing_rationales,
                "proposal_dir": rejected_dir.relative_to(self.run_dir).as_posix(),
            },
        )

    def write_final_test(
        self,
        *,
        examples: Sequence[Mapping[str, Any]],
        seed_evaluation: Any,
        best_evaluation: Any,
        diagnostic_evaluations: Mapping[int, Any] | None = None,
        diagnostic_val_scores: Mapping[int, float] | None = None,
    ) -> dict[str, Any]:
        seed_payload = _evaluation_payload(examples, seed_evaluation)
        best_payload = _evaluation_payload(examples, best_evaluation)
        _write_json(self.run_dir / "final_test" / "seed.json", seed_payload)
        _write_json(self.run_dir / "final_test" / "best.json", best_payload)
        seed_mean = _score_mean(seed_payload["scores"])
        best_mean = _score_mean(best_payload["scores"])
        distinct_evaluations = {id(seed_evaluation): seed_evaluation, id(best_evaluation): best_evaluation}
        distinct_evaluations.update(
            {id(evaluation): evaluation for evaluation in (diagnostic_evaluations or {}).values()}
        )
        self._final_test_metric_calls = sum(
            _metric_calls(evaluation, len(examples)) for evaluation in distinct_evaluations.values()
        )
        comparison = {
            "count": len(examples),
            "metric_calls": self._final_test_metric_calls,
            "seed_mean": seed_mean,
            "best_mean": best_mean,
            "improvement": best_mean - seed_mean,
            "per_example": [
                {
                    "input": example.get("input"),
                    "seed_score": seed_payload["scores"][index],
                    "best_score": best_payload["scores"][index],
                    "delta": best_payload["scores"][index] - seed_payload["scores"][index],
                }
                for index, example in enumerate(examples)
            ],
            "seed_cohorts": _evaluation_cohort_rows(examples, seed_evaluation),
            "best_cohorts": _evaluation_cohort_rows(examples, best_evaluation),
        }
        diagnostic_rows: list[dict[str, Any]] = []
        for candidate_idx, evaluation in sorted((diagnostic_evaluations or {}).items()):
            payload = _evaluation_payload(examples, evaluation)
            _write_json(self.run_dir / "final_test" / f"candidate_{candidate_idx:04d}.json", payload)
            mean = _score_mean(payload["scores"])
            diagnostic_rows.append(
                {
                    "candidate_idx": candidate_idx,
                    "validation_score": (diagnostic_val_scores or {}).get(candidate_idx),
                    "test_mean": mean,
                    "delta_vs_seed": mean - seed_mean,
                    "selection_effect": "diagnostic_only",
                }
            )
        if diagnostic_rows:
            comparison["diagnostic_candidates"] = diagnostic_rows
        _write_json(self.run_dir / "final_test" / "summary.json", comparison)
        return comparison

    def write_result_summary(
        self,
        result: Any,
        final_test: Mapping[str, Any] | None = None,
        *,
        best_idx: int | None = None,
    ) -> dict[str, Any]:
        gepa_best_idx = getattr(result, "best_idx", None)
        selected_best_idx = gepa_best_idx if best_idx is None else best_idx
        candidates = list(getattr(result, "candidates", []) or [])
        best_candidate = (
            candidates[selected_best_idx]
            if selected_best_idx is not None and 0 <= selected_best_idx < len(candidates)
            else getattr(result, "best_candidate", {})
        )
        val_scores = list(getattr(result, "val_aggregate_scores", []) or [])
        tied_best_indices: list[int] = []
        if val_scores:
            max_score = max(float(score) for score in val_scores)
            tied_best_indices = [
                index for index, score in enumerate(val_scores) if abs(float(score) - max_score) <= 1e-12
            ]
        gepa_metric_calls = getattr(result, "total_metric_calls", None)
        preflight_metric_calls = (
            int(self._preflight_summary.get("metric_calls", 0)) if self._preflight_summary is not None else 0
        )
        all_phase_metric_calls = (
            int(gepa_metric_calls) + preflight_metric_calls + self._final_test_metric_calls
            if gepa_metric_calls is not None
            else None
        )
        summary = {
            "result_type": type(result).__name__,
            "best_idx": selected_best_idx,
            "gepa_best_idx": gepa_best_idx,
            "tie_break_applied": selected_best_idx != gepa_best_idx,
            "selection_policy": "incumbent_on_validation_tie",
            "tied_best_indices": tied_best_indices,
            "best_val_score": (
                val_scores[selected_best_idx]
                if selected_best_idx is not None and 0 <= selected_best_idx < len(val_scores)
                else None
            ),
            "val_aggregate_scores": val_scores,
            "parents": getattr(result, "parents", None),
            "discovery_eval_counts": getattr(result, "discovery_eval_counts", None),
            "total_metric_calls": gepa_metric_calls,
            "overall_metric_calls": all_phase_metric_calls,
            "metric_calls_by_phase": {
                "gepa": gepa_metric_calls,
                "preflight": preflight_metric_calls,
                "final_test": self._final_test_metric_calls,
                "all_phases": all_phase_metric_calls,
            },
            "num_full_val_evals": getattr(result, "num_full_val_evals", None),
            "num_candidates": getattr(result, "num_candidates", None),
            "run_dir": str(self.run_dir),
            "component_lengths": (
                {name: len(text) for name, text in best_candidate.items()} if isinstance(best_candidate, dict) else None
            ),
            "preflight_actionability": self._preflight_summary,
            "metric_budget_plan": self._budget_plan,
            "final_test": dict(final_test) if final_test is not None else None,
        }
        _write_json(self.run_dir / "result_summary.json", summary)
        return summary

    def write_result_candidates(self, result: Any, *, best_idx: int | None = None) -> None:
        candidates = getattr(result, "candidates", [])
        val_scores = list(getattr(result, "val_aggregate_scores", []) or [])
        selected_best_idx = getattr(result, "best_idx", None) if best_idx is None else best_idx
        for index, candidate in enumerate(candidates):
            label = "best_candidate" if index == selected_best_idx else None
            parent_candidate = self._parent_candidate_for_result(
                index, candidates, getattr(result, "parents", []) or []
            )
            self.write_candidate(index, candidate, label=label, parent_candidate=parent_candidate)
            metadata = {
                "index": index,
                "val_score": val_scores[index] if index < len(val_scores) else None,
                "parent_indices": (getattr(result, "parents", []) or [[]])[index]
                if index < len(getattr(result, "parents", []) or [])
                else [],
                "status": "best" if index == selected_best_idx else "accepted_non_best",
            }
            _write_json(self.run_dir / "candidates" / f"{index:04d}" / "metadata.json", metadata)
            if index != selected_best_idx:
                rejected_dir = self.run_dir / "rejected_candidates" / f"{index:04d}"
                rejected_dir.mkdir(parents=True, exist_ok=True)
                _write_json(rejected_dir / "metadata.json", metadata)
                _write_json(rejected_dir / "candidate.json", candidate)
                _write_candidate_diffs(
                    rejected_dir,
                    candidate,
                    parent_candidate=parent_candidate,
                    seed_candidate=self._seed_candidate,
                )

    def _parent_candidate_for_result(
        self,
        index: int,
        candidates: Sequence[Mapping[str, str]],
        parents: Sequence[Sequence[int | None]],
    ) -> Mapping[str, str] | None:
        if index >= len(parents) or not parents[index]:
            return None
        parent_idx = parents[index][0]
        if parent_idx is None or parent_idx >= len(candidates):
            return None
        return candidates[parent_idx]

    def materialize_best_candidate(
        self,
        *,
        result: Any,
        project: Any,
        apply_candidate: Callable[[Any, Mapping[str, str], Path], Any],
        best_idx: int | None = None,
    ) -> None:
        candidates = list(getattr(result, "candidates", []) or [])
        best_candidate = (
            candidates[best_idx]
            if best_idx is not None and 0 <= best_idx < len(candidates)
            else getattr(result, "best_candidate", None)
        )
        if not isinstance(best_candidate, Mapping):
            return
        materialized_dir = self.run_dir / "materialized_best_candidate"
        apply_candidate(project, best_candidate, materialized_dir)

    def finalize(
        self,
        *,
        result: Any,
        project: Any,
        apply_candidate: Callable[[Any, Mapping[str, str], Path], Any],
        final_test: Mapping[str, Any] | None = None,
        best_idx: int | None = None,
    ) -> dict[str, Any]:
        self.write_result_candidates(result, best_idx=best_idx)
        self.materialize_best_candidate(
            result=result,
            project=project,
            apply_candidate=apply_candidate,
            best_idx=best_idx,
        )
        return self.write_result_summary(result, final_test=final_test, best_idx=best_idx)


class RunArtifactCallback:
    """GEPA callback that persists proposal lifecycle events."""

    def __init__(self, store: RunArtifactStore, max_rejected_history: int = 5) -> None:
        self.store = store
        self.max_rejected_history = max(1, max_rejected_history)
        self._pending: dict[int, dict[str, Any]] = {}
        self._parent_eval: dict[int, dict[str, Any]] = {}
        self._rejected_history: list[dict[str, Any]] = []

    def on_evaluation_end(self, event: Mapping[str, Any]) -> None:
        iteration = int(event["iteration"])
        payload = {
            "scores": list(event.get("scores") or []),
            "outputs": event.get("outputs") or [],
            "trajectories": event.get("trajectories") or [],
        }
        if event.get("candidate_idx") is None and not event.get("is_seed_candidate", False):
            pending = self._pending.setdefault(iteration, {"iteration": iteration})
            pending["proposed_scores"] = payload["scores"]
            pending["proposed_outputs"] = payload["outputs"]
            pending["proposed_trajectories"] = payload["trajectories"]
            pending["failure_classifications"] = _failure_classes_from_trajectories(payload["trajectories"])
            changed_components = sorted(dict(pending.get("new_instructions") or {}))
            consumption = {
                component: component_consumption(component, payload["trajectories"]) for component in changed_components
            }
            pending["component_consumption"] = consumption
            pending["changed_but_unconsumed"] = [
                component for component, consumed in consumption.items() if consumed is False
            ]
            if pending.get("candidate"):
                self.store.write_proposal_snapshot(iteration, pending, status=str(pending.get("status", "evaluated")))
            return

        self._parent_eval[iteration] = payload
        if iteration in self._pending:
            self._pending[iteration]["parent_scores"] = payload["scores"]
            self._pending[iteration]["parent_outputs"] = payload["outputs"]
            self._pending[iteration]["parent_trajectories"] = payload["trajectories"]

    def on_proposal_start(self, event: Mapping[str, Any]) -> None:
        iteration = int(event["iteration"])
        parent_eval = self._parent_eval.get(iteration, {})
        self._pending[iteration] = {
            "iteration": iteration,
            "status": "started",
            "parent_candidate": dict(event.get("parent_candidate") or {}),
            "components": list(event.get("components") or []),
            "reflective_dataset": event.get("reflective_dataset") or {},
            "parent_scores": parent_eval.get("scores", []),
            "parent_outputs": parent_eval.get("outputs", []),
            "parent_trajectories": parent_eval.get("trajectories", []),
        }
        self.store.write_proposal_snapshot(iteration, self._pending[iteration], status="started")

    def on_proposal_end(self, event: Mapping[str, Any]) -> None:
        iteration = int(event["iteration"])
        pending = self._pending.setdefault(iteration, {"iteration": iteration})
        parent_candidate = dict(pending.get("parent_candidate") or {})
        raw_new_instructions = dict(event.get("new_instructions") or {})
        new_instructions = _material_new_instructions(parent_candidate, raw_new_instructions)
        semantic_noop_components = sorted(set(raw_new_instructions) - set(new_instructions))
        candidate = dict(parent_candidate)
        candidate.update(new_instructions)
        pending.update(
            {
                "status": "proposed",
                "new_instructions": new_instructions,
                "raw_new_instructions": raw_new_instructions,
                "semantic_noop_components": semantic_noop_components,
                "candidate": candidate,
                "prompts": dict(event.get("prompts") or {}),
                "raw_lm_outputs": dict(event.get("raw_lm_outputs") or {}),
            }
        )
        self.store.write_proposal_snapshot(iteration, pending, status="proposed")

    def on_candidate_rejected(self, event: Mapping[str, Any]) -> None:
        iteration = int(event["iteration"])
        pending = self._pending.setdefault(iteration, {"iteration": iteration})
        pending.update(
            {
                "status": "rejected",
                "old_score": event.get("old_score"),
                "new_score": event.get("new_score"),
                "reason": event.get("reason"),
            }
        )
        self._remember_rejected(iteration, pending)
        self.store.write_proposal_snapshot(iteration, pending, status="rejected")
        self.store.write_rejected_proposal(iteration, pending)

    def on_candidate_accepted(self, event: Mapping[str, Any]) -> None:
        iteration = int(event["iteration"])
        pending = self._pending.get(iteration)
        if not pending:
            return
        pending.update(
            {
                "status": "accepted",
                "candidate_idx": event.get("new_candidate_idx"),
                "new_score": event.get("new_score"),
                "parent_ids": event.get("parent_ids", []),
            }
        )
        self.store.write_proposal_snapshot(iteration, pending, status="accepted")

    def rejected_history_prompt_block(self) -> str:
        if not self._rejected_history:
            return ""
        lines = ["Recent rejected proposal lessons (negative evidence, not text to copy):"]
        for item in self._rejected_history[-self.max_rejected_history :]:
            lines.append(
                "- iteration {iteration}, changed={changed_components}, old_score={old_score}, new_score={new_score}, "
                "reason={reason}, failure_classes={failure_classifications}, semantic_noop={semantic_noop}".format(
                    **item
                )
            )
            if item.get("proposal_lessons"):
                lines.append("  Rejected rationale: " + " | ".join(item["proposal_lessons"]))
            if item.get("diff_preview"):
                lines.append("  Rejected diff preview:\n" + str(item["diff_preview"]))
        lines.append(
            "Do not resubmit the rejected edit or merely paraphrase it. Use the evidence to choose a different causal fix, "
            "or preserve the incumbent when no text mutation is justified."
        )
        return "\n".join(lines)

    def _remember_rejected(self, iteration: int, pending: Mapping[str, Any]) -> None:
        parent_candidate = dict(pending.get("parent_candidate") or {})
        candidate = dict(pending.get("candidate") or parent_candidate)
        changed_components = [
            key
            for key in sorted(set(parent_candidate) | set(candidate))
            if parent_candidate.get(key) != candidate.get(key)
        ]
        semantic_noop = bool(pending.get("semantic_noop_components")) and not changed_components
        proposal_lessons = []
        for component, raw_output in (pending.get("raw_lm_outputs") or {}).items():
            rationale = _extract_proposal_rationale(str(raw_output))
            if rationale:
                compact = re.sub(r"\s+", " ", rationale).strip()[:700]
                proposal_lessons.append(f"{component}: {compact}")
        diff_preview = _candidate_diff(
            candidate,
            parent_candidate,
            baseline_label="parent",
            candidate_label="rejected",
        )[:1200]
        self._rejected_history.append(
            {
                "iteration": iteration,
                "components": list(pending.get("components") or []),
                "changed_components": changed_components,
                "old_score": pending.get("old_score"),
                "new_score": pending.get("new_score"),
                "reason": str(pending.get("reason", ""))[:500],
                "failure_classifications": list(pending.get("failure_classifications") or []),
                "semantic_noop": semantic_noop,
                "proposal_lessons": proposal_lessons[:3],
                "diff_preview": diff_preview,
            }
        )
        if len(self._rejected_history) > self.max_rejected_history:
            self._rejected_history = self._rejected_history[-self.max_rejected_history :]
