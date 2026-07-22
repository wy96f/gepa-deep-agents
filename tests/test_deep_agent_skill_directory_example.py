from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core", reason="requires gepa[langchain] extra")
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import ToolMessage


class ToolFriendlyFakeChatModel(FakeListChatModel):
    def bind_tools(self, tools, **kwargs):
        del tools, kwargs
        return self


def _load_example_module():
    path = Path(__file__).parents[1] / "examples" / "langchain_adapter" / "deep_agent_skill_directory.py"
    spec = importlib.util.spec_from_file_location("deep_agent_skill_directory_example", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_analyzer_module():
    path = Path(__file__).parents[1] / "examples" / "langchain_adapter" / "analyze_deepagents_gepa_run.py"
    spec = importlib.util.spec_from_file_location("analyze_deepagents_gepa_run", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_runner_module():
    path = Path(__file__).parents[1] / "examples" / "langchain_adapter" / "run_deepagents_gepa_local.py"
    spec = importlib.util.spec_from_file_location("run_deepagents_gepa_local", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_cleaner_module():
    path = Path(__file__).parents[1] / "examples" / "langchain_adapter" / "clean_credit_risk_dataset.py"
    spec = importlib.util.spec_from_file_location("clean_credit_risk_dataset", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_candidate_discovery_includes_expected_text_surfaces_and_excludes_scripts(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path)

    candidate, surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)

    assert "memory:AGENTS.md" in candidate
    assert "main:system_prompt" in candidate
    assert "main:tool:tag_ticket:description" in candidate
    assert "subagent:triage:description" in candidate
    assert "subagent:triage:system_prompt" in candidate
    assert "subagent:triage:tool:lookup_policy:description" in candidate
    assert "skill:support-router:SKILL.md" in candidate
    assert "skill:support-router:reference/routing.md" in candidate
    assert "subagent:triage:skill:triage-notes:SKILL.md" in candidate
    assert "subagent:triage:skill:triage-notes:reference/signals.md" in candidate
    assert all("scripts/" not in key for key in candidate)

    assert surfaces["skill:support-router:SKILL.md"].source_path == "skills"
    assert surfaces["subagent:triage:skill:triage-notes:SKILL.md"].source_path == "subagents/triage/skills"


def test_apply_candidate_uses_native_deepagents_paths_and_preserves_original_tools(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path / "seed")
    candidate, surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)

    original_main_description = seed_spec.tools[0].description
    original_subagent_description = seed_spec.subagents[0]["tools"][0].description
    candidate["memory:AGENTS.md"] += "\n\nRemember routing confidence."
    candidate["main:tool:tag_ticket:description"] = "Apply the chosen support route label to a ticket."
    candidate["subagent:triage:tool:lookup_policy:description"] = "Retrieve exact routing policy evidence."
    candidate["subagent:triage:skill:triage-notes:reference/signals.md"] += "\n- Refund means billing."

    application = example.apply_candidate_to_deep_agent_spec(seed_spec, candidate, surfaces, tmp_path / "applied")

    assert application.kwargs["memory"] == ["AGENTS.md"]
    assert application.kwargs["skills"] == ["skills"]
    assert application.kwargs["subagents"][0]["skills"] == ["subagents/triage/skills"]
    assert application.kwargs["tools"][0].description == "Apply the chosen support route label to a ticket."
    assert application.kwargs["subagents"][0]["tools"][0].description == "Retrieve exact routing policy evidence."

    assert seed_spec.tools[0].description == original_main_description
    assert seed_spec.subagents[0]["tools"][0].description == original_subagent_description
    assert (tmp_path / "applied" / "AGENTS.md").read_text().endswith("Remember routing confidence.")
    assert (tmp_path / "applied" / "skills" / "support-router" / "SKILL.md").exists()
    assert (tmp_path / "applied" / "subagents" / "triage" / "skills" / "triage-notes" / "SKILL.md").exists()
    assert "scripts/ignored.py" not in candidate
    assert (tmp_path / "applied" / "skills" / "support-router" / "scripts" / "ignored.py").exists()
    assert (tmp_path / "applied" / "scripts" / "ignored.py").exists()
    assert (
        tmp_path / "applied" / "subagents" / "triage" / "skills" / "triage-notes" / "scripts" / "ignored.py"
    ).exists()


def test_apply_candidate_cleans_stale_skill_files_when_reusing_output_dir(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path / "seed")
    candidate, surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)
    applied_root = tmp_path / "applied"

    example.apply_candidate_to_deep_agent_spec(seed_spec, candidate, surfaces, applied_root)
    stale_file = applied_root / "skills" / "support-router" / "reference" / "stale.md"
    stale_file.write_text("old generated content", encoding="utf-8")
    stale_alias = applied_root / "scripts" / "stale.py"
    stale_alias.write_text("print('old')\n", encoding="utf-8")

    example.apply_candidate_to_deep_agent_spec(seed_spec, candidate, surfaces, applied_root)

    assert not stale_file.exists()
    assert not stale_alias.exists()


def test_script_references_are_validated_against_materialized_workspace(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path / "seed")
    candidate, surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)
    applied_root = tmp_path / "applied"
    example.apply_candidate_to_deep_agent_spec(seed_spec, candidate, surfaces, applied_root)

    constraints = example.validate_candidate_constraints(candidate, candidate, surfaces, materialized_root=applied_root)
    script_constraints = [constraint for constraint in constraints if ":script:scripts/ignored.py" in constraint.name]

    assert script_constraints
    assert all(constraint.passed for constraint in script_constraints)

    candidate["skill:support-router:SKILL.md"] += "\nRun `python scripts/missing.py` if unsure.\n"
    failed = example.validate_candidate_constraints(candidate, candidate, surfaces, materialized_root=applied_root)

    assert any(
        constraint.name == "skill:support-router:SKILL.md:script:scripts/missing.py" and not constraint.passed
        for constraint in failed
    )


def test_executable_backend_can_run_materialized_skill_scripts(tmp_path):
    pytest.importorskip("deepagents", reason="requires deepagents for LocalShellBackend")
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path / "seed")
    candidate, surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)
    applied_root = tmp_path / "applied"
    example.apply_candidate_to_deep_agent_spec(seed_spec, candidate, surfaces, applied_root)

    backend = example.create_executable_deep_agent_backend(applied_root)
    result = backend.execute("python scripts/ignored.py", timeout=10)

    assert result.exit_code == 0


def test_runtime_gate_and_dry_run_feedback(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path)
    candidate, surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)
    candidate["skill:support-router:SKILL.md"] += "\nUse this Claude Code skill only.\n"

    constraints = example.validate_candidate_constraints(candidate, candidate, surfaces)
    failures = [constraint for constraint in constraints if not constraint.passed]

    assert any(constraint.name == "runtime_neutrality" for constraint in failures)
    score, mode = example.effect_score("<route>billing</route>", "DRY_RUN_BASELINE_UNAVAILABLE: missing", "billing")
    assert mode == "dry_run"
    assert score > 0


def test_feedback_classifies_execution_lapse_and_recommends_memory(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path)
    candidate, _surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)
    state = {
        "candidate_excerpt": candidate,
        "candidate_constraints": [],
    }
    fitness = {
        "hard": 0.0,
        "soft": 0.0,
        "mixed": 0.0,
        "baseline_hard": 0.0,
        "baseline_soft": 0.0,
        "baseline_mixed": 0.0,
        "effect": 0.0,
        "structure": 1.0,
        "specificity": 1.0,
        "gate_penalty": 0.0,
        "eval_mode": "full_test",
        "composite": 0.0,
    }

    feedback = example.build_feedback(
        {"input": "Where can I download my receipt?", "answer": "billing"},
        state,
        "billing",
        "",
        [],
        fitness,
    )

    assert "- failure_classification: EXECUTION_LAPSE" in feedback
    assert "- suggested_component: memory:AGENTS.md" in feedback


def test_feedback_classifies_skill_defect_and_recommends_failed_component(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path)
    candidate, _surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)
    state = {
        "candidate_excerpt": candidate,
        "candidate_constraints": [
            {
                "passed": False,
                "name": "skill:support-router:SKILL.md:runtime_neutrality",
                "message": "runtime-specific terms",
            }
        ],
    }
    fitness = {
        "hard": 0.0,
        "soft": 0.35,
        "mixed": 0.175,
        "baseline_hard": 0.0,
        "baseline_soft": 0.0,
        "baseline_mixed": 0.0,
        "effect": 0.0,
        "structure": 0.8,
        "specificity": 1.0,
        "gate_penalty": 0.25,
        "eval_mode": "full_test",
        "composite": 0.0,
    }

    feedback = example.build_feedback(
        {"input": "Please reset my password.", "answer": "account"},
        state,
        "<route>billing</route>",
        "",
        state["candidate_constraints"],
        fitness,
    )

    assert "- failure_classification: SKILL_DEFECT" in feedback
    assert "- suggested_component: skill:support-router:SKILL.md" in feedback


def test_darwin_feedback_component_selector_aggregates_low_score_suggestions(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path)
    candidate, _surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)
    selector = example.DarwinFeedbackComponentSelector()

    selected = selector(
        state=None,
        trajectories=[
            {
                "score": 0.2,
                "feedback": "Scores:\n- suggested_component: skill:support-router:SKILL.md:runtime_neutrality",
            },
            {
                "score": 0.9,
                "feedback": "Scores:\n- suggested_component: main:system_prompt",
            },
            {
                "score": 0.1,
                "feedback": "Scores:\n- suggested_component: skill:support-router:SKILL.md",
            },
        ],
        subsample_scores=[0.2, 0.9, 0.1],
        candidate_idx=0,
        candidate=candidate,
    )

    assert selected == ["skill:support-router:SKILL.md"]


def test_darwin_feedback_component_selector_falls_back_without_valid_suggestion(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path)
    candidate, _surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)
    selector = example.DarwinFeedbackComponentSelector()

    selected = selector(
        state=None,
        trajectories=[{"score": 0.0, "feedback": "Scores:\n- suggested_component: missing:key"}],
        subsample_scores=[0.0],
        candidate_idx=0,
        candidate=candidate,
    )

    assert selected == ["main:system_prompt"]


def test_darwin_feedback_component_selector_cools_down_repeated_failed_component(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path)
    candidate, _surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)
    selector = example.DarwinFeedbackComponentSelector(cooldown_after=2)
    trajectory = {"score": 0.0, "feedback": "Scores:\n- suggested_component: memory:AGENTS.md"}

    first = selector(None, [trajectory], [0.0], 0, candidate)
    second = selector(None, [trajectory], [0.0], 0, candidate)
    third = selector(None, [trajectory], [0.0], 0, candidate)

    assert first == ["memory:AGENTS.md"]
    assert second == ["memory:AGENTS.md"]
    assert third != ["memory:AGENTS.md"]


def test_component_selector_skips_text_mutation_for_tool_capability_only_batch(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path)
    candidate, _surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)
    selector = example.DarwinFeedbackComponentSelector()

    selected = selector(
        None,
        [
            {
                "score": 0.1,
                "feedback": "Scores:\n- failure_classification: TOOL_CAPABILITY_GAP\n- suggested_component: none",
            }
        ],
        [0.1],
        0,
        candidate,
    )

    assert selected == []


def test_component_selector_skips_insufficient_runtime_evidence_votes(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path)
    candidate, _surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)
    selector = example.DarwinFeedbackComponentSelector()

    selected = selector(
        None,
        [
            {
                "score": 0.1,
                "feedback": "Scores:\n"
                "- failure_classification: INSUFFICIENT_RUNTIME_EVIDENCE\n"
                "- mutation_eligible: false\n"
                "- suggested_component: skill:support-router:reference/routing.md",
            }
        ],
        [0.1],
        0,
        candidate,
    )

    assert selected == []


def test_component_selector_ignores_no_failure_votes(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path)
    candidate, _surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)
    selector = example.DarwinFeedbackComponentSelector()
    no_failure_feedback = (
        "Scores:\n"
        "- failure_classification: NO_FAILURE\n"
        "- suggested_component: skill:support-router:reference/routing.md"
    )
    execution_feedback = "Scores:\n- failure_classification: EXECUTION_LAPSE\n- suggested_component: memory:AGENTS.md"

    selected = selector(
        None,
        [
            {"score": 1.0, "feedback": no_failure_feedback},
            {"score": 1.0, "feedback": no_failure_feedback},
            {"score": 0.0, "feedback": execution_feedback},
        ],
        [1.0, 1.0, 0.0],
        0,
        candidate,
    )

    assert selected == ["memory:AGENTS.md"]
    assert selector(None, [{"score": 1.0, "feedback": no_failure_feedback}], [1.0], 1, candidate) == []


def test_component_selector_only_mutates_owning_skill_when_reference_route_is_missing():
    example = _load_example_module()
    learned_key = "skill:credit-risk-review:reference/learned_expert_patterns.md"
    skill_key = "skill:credit-risk-review:SKILL.md"
    ordinary_reference_key = "skill:credit-risk-review:reference/cashflow_and_repayment.md"
    candidate = {
        learned_key: "# Learned",
        skill_key: (
            "---\nname: credit-risk-review\ndescription: Review credit risk.\n---\n"
            "# Workflow\nRead `reference/learned_expert_patterns.md` when applicable."
        ),
        ordinary_reference_key: "# Cash flow",
    }
    selector = example.DarwinFeedbackComponentSelector()

    selected = selector(
        None,
        [
            {
                "score": 0.2,
                "feedback": f"Scores:\n- failure_classification: SKILL_DEFECT\n- suggested_component: {learned_key}",
            }
        ],
        [0.2],
        0,
        candidate,
    )
    ordinary_selected = selector(
        None,
        [
            {
                "score": 0.2,
                "feedback": "Scores:\n"
                "- failure_classification: SKILL_DEFECT\n"
                f"- suggested_component: {ordinary_reference_key}",
            }
        ],
        [0.2],
        1,
        candidate,
    )
    candidate_without_route = dict(candidate)
    candidate_without_route[skill_key] = (
        "---\nname: credit-risk-review\ndescription: Review credit risk.\n---\n# Workflow"
    )
    selected_without_route = selector(
        None,
        [
            {
                "score": 0.2,
                "feedback": f"Scores:\n- failure_classification: SKILL_DEFECT\n- suggested_component: {learned_key}",
            }
        ],
        [0.2],
        2,
        candidate_without_route,
    )

    assert selected == [learned_key]
    assert selected_without_route == [learned_key, skill_key]
    assert ordinary_selected == [ordinary_reference_key]


def test_component_selector_makes_unread_reference_reachable_from_execution_policy():
    example = _load_example_module()
    reference_key = "skill:credit-risk-review:reference/financial_statement_analysis.md"
    skill_key = "skill:credit-risk-review:SKILL.md"
    candidate = {
        "memory:AGENTS.md": "Use the credit risk skill.",
        skill_key: "Read the relevant reference when needed.",
        reference_key: "# Financial analysis",
    }
    feedback = f"Scores:\n- failure_classification: SKILL_DEFECT\n- suggested_component: {reference_key}"
    selector = example.DarwinFeedbackComponentSelector()

    selected = selector(
        None,
        [{"score": 0.4, "feedback": feedback, "state": {"messages": []}}],
        [0.4],
        0,
        candidate,
    )

    assert selected == ["memory:AGENTS.md"]


def test_component_selector_routes_from_consumed_skill_to_unread_reference():
    example = _load_example_module()
    reference_key = "skill:credit-risk-review:reference/financial_statement_analysis.md"
    skill_key = "skill:credit-risk-review:SKILL.md"
    candidate = {
        "memory:AGENTS.md": "Use the credit risk skill.",
        skill_key: "Read the relevant reference when needed.",
        reference_key: "# Financial analysis",
    }
    feedback = f"Scores:\n- failure_classification: SKILL_DEFECT\n- suggested_component: {reference_key}"
    skill_read = {
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"file_path": "/skills/credit-risk-review/SKILL.md"},
            }
        ]
    }
    selector = example.DarwinFeedbackComponentSelector()

    selected = selector(
        None,
        [{"score": 0.4, "feedback": feedback, "state": {"messages": [skill_read]}}],
        [0.4],
        0,
        candidate,
    )

    assert selected == [skill_key]


def test_component_selector_keeps_consumed_reference_as_single_target():
    example = _load_example_module()
    reference_key = "skill:credit-risk-review:reference/financial_statement_analysis.md"
    candidate = {
        "memory:AGENTS.md": "Use the credit risk skill.",
        "skill:credit-risk-review:SKILL.md": "Read the relevant reference when needed.",
        reference_key: "# Financial analysis",
    }
    feedback = f"Scores:\n- failure_classification: SKILL_DEFECT\n- suggested_component: {reference_key}"
    reference_read = {
        "tool_calls": [
            {
                "name": "read_file",
                "args": {"path": "/skills/credit-risk-review/reference/financial_statement_analysis.md"},
            }
        ]
    }
    selector = example.DarwinFeedbackComponentSelector()

    selected = selector(
        None,
        [{"score": 0.4, "feedback": feedback, "state": {"messages": [reference_read]}}],
        [0.4],
        0,
        candidate,
    )

    assert selected == [reference_key]


def test_deployment_candidate_selection_preserves_incumbent_on_validation_tie():
    example = _load_example_module()
    result = SimpleNamespace(
        candidates=[{"prompt": "seed"}, {"prompt": "accepted"}, {"prompt": "lower"}],
        val_aggregate_scores=[1.0, 1.0, 0.8],
        best_idx=0,
    )

    assert example.select_deployment_candidate_index(result) == 0
    result.val_aggregate_scores = [1.0, 1.0 + 1e-8, 0.8]
    assert example.select_deployment_candidate_index(result) == 1


def test_default_evaluator_writes_fitness_back_to_original_state():
    example = _load_example_module()
    state = {"messages": []}

    def evaluate(_example, mutable_state):
        mutable_state["fitness"] = {"composite": 0.75}
        return 0.75, "ok"

    score, _feedback = example.DefaultEvaluator(evaluate).evaluate({"input": "x"}, state)

    assert score == 0.75
    assert state["fitness"] == {"composite": 0.75}


def test_default_actionability_policy_builds_mutation_and_diagnostic_cohorts():
    example = _load_example_module()
    evaluation = SimpleNamespace(
        scores=[0.2, 0.1, 1.0, 0.8],
        outputs=[
            {"state": {"fitness": {"mutation_eligible": True, "failure_classification": "SKILL_DEFECT"}}},
            {"state": {"fitness": {"mutation_eligible": False, "failure_classification": "TOOL_CAPABILITY_GAP"}}},
            {"state": {"fitness": {"mutation_eligible": False, "failure_classification": "NO_FAILURE"}}},
            {"state": {"fitness": {"mutation_eligible": False, "failure_classification": "NO_FAILURE"}}},
        ],
    )

    partition = example.DefaultActionabilityPolicy().partition(
        [{"input": str(index)} for index in range(4)],
        evaluation,
        regression_guard_limit=1,
    )

    assert partition.actionable_indices == (0,)
    assert partition.tool_blocked_indices == (1,)
    assert partition.regression_guard_indices == (2,)
    assert partition.optimization_indices == (0, 2)
    assert partition.fallback_to_unfiltered is False


def test_default_actionability_policy_falls_back_when_nothing_is_text_actionable():
    example = _load_example_module()
    evaluation = SimpleNamespace(
        scores=[0.1, 0.2],
        outputs=[
            {"state": {"fitness": {"mutation_eligible": False, "failure_classification": "TOOL_CAPABILITY_GAP"}}},
            {"state": {"fitness": {"mutation_eligible": False, "failure_classification": "NO_FAILURE"}}},
        ],
    )

    partition = example.DefaultActionabilityPolicy().partition(
        [{"input": "a"}, {"input": "b"}],
        evaluation,
        regression_guard_limit=1,
    )

    assert partition.optimization_indices == (0, 1)
    assert partition.fallback_to_unfiltered is True


def test_noop_aware_adapter_reuses_evaluation_when_selector_returns_no_components():
    example = _load_example_module()
    rollout_calls = 0

    def rollout(_candidate, row):
        nonlocal rollout_calls
        rollout_calls += 1
        return {"messages": [example.AIMessage(content=f"response for {row['input']}")]}

    adapter = example.NoOpAwareLangChainAdapter(
        rollout_fn=rollout,
        eval_fn=lambda _row, _state: (0.5, "diagnostic only"),
        num_threads=1,
        show_progress=False,
    )
    batch = [{"input": "example"}]
    candidate = {"memory:AGENTS.md": "Keep the current behavior."}

    current = adapter.evaluate(batch, candidate, capture_traces=True)
    reflective_dataset = adapter.make_reflective_dataset(candidate, current, [])
    reused = adapter.evaluate(batch, candidate, capture_traces=True)

    assert reflective_dataset == {}
    assert rollout_calls == 1
    assert reused.scores == current.scores
    assert reused.trajectories == current.trajectories
    assert reused.num_metric_calls == 0

    adapter.evaluate(batch, candidate, capture_traces=True)
    assert rollout_calls == 2


def test_reflective_dataset_deduplicates_repeated_records_and_shared_component_map():
    example = _load_example_module()

    def record(row, _state, score, feedback):
        return {
            "Runtime input": row["input"],
            "Score": score,
            "Feedback": feedback,
            "Project component map": {"memory:AGENTS.md": "shared"},
        }

    adapter = example.NoOpAwareLangChainAdapter(
        rollout_fn=lambda _candidate, _row: {"messages": []},
        eval_fn=lambda _row, _state: (0.5, "feedback"),
        reflective_record_fn=record,
        num_threads=1,
        show_progress=False,
    )
    repeated = {
        "data": {"input": "same"},
        "state": {"messages": []},
        "score": 0.5,
        "feedback": "feedback",
    }
    distinct = {
        "data": {"input": "different"},
        "state": {"messages": []},
        "score": 0.4,
        "feedback": "other feedback",
    }
    evaluation = example.EvaluationBatch(
        outputs=[],
        scores=[0.5, 0.5, 0.4],
        trajectories=[repeated, repeated, distinct],
    )

    dataset = adapter.make_reflective_dataset(
        {"memory:AGENTS.md": "shared"},
        evaluation,
        ["memory:AGENTS.md", "main:system_prompt"],
    )["memory:AGENTS.md"]

    assert len(dataset) == 2
    assert dataset[0]["Selected component bundle"] == ["memory:AGENTS.md", "main:system_prompt"]
    assert "Project component map" in dataset[0]
    assert "Project component map" not in dataset[1]


def test_effective_reflection_minibatch_is_limited_by_unique_optimization_pool():
    example = _load_example_module()
    rows = example.deduplicate_examples([{"input": "steel"}, {"input": "steel"}])

    assert rows == [{"input": "steel"}]
    assert example.effective_reflection_minibatch_size(3, rows) == 1


def test_noop_aware_adapter_reuses_trailing_whitespace_only_proposal():
    example = _load_example_module()
    rollout_calls = 0

    def rollout(_candidate, _row):
        nonlocal rollout_calls
        rollout_calls += 1
        return {"messages": [example.AIMessage(content="same behavior")]}

    adapter = example.NoOpAwareLangChainAdapter(
        rollout_fn=rollout,
        eval_fn=lambda _row, _state: (0.5, "same"),
        num_threads=1,
        show_progress=False,
    )
    batch = [{"input": "example"}]
    parent = {"memory:AGENTS.md": "Keep the current behavior.\n"}
    whitespace_only = {"memory:AGENTS.md": "Keep the current behavior."}

    current = adapter.evaluate(batch, parent, capture_traces=True)
    reused = adapter.evaluate(batch, whitespace_only, capture_traces=True)

    assert rollout_calls == 1
    assert reused.scores == current.scores
    assert reused.num_metric_calls == 0


def test_eval_caps_score_when_expected_route_is_missing(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path)
    candidate, surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)
    constraints = example.validate_candidate_constraints(candidate, candidate, surfaces)
    state = {
        "messages": [example.AIMessage(content="I will inspect the invoice file first.")],
        "baseline_response": "",
        "candidate_excerpt": candidate,
        "candidate_constraints": [constraint.__dict__ for constraint in constraints],
    }

    score, feedback = example.evaluate_response({"input": "I need my invoice.", "expected": "billing"}, state)

    assert score <= 0.40
    assert "- correctness_cap: 0.40" in feedback


def test_boundary_gate_blocks_skill_content_pasted_into_system_prompt(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path)
    candidate, surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)
    baseline_candidate = dict(candidate)
    candidate["main:system_prompt"] = candidate["skill:support-router:SKILL.md"]
    constraints = example.validate_candidate_constraints(candidate, baseline_candidate, surfaces)
    failures = [constraint.__dict__ for constraint in constraints if not constraint.passed]
    state = {
        "messages": [example.AIMessage(content="<route>billing</route>")],
        "baseline_response": "I need the invoice file path.",
        "candidate_excerpt": candidate,
        "candidate_constraints": failures,
    }

    score, feedback = example.evaluate_response({"input": "I need my invoice.", "expected": "billing"}, state)

    assert score == 0.0
    assert any(":boundary:" in failure["name"] for failure in failures)
    assert "- constraint_cap: 0.00" in feedback
    assert "- suggested_component: main:system_prompt" in feedback


def test_invalid_skill_frontmatter_is_a_zero_cap_runtime_gate(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path)
    candidate, surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)
    baseline_candidate = dict(candidate)
    skill_key = "skill:support-router:SKILL.md"
    candidate[skill_key] = "# Support Router\n\nRoute support requests."
    constraints = example.validate_candidate_constraints(candidate, baseline_candidate, surfaces)
    failures = [constraint.__dict__ for constraint in constraints if not constraint.passed]
    state = {
        "messages": [example.AIMessage(content="<route>billing</route>")],
        "baseline_response": "<route>billing</route>",
        "candidate_excerpt": candidate,
        "candidate_constraints": [constraint.__dict__ for constraint in constraints],
    }

    score, feedback = example.evaluate_response(
        {"input": "I need my invoice.", "expected": "billing"},
        state,
    )

    assert score == 0.0
    assert any(failure["name"] == f"{skill_key}:frontmatter" for failure in failures)
    assert any(failure["name"] == f"{skill_key}:frontmatter_yaml" for failure in failures)
    assert "- constraint_cap: 0.00" in feedback

    judge_calls = []
    judged_score, judged_feedback = example.evaluate_response_with_judge(
        {"input": "I need my invoice.", "expected": "billing"},
        state,
        lambda prompt: judge_calls.append(prompt) or '{"score": 1.0}',
    )

    assert judged_score == 0.0
    assert judge_calls == []
    assert "Reflection judge skipped" in judged_feedback


def test_boundary_gate_blocks_bare_candidate_key_in_component_text(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path)
    candidate, surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)
    baseline_candidate = dict(candidate)
    candidate["main:system_prompt"] = "main:system_prompt\nReturn <route>account</route> for password issues."

    constraints = example.validate_candidate_constraints(candidate, baseline_candidate, surfaces)
    failure = next(
        constraint for constraint in constraints if constraint.name == "main:system_prompt:boundary:no_component_labels"
    )
    state = {
        "messages": [example.AIMessage(content="<route>account</route>")],
        "baseline_response": "",
        "candidate_excerpt": candidate,
        "candidate_constraints": [constraint.__dict__ for constraint in constraints],
    }

    score, feedback = example.evaluate_response({"input": "I forgot my password.", "expected": "account"}, state)

    assert failure.passed is False
    assert score == 0.0
    assert "- constraint_cap: 0.00" in feedback


def test_judge_prompt_treats_expected_as_authoritative(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path)
    candidate, surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)
    constraints = example.validate_candidate_constraints(candidate, candidate, surfaces)
    state = {
        "messages": [example.AIMessage(content="You should inspect the server logs.")],
        "baseline_response": "<route>billing</route>",
        "candidate_excerpt": candidate,
        "candidate_constraints": [constraint.__dict__ for constraint in constraints],
    }

    prompt = example.build_judge_prompt(
        {"input": "The export button crashes with a 500 error.", "expected": "engineering"},
        state,
        deterministic_score=0.0,
        deterministic_feedback="missing expected route",
        failures=[],
    )

    assert "Expected is not `rubric-only`, treat it as the authoritative target" in prompt
    assert "Operational troubleshooting advice instead of the expected label is a failure" in prompt
    assert "Expected: engineering" in prompt


def test_judge_prompt_includes_expert_data_and_trace_expectations(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path)
    candidate, surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)
    constraints = example.validate_candidate_constraints(candidate, candidate, surfaces)
    state = {
        "messages": [
            example.AIMessage(content="tool_calls=[{'name': 'lookup_policy', 'args': {'query': '钢铁 行业 周期'}}]"),
            example.AIMessage(content="识别行业周期风险。"),
        ],
        "baseline_response": "",
        "candidate_excerpt": candidate,
        "candidate_constraints": [constraint.__dict__ for constraint in constraints],
    }

    prompt = example.build_judge_prompt(
        {
            "input": "华东钢铁集团有限公司",
            "data": "七、项目风险点\n1、钢铁行业周期性风险",
            "rubric": "评价相关数据获取和风险点覆盖。",
            "metadata": {
                "checkpoints": [{"label": "钢铁行业周期性风险", "keywords": ["行业周期"]}],
                "trace_expectations": [{"label": "行业周期信息获取", "tool_intent_keywords": ["行业", "周期"]}],
            },
        },
        state,
        deterministic_score=0.5,
        deterministic_feedback="partial",
        failures=[],
    )

    assert "Expert evaluation data:" in prompt
    assert "七、项目风险点" in prompt
    assert "Trace expectations:" in prompt
    assert "行业周期信息获取" in prompt
    assert "Do not grow SKILL.md into an industry catalog" in prompt
    assert "an unread reference edit cannot change behavior" in prompt
    assert "Partial evidence may support a correspondingly narrow risk finding" in prompt
    assert "cross_case_regression_risk" in prompt
    assert "those fields are hidden during rollout" in prompt


def test_adaptive_trace_summary_uses_llm_only_after_budget(monkeypatch):
    example = _load_example_module()
    messages = [example.AIMessage(content=f"message {index} " + "x" * 100) for index in range(50)]
    state = {"messages": messages}
    summary_prompts = []

    def summarizer(prompt):
        summary_prompts.append(prompt)
        return "模型生成的旧轨迹摘要"

    monkeypatch.setenv("GEPA_CONTEXT_WINDOW_TOKENS", "200000")
    monkeypatch.setenv("GEPA_TRACE_CONTEXT_RATIO", "0.12")
    full_summary = example.summarize_messages(state, summarizer=summarizer)

    assert "<trace_summary>" not in full_summary
    assert "message 0" in full_summary
    assert "message 49" in full_summary
    assert summary_prompts == []

    monkeypatch.setenv("GEPA_TRACE_MIN_CHARS", "300")
    monkeypatch.setenv("GEPA_TRACE_MAX_CHARS", "300")
    compressed = example.summarize_messages(state, summarizer=summarizer)

    assert "<trace_summary>" in compressed
    assert "模型生成的旧轨迹摘要" in compressed
    assert "<recent_trace>" in compressed
    assert summary_prompts
    assert "message 0" in summary_prompts[0]
    assert "...[message truncated]" not in summary_prompts[0]
    assert "...[AI message compressed]" not in summary_prompts[0]


def test_evaluation_trace_omits_file_mutation_noise_but_keeps_ai_messages():
    example = _load_example_module()
    state = {
        "messages": [
            example.AIMessage(
                content="先查询行业数据, 再整理结果。",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": "report.md", "content": "very noisy draft"},
                        "id": "write-1",
                        "type": "tool_call",
                    },
                    {
                        "name": "lookup_company_risk",
                        "args": {"company": "华东钢铁集团有限公司"},
                        "id": "lookup-1",
                        "type": "tool_call",
                    },
                ],
            ),
            ToolMessage(content="write succeeded with a large patch", tool_call_id="write-1"),
            ToolMessage(content="库存和债务数据", tool_call_id="lookup-1", name="lookup_company_risk"),
            example.AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "edit_file",
                        "args": {"file_path": "report.md", "old": "a", "new": "b"},
                        "id": "edit-1",
                        "type": "tool_call",
                    }
                ],
            ),
            example.AIMessage(content="最终识别出库存减值和短债压力。"),
        ]
    }

    trace = example.summarize_messages(state, max_chars=20_000)

    assert "先查询行业数据" in trace
    assert "lookup_company_risk" in trace
    assert '"company": "华东钢铁集团有限公司"' in trace
    assert "库存和债务数据" in trace
    assert "[no textual content]" in trace
    assert "最终识别出库存减值和短债压力" in trace
    assert "write_file" not in trace
    assert "edit_file" not in trace
    assert "very noisy draft" not in trace
    assert "write succeeded" not in trace


def test_trace_without_summarizer_is_not_character_truncated():
    example = _load_example_module()
    messages = [example.AIMessage(content=f"AI-{index} " + "x" * 300) for index in range(5)]

    trace = example.summarize_messages({"messages": messages}, max_chars=600)

    assert all(f"AI-{index}" in trace for index in range(5))
    assert "message truncated" not in trace
    assert "AI message compressed" not in trace
    assert len(trace) > 600


def test_prepared_trace_summary_is_cached_for_reflection_record():
    example = _load_example_module()
    state = {
        "messages": [example.AIMessage(content=f"AI-{index} " + "x" * 300) for index in range(5)],
        "fitness": {},
    }
    calls = []

    def summarizer(prompt):
        calls.append(prompt)
        return "保留旧轨迹中的关键查询和风险结论。"

    prepared = example.prepare_evaluation_trace(state, summarizer, max_chars=600)
    record = example.reflective_record({"input": "测试企业"}, state, 0.5, "feedback")

    assert state["evaluation_trace_mode"] == "llm_summary"
    assert record["Recent trace"] == prepared
    assert "保留旧轨迹中的关键查询和风险结论" in record["Recent trace"]
    assert len(calls) == 1


def test_reflective_record_removes_duplicate_large_sections():
    example = _load_example_module()
    response = "最终审批分析。"
    state = {
        "messages": [example.AIMessage(content=response)],
        "baseline_response": response,
        "fitness": {
            "composite": 0.5,
            "successful_tool_evidence": [{"name": "lookup", "result": "large result"}],
            "failed_tool_evidence": [],
            "trace_expectation_evidence": {"risk": ["large evidence"]},
        },
        "evaluation_trace_summary": "完整 AI 轨迹摘要。",
        "evaluation_trace_summary_budget": example.trace_prompt_char_budget(),
    }
    feedback = (
        "Scores:\n- final_score: 0.50\n\n"
        "Judge feedback:\nImprove the scoped rule.\n\n"
        "With candidate output:\n重复的最终审批分析。\n\n"
        "Baseline output:\n重复的 baseline。\n\n"
        "Adaptive trace summary:\n重复的完整轨迹。"
    )

    record = example.reflective_record({"input": "测试企业"}, state, 0.5, feedback)

    assert record["Feedback"].endswith("Improve the scoped rule.")
    assert "重复的最终审批分析" not in record["Feedback"]
    assert record["Baseline response"] == "[same as Agent response]"
    assert record["Recent trace"] == "完整 AI 轨迹摘要。"
    assert record["Fitness"] == {"composite": 0.5}


def test_trace_expectation_matching_uses_successful_tool_evidence_from_full_trace():
    example = _load_example_module()
    state = {
        "messages": [
            example.AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_industry_cycle",
                        "args": {"query": "钢铁行业周期和库存减值"},
                        "id": "industry-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="钢铁行业处于下行周期, 存货存在减值压力。",
                tool_call_id="industry-1",
                name="lookup_industry_cycle",
            ),
            *[example.AIMessage(content=f"后续普通消息 {index}") for index in range(40)],
        ],
        "capability_tools": [
            {
                "owner": "main",
                "name": "lookup_industry_cycle",
                "description": "查询钢铁行业周期、价格和库存减值信息。",
            }
        ],
    }
    row = {
        "metadata": {
            "trace_expectations": [{"label": "行业周期信息获取", "tool_intent_keywords": ["钢铁行业周期", "库存减值"]}]
        }
    }

    matched, missing, coverage = example.trace_expectation_results(row, state)

    assert matched == ["行业周期信息获取"]
    assert missing == []
    assert coverage == 1.0


def test_trace_expectation_rejects_ai_prose_and_failed_tool_results():
    example = _load_example_module()
    row = {
        "metadata": {"trace_expectations": [{"label": "司法工商信息获取", "tool_intent_keywords": ["司法", "被执行"]}]}
    }
    state = {
        "messages": [
            example.AIMessage(content="已查询司法和被执行信息。"),
            example.AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_judicial",
                        "args": {"query": "司法 被执行"},
                        "id": "judicial-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="ERROR: upstream service unavailable",
                tool_call_id="judicial-1",
                name="lookup_judicial",
                status="error",
            ),
        ],
        "capability_tools": [
            {
                "owner": "main",
                "name": "lookup_judicial",
                "description": "查询企业司法、被执行和诉讼信息。",
            }
        ],
    }

    diagnostics = example.data_acquisition_diagnostics(row, state)

    assert diagnostics["matched_trace_expectations"] == []
    assert diagnostics["missing_trace_expectations"] == ["司法工商信息获取"]
    assert diagnostics["tool_supported_missing_expectations"] == ["司法工商信息获取"]
    assert diagnostics["tool_capability_gaps"] == []
    assert diagnostics["successful_tool_evidence"] == []
    assert diagnostics["failed_tool_evidence"][0]["name"] == "lookup_judicial"
    assert diagnostics["failed_tool_expectations"] == ["司法工商信息获取"]
    assert diagnostics["failed_tool_runtime_expectations"] == ["司法工商信息获取"]
    assert diagnostics["failed_tool_invocation_expectations"] == []


def test_no_data_tool_result_is_incomplete_even_when_message_status_is_success():
    example = _load_example_module()
    row = {
        "input": "不存在企业",
        "rubric": "覆盖司法执行风险。",
        "metadata": {
            "checkpoints": [
                {
                    "label": "司法执行风险",
                    "keywords": ["司法执行", "被执行"],
                    "evidence_expectations": ["司法工商信息获取"],
                }
            ],
            "trace_expectations": [
                {
                    "label": "司法工商信息获取",
                    "tool_names": ["lookup_judicial"],
                    "tool_intent_keywords": ["司法", "被执行"],
                }
            ],
        },
    }
    state = {
        "messages": [
            example.AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_judicial",
                        "args": {"company": "不存在企业"},
                        "id": "judicial-no-data-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="ERROR: 未找到企业记录",
                tool_call_id="judicial-no-data-1",
                name="lookup_judicial",
                status="success",
            ),
        ],
        "capability_tools": [{"owner": "main", "name": "lookup_judicial", "description": "查询企业司法和被执行信息。"}],
        "baseline_response": "",
        "candidate_excerpt": {"main:tool:lookup_judicial:description": "查询企业司法和被执行信息。"},
        "candidate_constraints": [],
    }

    diagnostics = example.data_acquisition_diagnostics(row, state)

    assert diagnostics["matched_trace_expectations"] == []
    assert diagnostics["incomplete_tool_result_expectations"] == ["司法工商信息获取"]
    assert diagnostics["tool_data_coverage_gaps"] == ["司法工商信息获取"]
    assert diagnostics["failed_tool_runtime_expectations"] == []
    assert diagnostics["failed_tool_evidence"][0]["status"] == "no_data"

    _score, feedback = example.evaluate_response(row, state)

    assert state["fitness"]["mutation_eligible"] is False
    assert state["fitness"]["failure_classification"] == "TOOL_CAPABILITY_GAP"
    assert state["fitness"]["remediation_type"] == "ADD_TOOL_OR_MCP"
    assert "- suggested_component: none" in feedback


def test_data_acquisition_diagnostics_distinguishes_missing_tool_from_skipped_tool():
    example = _load_example_module()
    row = {
        "metadata": {
            "trace_expectations": [
                {"label": "环保安监信息获取", "tool_intent_keywords": ["环保", "安全生产"]},
                {"label": "债务结构信息获取", "tool_intent_keywords": ["负债", "借款"]},
            ]
        }
    }
    state = {
        "messages": [example.AIMessage(content="只输出了结论, 没有调用工具。")],
        "available_tools": [
            {
                "owner": "main",
                "name": "lookup_debt",
                "description": "查询企业负债、借款、票据和融资结构。",
            }
        ],
    }

    diagnostics = example.data_acquisition_diagnostics(row, state)

    assert diagnostics["tool_supported_missing_expectations"] == ["债务结构信息获取"]
    assert diagnostics["skipped_supported_expectations"] == ["债务结构信息获取"]
    assert diagnostics["tool_capability_gaps"] == ["环保安监信息获取"]


def test_skipped_supported_tool_is_mutation_eligible_execution_lapse():
    example = _load_example_module()
    row = {
        "input": "示例企业",
        "rubric": "调用现有工具取得债务结构信息。",
        "metadata": {"trace_expectations": [{"label": "债务结构信息获取", "tool_intent_keywords": ["负债", "借款"]}]},
    }
    state = {
        "messages": [example.AIMessage(content="当前无法判断债务结构。")],
        "baseline_response": "",
        "candidate_excerpt": {"memory:AGENTS.md": "Use available tools before answering."},
        "candidate_constraints": [],
        "available_tools": [
            {
                "owner": "main",
                "name": "lookup_debt",
                "description": "查询企业负债、借款、票据和融资结构。",
            }
        ],
    }

    _score, feedback = example.evaluate_response(row, state)

    assert state["fitness"]["mutation_eligible"] is True
    assert state["fitness"]["failure_classification"] == "EXECUTION_LAPSE"
    assert state["fitness"]["remediation_type"] == "IMPROVE_TOOL_USAGE"
    assert "- suggested_component: memory:AGENTS.md" in feedback


def test_policy_topic_overlap_does_not_claim_borrower_data_capability():
    example = _load_example_module()
    row = {
        "metadata": {
            "trace_expectations": [
                {
                    "label": "财务现金流信息获取",
                    "tool_intent_keywords": ["财务", "现金流", "银行流水"],
                }
            ]
        }
    }
    state = {
        "messages": [example.AIMessage(content="尚未取得企业现金流数据。")],
        "available_tools": [
            {
                "owner": "main",
                "name": "lookup_policy",
                "description": "按主题查询内部信贷审查政策; 主题可为现金流、抵质押、保证或行业.",
            }
        ],
    }

    diagnostics = example.data_acquisition_diagnostics(row, state)

    assert diagnostics["tool_supported_missing_expectations"] == []
    assert diagnostics["tool_capability_gaps"] == ["财务现金流信息获取"]


def test_undeclared_task_delegation_does_not_claim_data_capability():
    example = _load_example_module()
    row = {
        "metadata": {
            "trace_expectations": [
                {
                    "label": "集团穿透信息获取",
                    "tool_intent_keywords": ["集团", "子公司", "担保"],
                }
            ]
        }
    }
    state = {
        "messages": [
            example.AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {"description": "查询集团子公司和担保信息"},
                        "id": "task-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="当前没有企业数据查询工具, 无法取得集团穿透信息。",
                tool_call_id="task-1",
                name="task",
            ),
        ],
        "capability_tools": [
            {
                "owner": "main",
                "name": "lookup_policy",
                "description": "仅按主题返回内部通用审查政策, 不查询具体企业事实。",
            }
        ],
    }

    diagnostics = example.data_acquisition_diagnostics(row, state)

    assert diagnostics["tool_supported_missing_expectations"] == []
    assert diagnostics["incomplete_tool_result_expectations"] == []
    assert diagnostics["tool_capability_gaps"] == ["集团穿透信息获取"]


def test_successful_explicit_tool_with_empty_payload_is_incomplete_not_matched():
    example = _load_example_module()
    row = {
        "metadata": {
            "trace_expectations": [
                {
                    "label": "客户交易信息获取",
                    "tool_names": ["lookup_customers"],
                    "tool_intent_keywords": ["客户", "回款"],
                }
            ]
        }
    }
    state = {
        "messages": [
            example.AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_customers",
                        "args": {"company": "示例企业"},
                        "id": "customers-empty-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="暂无可用数据",
                tool_call_id="customers-empty-1",
                name="lookup_customers",
            ),
        ],
        "capability_tools": [{"owner": "main", "name": "lookup_customers", "description": "查询客户集中和回款信息。"}],
    }

    diagnostics = example.data_acquisition_diagnostics(row, state)

    assert diagnostics["matched_trace_expectations"] == []
    assert diagnostics["incomplete_tool_result_expectations"] == ["客户交易信息获取"]


def test_unrelated_successful_tool_does_not_unlock_missing_checkpoint_mutation():
    example = _load_example_module()
    row = {
        "input": "示例企业",
        "rubric": "覆盖客户集中风险。",
        "metadata": {
            "checkpoints": [
                {
                    "label": "客户集中风险",
                    "keywords": ["客户集中"],
                    "evidence_expectations": ["客户交易信息获取"],
                }
            ],
            "trace_expectations": [
                {
                    "label": "财务信息获取",
                    "tool_names": ["lookup_financial"],
                    "tool_intent_keywords": ["财务", "利润"],
                },
                {
                    "label": "客户交易信息获取",
                    "tool_names": ["lookup_customers"],
                    "tool_intent_keywords": ["客户", "回款"],
                },
            ],
        },
    }
    state = {
        "messages": [
            example.AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_financial",
                        "args": {"company": "示例企业"},
                        "id": "financial-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="净利润增长10%。",
                tool_call_id="financial-1",
                name="lookup_financial",
            ),
            example.AIMessage(content="盈利保持增长。"),
        ],
        "baseline_response": "",
        "candidate_excerpt": {"skill:risk:reference/learned.md": "# Learned"},
        "candidate_constraints": [],
        "capability_tools": [{"owner": "main", "name": "lookup_financial", "description": "查询企业财务、利润信息。"}],
    }

    _score, feedback = example.evaluate_response(row, state)

    assert state["fitness"]["matched_trace_expectations"] == ["财务信息获取"]
    assert state["fitness"]["tool_blocked_missing_checkpoints"] == ["客户集中风险"]
    assert state["fitness"]["mutation_eligible"] is False
    assert state["fitness"]["failure_classification"] == "TOOL_CAPABILITY_GAP"
    assert "- suggested_component: none" in feedback


def test_explicit_financial_scope_limit_unlocks_information_asymmetry_checkpoint():
    example = _load_example_module()
    row = {
        "input": "华东钢铁集团有限公司",
        "metadata": {
            "checkpoints": [
                {
                    "label": "集团内部信息不对称风险",
                    "keywords": ["信息不对称"],
                    "evidence_expectations": ["集团穿透信息获取", "财务口径限制信息获取"],
                    "evidence_mode": "any",
                }
            ],
            "trace_expectations": [
                {
                    "label": "集团穿透信息获取",
                    "tool_intent_keywords": ["集团", "子公司", "担保"],
                },
                {
                    "label": "财务口径限制信息获取",
                    "tool_names": ["lookup_financial_snapshot"],
                    "tool_intent_keywords": ["数据口径", "数据限制"],
                },
            ],
        },
    }
    response = "仅有集团合并口径且未取得子公司单体明细, 存在集团内部信息不对称风险。"
    state = {
        "messages": [
            example.AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_financial_snapshot",
                        "args": {"company_name": "华东钢铁集团有限公司"},
                        "id": "financial-scope-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content='{"数据口径":"集团合并口径","数据限制":"未取得子公司单体财务和担保明细"}',
                tool_call_id="financial-scope-1",
                name="lookup_financial_snapshot",
            ),
            example.AIMessage(content=response),
        ],
        "capability_tools": [
            {
                "owner": "main",
                "name": "lookup_financial_snapshot",
                "description": "查询企业财务快照以及数据口径和数据限制。",
            }
        ],
    }

    acquisition = example.data_acquisition_diagnostics(row, state)
    matched, missing, coverage = example.rubric_checkpoint_results(row, response, acquisition)

    assert acquisition["matched_trace_expectations"] == ["财务口径限制信息获取"]
    assert acquisition["tool_capability_gaps"] == ["集团穿透信息获取"]
    assert matched == ["集团内部信息不对称风险"]
    assert missing == []
    assert coverage == 1.0


def test_chinese_keyword_matching_ignores_full_width_punctuation_and_spacing():
    example = _load_example_module()

    assert example.checkpoint_matches(
        "企业经营性\u3001现金流连续下降\uff0c短期偿债压力上升。",
        {"label": "现金流风险", "keywords": ["经营性现金流"]},
    )


def test_missing_data_tool_is_classified_as_tool_capability_gap():
    example = _load_example_module()
    row = {
        "input": "华东钢铁集团有限公司",
        "rubric": "核验环保处罚信息。",
        "metadata": {
            "trace_expectations": [{"label": "环保安监信息获取", "tool_intent_keywords": ["环保", "安全生产"]}]
        },
    }
    state = {
        "messages": [example.AIMessage(content="目前缺少环保处罚数据。")],
        "baseline_response": "",
        "candidate_excerpt": {"skill:credit-risk-review:SKILL.md": "Review available evidence."},
        "candidate_constraints": [],
        "available_tools": [{"owner": "main", "name": "lookup_policy", "description": "查询内部授信政策。"}],
    }

    _score, feedback = example.evaluate_response(row, state)

    assert state["fitness"]["failure_classification"] == "TOOL_CAPABILITY_GAP"
    assert state["fitness"]["tool_capability_gaps"] == ["环保安监信息获取"]
    assert state["fitness"]["remediation_type"] == "ADD_TOOL_OR_MCP"
    assert "- failure_classification: TOOL_CAPABILITY_GAP" in feedback
    assert "- suggested_component: none" in feedback


def test_failed_tool_runtime_is_reported_without_text_mutation():
    example = _load_example_module()
    row = {
        "input": "示例企业",
        "rubric": "覆盖司法执行风险。",
        "metadata": {
            "checkpoints": [
                {
                    "label": "司法执行风险",
                    "keywords": ["司法执行", "被执行"],
                    "evidence_expectations": ["司法工商信息获取"],
                }
            ],
            "trace_expectations": [
                {
                    "label": "司法工商信息获取",
                    "tool_names": ["lookup_judicial"],
                    "tool_intent_keywords": ["司法", "被执行"],
                }
            ],
        },
    }
    state = {
        "messages": [
            example.AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_judicial",
                        "args": {"company": "示例企业"},
                        "id": "judicial-runtime-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="ERROR: upstream service unavailable",
                tool_call_id="judicial-runtime-1",
                name="lookup_judicial",
                status="error",
            ),
            example.AIMessage(content="现有信息不足以识别具体风险点。"),
        ],
        "baseline_response": "",
        "candidate_excerpt": {
            "skill:credit-risk-review:SKILL.md": "Use the judicial tool when relevant.",
            "main:tool:lookup_judicial:description": "查询企业司法执行信息。",
        },
        "candidate_constraints": [],
        "available_tools": [
            {
                "owner": "main",
                "name": "lookup_judicial",
                "description": "查询企业司法执行和被执行信息。",
            }
        ],
    }

    _score, feedback = example.evaluate_response(row, state)

    assert state["fitness"]["failure_classification"] == "EXECUTION_LAPSE"
    assert state["fitness"]["mutation_eligible"] is False
    assert state["fitness"]["remediation_type"] == "FIX_TOOL_RUNTIME"
    assert "- suggested_component: none" in feedback
    assert "FIX_TOOL_RUNTIME" in feedback


def test_failed_tool_arguments_prefer_tool_description_mutation():
    example = _load_example_module()
    row = {
        "input": "示例企业",
        "rubric": "覆盖司法执行风险。",
        "metadata": {
            "checkpoints": [
                {
                    "label": "司法执行风险",
                    "keywords": ["司法执行", "被执行"],
                    "evidence_expectations": ["司法工商信息获取"],
                }
            ],
            "trace_expectations": [
                {
                    "label": "司法工商信息获取",
                    "tool_names": ["lookup_judicial"],
                    "tool_intent_keywords": ["司法", "被执行"],
                }
            ],
        },
    }
    state = {
        "messages": [
            example.AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_judicial",
                        "args": {"name": "示例企业"},
                        "id": "judicial-args-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="Validation error: missing required argument company_id",
                tool_call_id="judicial-args-1",
                name="lookup_judicial",
                status="error",
            ),
            example.AIMessage(content="现有信息不足以识别具体风险点。"),
        ],
        "baseline_response": "",
        "candidate_excerpt": {
            "skill:credit-risk-review:SKILL.md": "Use the judicial tool when relevant.",
            "main:tool:lookup_judicial:description": "查询企业司法执行信息。",
        },
        "candidate_constraints": [],
        "available_tools": [
            {
                "owner": "main",
                "name": "lookup_judicial",
                "description": "使用 company_id 查询企业司法执行和被执行信息。",
            }
        ],
    }

    _score, feedback = example.evaluate_response(row, state)

    assert state["fitness"]["mutation_eligible"] is True
    assert state["fitness"]["remediation_type"] == "IMPROVE_TOOL_INVOCATION"
    assert "- suggested_component: main:tool:lookup_judicial:description" in feedback


def test_checkpoint_with_unavailable_tool_does_not_select_learned_reference():
    example = _load_example_module()
    config_path = (
        Path(__file__).parents[1]
        / "examples"
        / "langchain_adapter"
        / "deepagents_gepa_configs"
        / "credit_approval.toml"
    )
    project = example.build_candidate_from_deep_agent_project(example.load_deepagents_gepa_config(config_path))
    row = {
        "input": "示例制造企业",
        "rubric": "覆盖客户集中风险并取得客户交易证据。",
        "metadata": {
            "checkpoints": [{"label": "客户集中风险", "keywords": ["客户集中", "回款集中"]}],
            "trace_expectations": [
                {
                    "label": "客户交易信息获取",
                    "tool_intent_keywords": ["客户", "合同", "回款"],
                }
            ],
        },
    }
    state = {
        "messages": [example.AIMessage(content="当前资料不足, 暂不作结论.")],
        "baseline_response": "",
        "candidate_excerpt": project.candidate,
        "candidate_constraints": [],
        "available_tools": [
            {
                "owner": "main",
                "name": "lookup_policy",
                "description": "查询内部授信政策。",
            }
        ],
    }

    _score, feedback = example.evaluate_response(row, state)

    assert state["fitness"]["failure_classification"] == "TOOL_CAPABILITY_GAP"
    assert state["fitness"]["mutation_eligible"] is False
    assert state["fitness"]["tool_capability_gaps"] == ["客户交易信息获取"]
    assert "- suggested_component: none" in feedback
    assert "- mutation_eligible: false" in feedback


def test_reflection_judge_cannot_override_unavailable_tool_gap_into_text_mutation():
    example = _load_example_module()
    config_path = (
        Path(__file__).parents[1]
        / "examples"
        / "langchain_adapter"
        / "deepagents_gepa_configs"
        / "credit_approval.toml"
    )
    project = example.build_candidate_from_deep_agent_project(example.load_deepagents_gepa_config(config_path))
    row = {
        "input": "示例制造企业",
        "rubric": "覆盖客户集中风险并取得客户交易证据。",
        "metadata": {
            "checkpoints": [{"label": "客户集中风险", "keywords": ["客户集中", "回款集中"]}],
            "trace_expectations": [
                {
                    "label": "客户交易信息获取",
                    "tool_intent_keywords": ["客户", "合同", "回款"],
                }
            ],
        },
    }
    state = {
        "messages": [example.AIMessage(content="当前资料不足, 暂不作结论.")],
        "baseline_response": "",
        "candidate_excerpt": project.candidate,
        "candidate_constraints": [],
        "available_tools": [
            {
                "owner": "main",
                "name": "lookup_policy",
                "description": "查询内部授信政策。",
            }
        ],
    }

    _score, feedback = example.evaluate_response_with_judge(
        row,
        state,
        lambda _prompt: json.dumps(
            {
                "score": 0.1,
                "failure_classification": "TOOL_CAPABILITY_GAP",
                "classification_reason": "missing customer data tool",
                "suggested_component": "",
                "suggested_component_reason": "no text component can retrieve data",
                "feedback": "add a customer data tool",
                "boundary_assessment": "ok",
            }
        ),
    )

    assert state["fitness"]["failure_classification"] == "TOOL_CAPABILITY_GAP"
    assert state["fitness"]["mutation_eligible"] is False
    assert "- suggested_component: none" in feedback
    assert "- mutation_eligible: false" in feedback


def test_runtime_tool_evidence_can_make_missing_expert_logic_text_actionable():
    example = _load_example_module()
    config_path = (
        Path(__file__).parents[1]
        / "examples"
        / "langchain_adapter"
        / "deepagents_gepa_configs"
        / "credit_approval.toml"
    )
    project = example.build_candidate_from_deep_agent_project(example.load_deepagents_gepa_config(config_path))
    row = {
        "input": "示例制造企业",
        "rubric": "识别客户集中导致的回款和流动性风险。",
        "metadata": {
            "checkpoints": [
                {
                    "label": "客户集中风险",
                    "keywords": ["客户集中", "流动性风险"],
                    "evidence_expectations": ["客户交易信息获取"],
                }
            ],
            "trace_expectations": [
                {
                    "label": "客户交易信息获取",
                    "tool_names": ["lookup_customers"],
                    "tool_intent_keywords": ["客户", "回款"],
                }
            ],
        },
    }
    state = {
        "messages": [
            example.AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_customers",
                        "args": {"company": "示例制造企业"},
                        "id": "customers-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="前五大客户占比较高, 近期回款周期延长。",
                tool_call_id="customers-1",
                name="lookup_customers",
            ),
            example.AIMessage(content="当前资料需要进一步核验。"),
        ],
        "baseline_response": "",
        "candidate_excerpt": project.candidate,
        "candidate_constraints": [],
        "available_tools": [
            {
                "owner": "main",
                "name": "lookup_customers",
                "description": "查询客户集中度和回款情况。",
            }
        ],
    }

    _score, feedback = example.evaluate_response(row, state)

    assert state["fitness"]["mutation_eligible"] is True
    assert state["fitness"]["failure_classification"] == "SKILL_DEFECT"
    assert state["fitness"]["remediation_type"] == "IMPROVE_SKILL_OR_REFERENCE"
    assert "- suggested_component: skill:credit-risk-review:reference/learned_expert_patterns.md" in feedback


def test_skill_defect_remains_primary_when_a_case_also_has_tool_gaps():
    example = _load_example_module()

    diagnostics = example.remediation_diagnostics(
        "SKILL_DEFECT",
        [],
        {
            "runtime_supported_missing_checkpoints": ["盈利质量风险"],
            "matched_trace_expectations": ["财务盈利质量信息获取"],
            "tool_capability_gaps": ["行业周期信息获取"],
        },
    )

    assert diagnostics["remediation_type"] == "IMPROVE_SKILL_OR_REFERENCE"
    assert {item["type"] for item in diagnostics["remediation_actions"]} == {
        "IMPROVE_SKILL_OR_REFERENCE",
        "ADD_TOOL_OR_MCP",
    }


def test_expected_mapping_diagnosis_distinguishes_missing_rule_from_execution_lapse():
    example = _load_example_module()
    candidate = {
        "skill:support-router:reference/routing.md": (
            "Billing covers invoices. Account covers login and passwords. "
            "Product covers feature requests and integrations."
        )
    }
    base_fitness = {"hard": 0.0, "mutation_eligible": True}

    profile_class, _reason = example.classify_failure(
        {
            "expected": "account",
            "rubric": "Requests about profile ownership or account identity must route to account.",
        },
        {"candidate_excerpt": candidate},
        "<route>product</route>",
        [],
        dict(base_fitness),
    )
    salesforce_class, _reason = example.classify_failure(
        {
            "expected": "product",
            "rubric": "Salesforce integration requests must route to product.",
        },
        {"candidate_excerpt": candidate},
        "No route tag.",
        [],
        dict(base_fitness),
    )

    assert profile_class == "SKILL_DEFECT"
    assert salesforce_class == "EXECUTION_LAPSE"


def test_judge_guidance_removes_case_derived_thresholds_but_keeps_source_rules():
    example = _load_example_module()
    payload = {
        "reusable_lesson": "净利率低于2%且现金流覆盖低于10%时必须预警。",
        "operational_rule": "政策明确要求资产负债率不得超过70%。",
    }

    sanitized = example.sanitize_judge_guidance(
        payload,
        {"data": "该企业净利率为2%。审批政策明确要求资产负债率不得超过70%。"},
    )

    assert "2%" not in sanitized["reusable_lesson"]
    assert "10%" not in sanitized["reusable_lesson"]
    assert "70%" in sanitized["operational_rule"]
    assert sanitized["guidance_sanitization"]


def test_reflection_judge_score_is_capped_when_expected_route_is_missing(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path)
    candidate, surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)
    constraints = example.validate_candidate_constraints(candidate, candidate, surfaces)
    state = {
        "messages": [example.AIMessage(content="Please provide logs and reproduction steps.")],
        "baseline_response": "I also need more details.",
        "candidate_excerpt": candidate,
        "candidate_constraints": [constraint.__dict__ for constraint in constraints],
    }

    score, feedback = example.evaluate_response_with_judge(
        {"input": "The export button crashes with a 500 error.", "expected": "engineering"},
        state,
        lambda _prompt: json.dumps(
            {
                "score": 0.95,
                "failure_classification": "EXECUTION_LAPSE",
                "classification_reason": "missing route",
                "suggested_component": "memory:AGENTS.md",
                "suggested_component_reason": "reinforce output contract",
                "feedback": "Return the expected route tag.",
                "boundary_assessment": "ok",
            }
        ),
    )

    assert score == 0.40
    assert "- judge_score: 0.95" in feedback
    assert "- correctness_cap: 0.40" in feedback
    assert "- final_cap: 0.40" in feedback


def test_reflection_judge_clears_suggestion_for_no_failure(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path)
    candidate, surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)
    constraints = example.validate_candidate_constraints(candidate, candidate, surfaces)
    state = {
        "messages": [example.AIMessage(content="<route>billing</route>")],
        "baseline_response": "<route>billing</route>",
        "candidate_excerpt": candidate,
        "candidate_constraints": [constraint.__dict__ for constraint in constraints],
    }

    score, feedback = example.evaluate_response_with_judge(
        {"input": "Where is my invoice?", "expected": "billing"},
        state,
        lambda _prompt: json.dumps(
            {
                "score": 1.0,
                "failure_classification": "NO_FAILURE",
                "classification_reason": "correct route",
                "suggested_component": "skill:support-router:reference/routing.md",
                "suggested_component_reason": "unnecessary refinement",
                "feedback": "correct",
                "boundary_assessment": "ok",
            }
        ),
    )

    assert score == 1.0
    assert state["fitness"]["failure_classification"] == "NO_FAILURE"
    assert state["fitness"]["mutation_eligible"] is False
    assert state["fitness"]["score_source"] == "deterministic_expected"
    assert "- suggested_component: none" in feedback


def test_reflection_judge_cannot_zero_an_exact_authoritative_target(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path)
    candidate, surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)
    constraints = example.validate_candidate_constraints(candidate, candidate, surfaces)
    state = {
        "messages": [example.AIMessage(content="<route>Engineering</route>")],
        "baseline_response": "<route>engineering</route>",
        "candidate_excerpt": candidate,
        "candidate_constraints": [constraint.__dict__ for constraint in constraints],
    }

    score, feedback = example.evaluate_response_with_judge(
        {
            "input": "The export button crashes with a 500 error.",
            "expected": "engineering",
        },
        state,
        lambda _prompt: json.dumps(
            {
                "score": 0.0,
                "failure_classification": "EXECUTION_LAPSE",
                "classification_reason": "inconsistent judge output",
                "suggested_component": "memory:AGENTS.md",
                "suggested_component_reason": "unnecessary",
                "feedback": "The route is actually correct.",
                "boundary_assessment": "ok",
            }
        ),
    )

    assert score == 1.0
    assert state["fitness"]["judge_score"] == 0.0
    assert state["fitness"]["score_source"] == "deterministic_expected"
    assert state["fitness"]["failure_classification"] == "NO_FAILURE"
    assert state["fitness"]["mutation_eligible"] is False
    assert "- score_source: deterministic_expected" in feedback
    assert "- suggested_component: none" in feedback


def test_reflection_judge_score_is_capped_by_missing_rubric_checkpoints(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path)
    candidate, surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)
    constraints = example.validate_candidate_constraints(candidate, candidate, surfaces)
    state = {
        "messages": [example.AIMessage(content="该企业现金回款弱化, 放款前需要核验应收账款账龄。")],
        "baseline_response": "Generic approval review.",
        "candidate_excerpt": candidate,
        "candidate_constraints": [constraint.__dict__ for constraint in constraints],
    }
    rubric_example = {
        "input": "企业存在负经营现金流、关联方应收款和自评抵押物。",
        "rubric": "审批专家意见: 覆盖现金回款、关联方应收可回收性和抵押物独立评估。",
        "metadata": {
            "checkpoints": [
                {"label": "现金回款弱化", "keywords": ["现金回款弱化", "负经营现金流"]},
                {"label": "关联方应收可回收性", "keywords": ["关联方应收", "可回收性"]},
                {"label": "抵押物独立评估", "keywords": ["独立评估", "第三方评估"]},
            ]
        },
    }

    score, feedback = example.evaluate_response_with_judge(
        rubric_example,
        state,
        lambda _prompt: json.dumps(
            {
                "score": 1.0,
                "failure_classification": "NO_FAILURE",
                "classification_reason": "looks strong",
                "suggested_component": "skill:support-router:SKILL.md",
                "suggested_component_reason": "n/a",
                "feedback": "n/a",
                "boundary_assessment": "ok",
            }
        ),
    )

    assert score == 0.45
    assert "- rubric_cap: 0.45" in feedback
    assert "- rubric_coverage: 0.33" in feedback
    assert "- failure_classification: INSUFFICIENT_RUNTIME_EVIDENCE" in feedback
    assert "- mutation_eligible: false" in feedback
    assert "- suggested_component: none" in feedback
    assert "关联方应收可回收性" in feedback
    assert "抵押物独立评估" in feedback


def test_rubric_checkpoint_matching_supports_chinese_keywords():
    example = _load_example_module()
    row = {
        "metadata": {
            "checkpoints": [
                {"label": "现金回款弱化", "keywords": ["经营性现金流连续两年为负", "现金回款弱化"]},
                {"label": "动产抵押顺位核查", "keywords": ["动产抵押", "抵押顺位"]},
            ]
        }
    }
    response = "该企业经营性现金流连续两年为负, 且新增动产抵押, 需核查抵押顺位。"

    matched, missing, coverage = example.rubric_checkpoint_results(row, response)

    assert matched == ["现金回款弱化", "动产抵押顺位核查"]
    assert missing == []
    assert coverage == 1.0


def test_rubric_checkpoint_matching_rejects_negated_or_unknown_mentions():
    example = _load_example_module()
    row = {
        "metadata": {
            "checkpoints": [
                {"label": "食品安全处罚", "keywords": ["食品安全处罚"]},
                {"label": "仓单重复质押", "keywords": ["重复质押"]},
            ]
        }
    }
    response = "未取得食品安全处罚记录, 没有发现食品安全处罚记录, 无法排除历史食品安全处罚。重复质押情况未知。"

    matched, missing, coverage = example.rubric_checkpoint_results(row, response)
    unsupported = example.unsupported_checkpoint_mentions(row, response)

    assert matched == []
    assert missing == ["食品安全处罚", "仓单重复质押"]
    assert coverage == 0.0
    assert unsupported == [
        {"label": "食品安全处罚", "reason": "negated_or_unknown_mention"},
        {"label": "仓单重复质押", "reason": "negated_or_unknown_mention"},
    ]


def test_rubric_checkpoint_matching_keeps_negative_enterprise_facts_as_evidence():
    example = _load_example_module()
    row = {
        "metadata": {
            "checkpoints": [
                {"label": "担保覆盖不足", "keywords": ["担保"]},
                {"label": "还款逾期", "keywords": ["按期还款"]},
            ]
        }
    }
    response = "企业缺少强担保覆盖, 且未按期还款。"

    matched, missing, coverage = example.rubric_checkpoint_results(row, response)

    assert matched == ["担保覆盖不足", "还款逾期"]
    assert missing == []
    assert coverage == 1.0


def test_rubric_checkpoint_requires_declared_runtime_evidence_when_available():
    example = _load_example_module()
    row = {
        "metadata": {
            "checkpoints": [
                {
                    "label": "食品安全处罚",
                    "keywords": ["食品安全处罚"],
                    "evidence_expectations": ["行政处罚"],
                }
            ]
        }
    }
    response = "企业近两年存在食品安全处罚, 可能影响经营连续性。"

    unmatched = example.rubric_checkpoint_results(
        row,
        response,
        {"matched_trace_expectations": []},
    )
    matched = example.rubric_checkpoint_results(
        row,
        response,
        {"matched_trace_expectations": ["行政处罚"]},
    )

    assert unmatched == ([], ["食品安全处罚"], 0.0)
    assert matched == (["食品安全处罚"], [], 1.0)


def test_strict_metric_budget_shrinks_minibatch_and_stops_before_overshoot():
    example = _load_example_module()
    rows = [{"input": f"case-{index}"} for index in range(4)]

    minibatch = example.strict_budgeted_reflection_minibatch_size(
        3,
        rows,
        valset_size=4,
        metric_budget=10,
    )
    stopper = example.StrictProposalMetricBudgetStopper(
        max_metric_calls=10,
        valset_size=4,
        reflection_minibatch_size=minibatch,
    )

    assert minibatch == 1
    assert stopper(SimpleNamespace(total_num_evals=4, program_candidates=[{}])) is False
    assert stopper(SimpleNamespace(total_num_evals=10, program_candidates=[{}, {}])) is True
    assert example.strict_budgeted_reflection_minibatch_size(
        3,
        rows,
        valset_size=4,
        metric_budget=9,
    ) == 0


def test_growth_limit_is_advisory_not_hard_gate(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path)
    candidate, surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)
    baseline_candidate = dict(candidate)
    candidate["main:system_prompt"] = candidate["main:system_prompt"] + "\n" + ("Use the existing skill. " * 50)

    constraints = example.validate_candidate_constraints(candidate, baseline_candidate, surfaces)
    growth = next(constraint for constraint in constraints if constraint.name == "main:system_prompt:growth_limit")
    state = {
        "candidate_constraints": [constraint.__dict__ for constraint in constraints],
    }

    assert growth.passed is False
    assert growth.severity == "advisory"
    assert not example.hard_constraint_failures(state)


def test_memory_reflection_template_discourages_copying_skill_content(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path)
    candidate, _surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)

    template = example.reflection_prompt_templates(candidate)["memory:AGENTS.md"]

    assert "Do not copy SKILL.md" in template
    assert "reference/*.md" in template
    assert "AGENTS.md does not own domain methodology" in template
    assert "change only how AGENTS.md makes the existing resource reachable" in template
    assert "Do not encode domain heuristics" in template
    assert "Optimize only the selected component" in template
    assert "rejected proposal lessons" in template
    assert "Proposal rationale" in template
    assert "Component boundary rules" in template
    assert "Selected component" in template
    assert "missing_rationale" in template
    assert "Do not start with a fenced code block" in template
    assert "Expert data, rubrics, checkpoints" in template
    assert "under <side_info> is optimizer-only evidence" in template
    assert "A TOOL_CAPABILITY_GAP means no current text component" in template
    assert "Hidden-data boundary check" in template
    assert "Applicability scope and exclusions" in template
    assert "Cross-case regression risk" in template
    assert "short signal -> concern -> consequence reminder" in template
    assert "do not expand every reminder into a fixed multi-section template" in template
    assert "Do not force every lesson into a full trigger/evidence/analysis/consequence template" in template
    assert "Preserve the natural language used by the current component" in template
    assert template.rfind("Authoritative target component: `memory:AGENTS.md`") > template.index("<side_info>")
    assert template.rfind("<curr_param>") > template.index("<side_info>")


def test_learned_reference_is_preferred_for_domain_knowledge_fallback():
    example = _load_example_module()
    config_path = (
        Path(__file__).parents[1]
        / "examples"
        / "langchain_adapter"
        / "deepagents_gepa_configs"
        / "credit_approval.toml"
    )
    project = example.build_candidate_from_deep_agent_project(example.load_deepagents_gepa_config(config_path))
    learned_key = "skill:credit-risk-review:reference/learned_expert_patterns.md"
    state = {
        "candidate_excerpt": project.candidate,
        "candidate_constraints": [],
    }

    suggested = example.suggest_component_to_update(state, "effect")
    template = example.reflection_prompt_templates(project.candidate)[learned_key]

    assert suggested == learned_key
    assert "observable signals or business model" in template
    assert "non-applicability" in template
    assert "signal -> concern -> consequence" in template
    assert "non-obvious condition, evidence distinction, comparison, or transmission logic" in template
    assert "shared economic mechanism" in template
    assert "do not append one section per evaluation example" in template
    assert "Do not repeat generic finance" in template
    assert "company name or keyword is only a weak discovery clue" in template
    assert "Do not invent fixed cutoffs" in template
    assert "unread reference needs a reachable routing instruction" in template
    assert "evidence lists as possible sources" in template


def test_proposal_reviewer_revises_and_persists_original_and_review(tmp_path):
    example = _load_example_module()
    store = example.RunArtifactStore(tmp_path / "run")
    prompt = (
        "Component boundary rules for `skill:risk:reference/learned.md`:\n"
        "Current target component (this is the only text you may replace):\n"
        "```\n# Existing\n\nKeep this.\n```\n\n"
        "Before answering, verify the replacement."
    )
    original = (
        "Proposal rationale:\n- Failure pattern: too broad\n\n"
        "Final replacement:\n```markdown\n# Existing\n\nA very long template.\n```"
    )
    revised = (
        "Proposal rationale:\n- Failure pattern: retain one compact cue\n\n"
        "Final replacement:\n```markdown\n# Existing\n\nOne compact signal -> concern -> consequence reminder.\n```"
    )
    review_output = f"Decision: REVISE\nIssues:\n- proposal repeats generic analysis\nReviewed response:\n{revised}"
    review_outputs = iter(
        [
            review_output,
            "Decision: ACCEPT\nIssues:\n- none\nReviewed response:\nsame",
        ]
    )
    reflection = example.with_proposal_quality_review(
        lambda _prompt: original,
        lambda _prompt: next(review_outputs),
        example.DefaultProposalReviewer(),
        store,
    )

    result = reflection(prompt)

    assert result == revised
    review_index = [
        json.loads(line)
        for line in (tmp_path / "run" / "proposal_reviews" / "index.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["review_pass"] for row in review_index] == [1, 2]
    assert review_index[0]["decision"] == "REVISE"
    assert review_index[1]["decision"] == "ACCEPT"
    assert review_index[0]["component"] == "skill:risk:reference/learned.md"
    detail_dir = tmp_path / "run" / review_index[0]["detail_dir"]
    assert (detail_dir / "original_proposal.txt").read_text(encoding="utf-8") == original
    assert (detail_dir / "reviewed_proposal.txt").read_text(encoding="utf-8") == revised
    second_detail_dir = tmp_path / "run" / review_index[1]["detail_dir"]
    assert (second_detail_dir / "original_proposal.txt").read_text(encoding="utf-8") == revised


def test_proposal_reviewer_rejects_to_exact_no_change():
    example = _load_example_module()
    prompt = (
        "Component boundary rules for `memory:AGENTS.md`:\n"
        "Current target component (this is the only text you may replace):\n"
        "```\n# Existing memory\n\nKeep this exact text.\n```\n\n"
        "Before answering, verify the replacement."
    )
    reflection = example.with_proposal_quality_review(
        lambda _prompt: "Proposal rationale:\n- bad\n\nFinal replacement:\n```\nMemorize hidden facts.\n```",
        lambda _prompt: ("Decision: REJECT\nIssues:\n- no runtime-observable evidence\nReviewed response:\nsame"),
        example.DefaultProposalReviewer(),
    )

    result = reflection(prompt)

    assert "# Existing memory\n\nKeep this exact text." in result
    assert "Memorize hidden facts." not in result


def test_proposal_reviewer_retries_revise_without_reviewed_response(tmp_path):
    example = _load_example_module()
    store = example.RunArtifactStore(tmp_path / "run")
    prompt = (
        "Component boundary rules for `memory:AGENTS.md`:\n"
        "Current target component (this is the only text you may replace):\n"
        "```\n# Existing memory\n\nUse the existing skill.\n```\n\n"
        "Before answering, verify the replacement."
    )
    original = (
        "Proposal rationale:\n- copy a domain rule\n\n"
        "Final replacement:\n```markdown\n# Existing memory\n\nAlways flag one company's risk.\n```"
    )
    revised = (
        "Proposal rationale:\n- keep only the execution-layer fix\n\n"
        "Final replacement:\n```markdown\n# Existing memory\n\nAlways load the existing skill before answering.\n```"
    )
    prompts: list[str] = []
    outputs = iter(
        [
            json.dumps({"decision": "REVISE", "issues": ["domain knowledge belongs in reference"]}),
            json.dumps(
                {
                    "decision": "REVISE",
                    "issues": ["retain only skill activation"],
                    "reviewed_response": revised,
                }
            ),
            json.dumps({"decision": "ACCEPT", "issues": [], "reviewed_response": "same"}),
        ]
    )

    def review_lm(review_prompt):
        prompts.append(review_prompt)
        return next(outputs)

    reflection = example.with_proposal_quality_review(
        lambda _prompt: original,
        review_lm,
        example.DefaultProposalReviewer(),
        store,
    )

    assert reflection(prompt) == revised
    assert "previous review returned REVISE but omitted" in prompts[1]
    assert "Do not ACCEPT the unchanged proposal" in prompts[1]
    review_index = [
        json.loads(line)
        for line in (tmp_path / "run" / "proposal_reviews" / "index.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["decision"] for row in review_index] == ["REVISE_MISSING_RESPONSE", "REVISE", "ACCEPT"]
    assert review_index[0]["has_reviewed_response"] is False
    assert review_index[0]["reviewed_response_chars"] is None


def test_proposal_reviewer_missing_revision_exhaustion_returns_no_change():
    example = _load_example_module()
    prompt = (
        "Component boundary rules for `memory:AGENTS.md`:\n"
        "Current target component (this is the only text you may replace):\n"
        "```\n# Existing memory\n\nUse the existing skill.\n```\n\n"
        "Before answering, verify the replacement."
    )
    reflection = example.with_proposal_quality_review(
        lambda _prompt: (
            "Proposal rationale:\n- copy hidden facts\n\n"
            "Final replacement:\n```markdown\n# Existing memory\n\nMemorize hidden company facts.\n```"
        ),
        lambda _prompt: json.dumps({"decision": "REVISE", "issues": ["remove hidden facts"]}),
        example.DefaultProposalReviewer(),
    )

    result = reflection(prompt)

    assert "# Existing memory\n\nUse the existing skill." in result
    assert "Memorize hidden company facts." not in result


def test_proposal_reviewer_parses_json_and_caps_issue_chatter():
    example = _load_example_module()
    issues = [f"issue {index} " + ("x" * 600) for index in range(8)]
    review = example.DefaultProposalReviewer._parse_output(
        json.dumps(
            {
                "decision": "REJECT",
                "issues": issues,
                "reviewed_response": "same",
            }
        )
    )

    assert review.decision == "REJECT"
    assert len(review.issues) == 5
    assert all(len(issue) <= 500 for issue in review.issues)


def test_proposal_reviewer_does_not_accept_malformed_revision():
    example = _load_example_module()

    review = example.DefaultProposalReviewer._parse_output(
        json.dumps(
            {
                "decision": "REVISE",
                "issues": ["remove domain knowledge from memory"],
                "reviewed_response": "Only reviewer commentary, without a complete replacement.",
            }
        )
    )

    assert review.decision == "REVISE"
    assert review.reviewed_response is None
    assert "complete reviewed_response is required" in review.issues[-1]


def test_proposal_reviewer_receives_concrete_growth_and_overfit_advisories():
    example = _load_example_module()
    current = "# Agent\n\nRoute requests."
    replacement = (
        "# Agent\n\n## Operating Instructions\nAlways read `reference/routing.md`.\n\n"
        "## Example\n- Input: Change the owner email.\n" + ("Repeat routing policy. " * 30)
    )
    prompt = (
        "Component boundary rules for `memory:AGENTS.md`:\n"
        "Current target component (this is the only text you may replace):\n"
        f"```\n{current}\n```\n\n"
        "Before answering, verify the replacement."
    )
    proposal = f"Proposal rationale:\n- expand policy\n\nFinal replacement:\n```markdown\n{replacement}\n```"

    review_prompt = example.DefaultProposalReviewer._build_prompt(prompt, proposal)

    assert "Deterministic pre-review advisories" in review_prompt
    assert "global surface grows" in review_prompt
    assert "embeds examples" in review_prompt
    assert "skill-relative reference path" in review_prompt


def test_proposal_reviewer_exhausted_revision_returns_no_change(tmp_path):
    example = _load_example_module()
    store = example.RunArtifactStore(tmp_path / "run")
    prompt = (
        "Component boundary rules for `memory:AGENTS.md`:\n"
        "Current target component (this is the only text you may replace):\n"
        "```\n# Existing memory\n\nKeep this exact text.\n```\n\n"
        "Before answering, verify the replacement."
    )
    bloated = (
        "Proposal rationale:\n- add a universal workflow\n\n"
        "Final replacement:\n```markdown\n# Existing memory\n\nRepeat every skill and reference path here.\n```"
    )
    review_output = f"Decision: REVISE\nIssues:\n- still duplicates skill-owned workflow\nReviewed response:\n{bloated}"
    reflection = example.with_proposal_quality_review(
        lambda _prompt: bloated,
        lambda _prompt: review_output,
        example.DefaultProposalReviewer(),
        store,
    )

    result = reflection(prompt)

    assert "# Existing memory\n\nKeep this exact text." in result
    assert "Repeat every skill and reference path here." not in result
    review_index = [
        json.loads(line)
        for line in (tmp_path / "run" / "proposal_reviews" / "index.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["decision"] for row in review_index] == ["REVISE", "REVISE", "REVISE_EXHAUSTED"]


def test_final_test_artifact_records_tied_candidate_as_diagnostic_only(tmp_path):
    example = _load_example_module()
    store = example.RunArtifactStore(tmp_path / "run")
    seed = SimpleNamespace(scores=[0.5], outputs=[])
    best = SimpleNamespace(scores=[0.5], outputs=[])
    tied_candidate = SimpleNamespace(scores=[1.0], outputs=[])

    summary = store.write_final_test(
        examples=[{"input": "held-out"}],
        seed_evaluation=seed,
        best_evaluation=best,
        diagnostic_evaluations={1: tied_candidate},
        diagnostic_val_scores={1: 1.0},
    )

    assert summary["improvement"] == 0.0
    assert summary["metric_calls"] == 3
    assert summary["diagnostic_candidates"] == [
        {
            "candidate_idx": 1,
            "validation_score": 1.0,
            "test_mean": 1.0,
            "delta_vs_seed": 0.5,
            "selection_effect": "diagnostic_only",
        }
    ]
    assert (tmp_path / "run" / "final_test" / "candidate_0001.json").exists()


def test_actionability_preflight_artifacts_separate_shared_rubric_and_optimization_pool(tmp_path):
    example = _load_example_module()
    store = example.RunArtifactStore(tmp_path / "run")
    config = SimpleNamespace(dataset=SimpleNamespace(rubric="统一评价规则"))
    project = SimpleNamespace(candidate={"memory:AGENTS.md": "Keep behavior."}, surfaces={})
    rows = [
        {"input": "可优化企业", "rubric": "统一评价规则"},
        {"input": "工具阻塞企业", "rubric": "统一评价规则"},
    ]
    evaluation = SimpleNamespace(
        scores=[0.2, 0.1],
        outputs=[
            {"state": {"fitness": {"mutation_eligible": True, "failure_classification": "SKILL_DEFECT"}}},
            {
                "state": {
                    "fitness": {
                        "mutation_eligible": False,
                        "failure_classification": "TOOL_CAPABILITY_GAP",
                        "tool_capability_gaps": ["客户交易信息获取"],
                    }
                }
            },
        ],
        trajectories=[
            {"feedback": "- suggested_component: skill:risk:reference/learned.md"},
            {"feedback": "- suggested_component: none"},
        ],
        num_metric_calls=None,
    )
    partition = example.DefaultActionabilityPolicy().partition(rows, evaluation, regression_guard_limit=1)

    store.write_run_inputs(
        config_path=tmp_path / "missing.toml",
        config=config,
        project=project,
        train_set=rows,
        val_set=[],
        test_set=[],
    )
    summary = store.write_actionability_preflight(
        examples=rows,
        evaluation=evaluation,
        partition=partition,
        optimization_examples=[rows[0]],
    )

    persisted_train = [
        json.loads(line)
        for line in (tmp_path / "run" / "datasets" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    optimization_train = [
        json.loads(line)
        for line in (tmp_path / "run" / "datasets" / "optimization_train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert (tmp_path / "run" / "datasets" / "rubric.md").read_text(encoding="utf-8").strip() == "统一评价规则"
    assert all("rubric" not in row for row in persisted_train)
    assert all("rubric" not in row for row in optimization_train)
    assert [row["input"] for row in optimization_train] == ["可优化企业"]
    assert summary["metric_calls"] == 2
    assert summary["tool_blocked_indices"] == [1]


def test_artifact_callback_writes_agent_logs_and_rejected_proposals(tmp_path):
    example = _load_example_module()
    store = example.RunArtifactStore(tmp_path / "run")
    callback = store.create_callback()
    parent_candidate = {
        "memory:AGENTS.md": "Keep support routing stable.",
        "main:system_prompt": "Return a route.",
    }

    store.write_agent_rollout(
        example={"input": "Where is my invoice?", "expected": "billing"},
        state={
            "messages": [example.AIMessage(content="<route>billing</route>")],
            "candidate_hash": "abc123",
            "fitness": {
                "hard": 1.0,
                "failure_classification": "TOOL_CAPABILITY_GAP",
                "remediation_actions": [
                    {
                        "type": "ADD_TOOL_OR_MCP",
                        "owner": "tool_or_mcp",
                        "targets": ["invoice_data"],
                        "reason": "no current tool can retrieve invoice data",
                    }
                ],
            },
        },
        score=1.0,
        feedback="ok",
    )
    callback.on_evaluation_end(
        {
            "iteration": 1,
            "candidate_idx": 0,
            "scores": [0.4],
            "outputs": [],
            "trajectories": [],
            "is_seed_candidate": True,
        }
    )
    callback.on_proposal_start(
        {
            "iteration": 1,
            "parent_candidate": parent_candidate,
            "components": ["memory:AGENTS.md"],
            "reflective_dataset": {"memory:AGENTS.md": [{"Feedback": "missing output contract"}]},
        }
    )
    callback.on_proposal_end(
        {
            "iteration": 1,
            "new_instructions": {"memory:AGENTS.md": "Copy every skill into memory."},
            "prompts": {"memory:AGENTS.md": "reflection prompt"},
            "raw_lm_outputs": {
                "memory:AGENTS.md": "Failure pattern: missing output contract\n"
                "Selected component: memory:AGENTS.md\n"
                "```Copy every skill into memory.```"
            },
        }
    )
    callback.on_evaluation_end(
        {
            "iteration": 1,
            "candidate_idx": None,
            "scores": [0.3],
            "outputs": [],
            "trajectories": [{"feedback": "- failure_classification: EXECUTION_LAPSE"}],
            "is_seed_candidate": False,
        }
    )
    callback.on_candidate_rejected(
        {
            "iteration": 1,
            "old_score": 0.4,
            "new_score": 0.3,
            "reason": "New subsample score 0.3 not better than old score 0.4",
        }
    )

    assert (tmp_path / "run" / "agent_logs" / "rollouts.jsonl").exists()
    remediation_rows = [
        json.loads(line)
        for line in (tmp_path / "run" / "diagnostics" / "remediation_actions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert remediation_rows[0]["type"] == "ADD_TOOL_OR_MCP"
    assert remediation_rows[0]["targets"] == ["invoice_data"]
    assert remediation_rows[0]["detail_file"] == "agent_logs/rollouts/000000.json"
    assert (tmp_path / "run" / "proposals" / "0001" / "candidate.json").exists()
    assert (tmp_path / "run" / "proposals" / "0001" / "diff_against_parent.patch").exists()
    assert (tmp_path / "run" / "proposals" / "0001" / "proposal_rationale.json").exists()
    assert not (tmp_path / "run" / "proposals" / "0001" / "proposal_rationale_missing.json").exists()
    assert (tmp_path / "run" / "proposals" / "0001" / "prompts" / "memory__AGENTS.md.txt").exists()
    assert (tmp_path / "run" / "rejected_proposals" / "0001" / "candidate.json").exists()
    assert (tmp_path / "run" / "rejected_proposals" / "0001" / "diff_against_parent.patch").exists()
    assert (tmp_path / "run" / "rejected_proposals" / "0001" / "proposal_rationale.json").exists()
    rejected_history = callback.rejected_history_prompt_block()
    assert "Recent rejected proposal lessons" in rejected_history
    assert "Rejected rationale:" in rejected_history
    assert "Rejected diff preview:" in rejected_history
    assert "memory:AGENTS.md" in rejected_history


def test_artifact_callback_marks_missing_proposal_rationale(tmp_path):
    example = _load_example_module()
    store = example.RunArtifactStore(tmp_path / "run")
    callback = store.create_callback()

    callback.on_proposal_start(
        {
            "iteration": 2,
            "parent_candidate": {"main:system_prompt": "Return a route."},
            "components": ["main:system_prompt"],
            "reflective_dataset": {"main:system_prompt": [{"Feedback": "missing output contract"}]},
        }
    )
    callback.on_proposal_end(
        {
            "iteration": 2,
            "new_instructions": {"main:system_prompt": "Return <route>billing</route>."},
            "prompts": {"main:system_prompt": "reflection prompt"},
            "raw_lm_outputs": {"main:system_prompt": "```\nReturn <route>billing</route>.\n```"},
        }
    )

    metadata = json.loads((tmp_path / "run" / "proposals" / "0002" / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["missing_proposal_rationale"] == ["main:system_prompt"]
    assert (tmp_path / "run" / "proposals" / "0002" / "proposal_rationale_missing.json").exists()
    assert not (tmp_path / "run" / "proposals" / "0002" / "proposal_rationale.json").exists()


def test_artifact_callback_marks_changed_reference_that_trace_did_not_read(tmp_path):
    example = _load_example_module()
    store = example.RunArtifactStore(tmp_path / "run")
    callback = store.create_callback()
    reference_key = "skill:credit-risk-review:reference/financial_statement_analysis.md"
    parent = {reference_key: "# Existing"}

    callback.on_evaluation_end(
        {
            "iteration": 1,
            "candidate_idx": 0,
            "scores": [0.4],
            "outputs": [],
            "trajectories": [],
            "is_seed_candidate": True,
        }
    )
    callback.on_proposal_start(
        {
            "iteration": 1,
            "parent_candidate": parent,
            "components": [reference_key],
            "reflective_dataset": {reference_key: []},
        }
    )
    callback.on_proposal_end(
        {
            "iteration": 1,
            "new_instructions": {reference_key: "# Revised"},
            "prompts": {reference_key: "prompt"},
            "raw_lm_outputs": {reference_key: "Final replacement:\n```markdown\n# Revised\n```"},
        }
    )
    callback.on_evaluation_end(
        {
            "iteration": 1,
            "candidate_idx": None,
            "scores": [0.5],
            "outputs": [],
            "trajectories": [{"state": {"messages": []}, "feedback": "ok"}],
            "is_seed_candidate": False,
        }
    )

    metadata = json.loads((tmp_path / "run" / "proposals" / "0001" / "metadata.json").read_text())
    assert metadata["component_consumption"] == {reference_key: False}
    assert metadata["changed_but_unconsumed"] == [reference_key]


def test_reflection_provider_errors_are_written_to_artifacts(tmp_path):
    example = _load_example_module()
    store = example.RunArtifactStore(tmp_path / "run")

    def failing_reflection(_prompt):
        raise TimeoutError("provider timed out")

    reflection = example.with_reflection_error_artifacts(failing_reflection, store)
    prompt = "Component boundary rules for `skill:test:SKILL.md`:\nReturn a replacement."

    with pytest.raises(TimeoutError, match="provider timed out"):
        reflection(prompt)

    rows = [
        json.loads(line)
        for line in (tmp_path / "run" / "reflection_errors" / "index.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["component"] == "skill:test:SKILL.md"
    assert rows[0]["error_type"] == "TimeoutError"
    assert rows[0]["prompt_chars"] == len(prompt)
    assert (tmp_path / "run" / "reflection_errors" / "000000.prompt.txt").read_text(encoding="utf-8") == prompt


def test_artifacts_materialize_explicitly_selected_candidate(tmp_path):
    example = _load_example_module()
    run_dir = tmp_path / "run"
    store = example.RunArtifactStore(run_dir)
    seed_candidate = {"memory:AGENTS.md": "Seed memory."}
    accepted_candidate = {"memory:AGENTS.md": "Accepted improved memory."}
    project = SimpleNamespace(candidate=seed_candidate, surfaces={})
    result = SimpleNamespace(
        candidates=[seed_candidate, accepted_candidate],
        parents=[[None], [0]],
        val_aggregate_scores=[1.0, 1.0],
        discovery_eval_counts=[0, 4],
        total_metric_calls=5,
        num_full_val_evals=2,
        num_candidates=2,
        best_idx=0,
        best_candidate=seed_candidate,
    )
    store.write_run_inputs(
        config_path=tmp_path / "missing.toml",
        config={},
        project=project,
        train_set=[],
        val_set=[],
        test_set=[],
    )

    def materialize(_project, candidate, destination):
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "selected.txt").write_text(candidate["memory:AGENTS.md"], encoding="utf-8")

    summary = store.finalize(
        result=result,
        project=project,
        apply_candidate=materialize,
        best_idx=1,
    )
    best_candidate = json.loads((run_dir / "best_candidate" / "candidate.json").read_text(encoding="utf-8"))

    assert summary["best_idx"] == 1
    assert summary["gepa_best_idx"] == 0
    assert summary["tie_break_applied"] is True
    assert summary["selection_policy"] == "incumbent_on_validation_tie"
    assert summary["tied_best_indices"] == [0, 1]
    assert summary["metric_calls_by_phase"] == {
        "gepa": 5,
        "preflight": 0,
        "final_test": 0,
        "all_phases": 5,
    }
    assert best_candidate == accepted_candidate
    assert (run_dir / "materialized_best_candidate" / "selected.txt").read_text(
        encoding="utf-8"
    ) == "Accepted improved memory."


def test_run_analyzer_flags_connection_blocked_experiment(tmp_path):
    analyzer = _load_analyzer_module()
    run_dir = tmp_path / "run"
    (run_dir / "agent_logs").mkdir(parents=True)
    (run_dir / "proposals").mkdir(parents=True)
    (run_dir / "rejected_proposals").mkdir(parents=True)
    (run_dir / "result_summary.json").write_text(
        json.dumps(
            {
                "best_val_score": 0.0,
                "val_aggregate_scores": [0.0],
                "total_metric_calls": 10,
                "num_candidates": 1,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "agent_logs" / "rollouts.jsonl").write_text(
        json.dumps({"score": 0.0, "state": {"error": "APIConnectionError('Connection error.')"}}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "proposals" / "index.jsonl").write_text(
        json.dumps({"status": "started", "components": ["memory:AGENTS.md"]}) + "\n",
        encoding="utf-8",
    )

    summary = analyzer.summarize_run(run_dir)

    assert summary["experiment_valid_for_effectiveness"] is False
    assert summary["runtime_errors"] == {"APIConnectionError('Connection error.')": 1}
    assert "not valid for algorithm-effectiveness analysis" in summary["diagnosis"][0]


def test_run_analyzer_counts_missing_proposal_rationale(tmp_path):
    analyzer = _load_analyzer_module()
    run_dir = tmp_path / "run"
    (run_dir / "agent_logs").mkdir(parents=True)
    (run_dir / "proposals" / "0001").mkdir(parents=True)
    (run_dir / "rejected_proposals").mkdir(parents=True)
    (run_dir / "result_summary.json").write_text(
        json.dumps({"best_val_score": 0.0, "val_aggregate_scores": [0.0]}),
        encoding="utf-8",
    )
    (run_dir / "agent_logs" / "rollouts.jsonl").write_text("", encoding="utf-8")
    (run_dir / "proposals" / "0001" / "proposal_rationale_missing.json").write_text(
        json.dumps(["main:system_prompt"]),
        encoding="utf-8",
    )
    (run_dir / "proposals" / "index.jsonl").write_text(
        json.dumps({"status": "proposed", "proposal_dir": "proposals/0001"}) + "\n",
        encoding="utf-8",
    )

    summary = analyzer.summarize_run(run_dir)

    assert summary["proposal_missing_rationale_files"] == 1
    assert any("missing rationale" in note for note in summary["diagnosis"])


def test_run_analyzer_keeps_preflight_out_of_optimization_statistics(tmp_path):
    analyzer = _load_analyzer_module()
    run_dir = tmp_path / "run"
    (run_dir / "agent_logs").mkdir(parents=True)
    (run_dir / "proposals").mkdir(parents=True)
    (run_dir / "rejected_proposals").mkdir(parents=True)
    (run_dir / "diagnostics").mkdir(parents=True)
    (run_dir / "result_summary.json").write_text(
        json.dumps({"best_val_score": 0.5, "val_aggregate_scores": [0.5], "total_metric_calls": 2}),
        encoding="utf-8",
    )
    preflight = {
        "evaluated_count": 1,
        "metric_calls": 1,
        "actionable_indices": [0],
        "regression_guard_indices": [],
        "tool_blocked_indices": [],
        "fallback_to_unfiltered": False,
    }
    (run_dir / "diagnostics" / "actionability_preflight.json").write_text(
        json.dumps(preflight),
        encoding="utf-8",
    )
    (run_dir / "agent_logs" / "rollouts.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"score": 0.1, "evaluation_phase": "preflight_train"}),
                json.dumps({"score": 0.5, "evaluation_phase": "optimization"}),
                json.dumps({"score": 1.0, "evaluation_phase": "final_test_seed"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = analyzer.summarize_run(run_dir)

    assert summary["preflight_rollout_count"] == 1
    assert summary["rollout_count"] == 1
    assert summary["final_test_rollout_count"] == 1
    assert summary["rollout_score_mean"] == 0.5
    assert summary["preflight_actionability"] == preflight


def test_run_analyzer_reports_judge_disagreement_and_no_actionable_proposal(tmp_path):
    analyzer = _load_analyzer_module()
    run_dir = tmp_path / "run"
    (run_dir / "agent_logs" / "rollouts").mkdir(parents=True)
    (run_dir / "proposals").mkdir(parents=True)
    (run_dir / "rejected_proposals").mkdir(parents=True)
    (run_dir / "result_summary.json").write_text(
        json.dumps(
            {
                "best_val_score": 1.0,
                "val_aggregate_scores": [1.0, 1.0],
                "tied_best_indices": [0, 1],
                "final_test": {
                    "seed_mean": 0.0,
                    "best_mean": 0.0,
                    "improvement": 0.0,
                    "diagnostic_candidates": [
                        {
                            "candidate_idx": 1,
                            "validation_score": 1.0,
                            "test_mean": 1.0,
                            "delta_vs_seed": 1.0,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    detail = {
        "input": "示例企业",
        "expected": "engineering",
        "fitness": {
            "hard": 1.0,
            "judge_score": 0.0,
            "mutation_eligible": False,
            "tool_capability_gaps": ["客户交易信息获取"],
            "remediation_actions": [
                {
                    "type": "ADD_TOOL_OR_MCP",
                    "owner": "tool_or_mcp",
                    "targets": ["客户交易信息获取"],
                    "reason": "missing capability",
                }
            ],
        },
        "constraints": [],
    }
    (run_dir / "agent_logs" / "rollouts" / "000000.json").write_text(
        json.dumps(detail),
        encoding="utf-8",
    )
    (run_dir / "agent_logs" / "rollouts.jsonl").write_text(
        json.dumps({"score": 0.0, "detail_file": "rollouts/000000.json"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "proposals" / "index.jsonl").write_text(
        json.dumps({"iteration": 1, "status": "rejected", "components": []}) + "\n",
        encoding="utf-8",
    )

    summary = analyzer.summarize_run(run_dir)

    assert summary["deterministic_judge_disagreement_count"] == 1
    assert summary["no_actionable_proposal_count"] == 1
    assert summary["remediation_types"] == {"ADD_TOOL_OR_MCP": 1}
    assert summary["remediation_owners"] == {"tool_or_mcp": 1}
    assert summary["tool_capability_gap_unique_inputs"] == {"客户交易信息获取": 1}
    assert any("exact authoritative target match" in note for note in summary["diagnosis"])
    assert any("no actionable text component" in note for note in summary["diagnosis"])
    assert any("ADD_TOOL_OR_MCP" in note for note in summary["diagnosis"])
    assert any("held-out diagnostic mean by 1.000" in note for note in summary["diagnosis"])


def test_run_analyzer_reports_unfinished_reflection_proposal(tmp_path):
    analyzer = _load_analyzer_module()
    run_dir = tmp_path / "run"
    (run_dir / "agent_logs").mkdir(parents=True)
    (run_dir / "proposals").mkdir(parents=True)
    (run_dir / "rejected_proposals").mkdir(parents=True)
    (run_dir / "result_summary.json").write_text(
        json.dumps({"best_val_score": 0.5, "val_aggregate_scores": [0.5]}),
        encoding="utf-8",
    )
    (run_dir / "agent_logs" / "rollouts.jsonl").write_text("", encoding="utf-8")
    (run_dir / "proposals" / "index.jsonl").write_text(
        json.dumps({"iteration": 1, "status": "started", "components": ["main:system_prompt"]}) + "\n",
        encoding="utf-8",
    )

    summary = analyzer.summarize_run(run_dir)

    assert summary["proposal_statuses"] == {"started": 1}
    assert summary["reflection_error_count"] == 0
    assert any("stopped after reflection started" in note for note in summary["diagnosis"])


def test_run_analyzer_handles_windows_style_proposal_dirs(tmp_path):
    analyzer = _load_analyzer_module()
    run_dir = tmp_path / "run"
    (run_dir / "agent_logs").mkdir(parents=True)
    (run_dir / "proposals" / "0001").mkdir(parents=True)
    (run_dir / "rejected_proposals").mkdir(parents=True)
    (run_dir / "result_summary.json").write_text(
        json.dumps({"best_val_score": 1.0, "val_aggregate_scores": [0.5, 1.0]}),
        encoding="utf-8",
    )
    (run_dir / "agent_logs" / "rollouts.jsonl").write_text("", encoding="utf-8")
    (run_dir / "proposals" / "0001" / "proposal_rationale.json").write_text(
        json.dumps({"main:system_prompt": "Proposal rationale: ..."}),
        encoding="utf-8",
    )
    (run_dir / "proposals" / "0001" / "diff_against_parent.patch").write_text("diff", encoding="utf-8")
    (run_dir / "proposals" / "index.jsonl").write_text(
        json.dumps({"status": "accepted", "proposal_dir": "proposals\\0001"}) + "\n",
        encoding="utf-8",
    )

    summary = analyzer.summarize_run(run_dir)

    assert summary["proposal_rationale_files"] == 1
    assert summary["proposal_diff_files"] == 1


def test_run_analyzer_deduplicates_proposal_lifecycle_events(tmp_path):
    analyzer = _load_analyzer_module()
    run_dir = tmp_path / "run"
    (run_dir / "agent_logs").mkdir(parents=True)
    (run_dir / "proposals" / "0001").mkdir(parents=True)
    (run_dir / "rejected_proposals" / "0001").mkdir(parents=True)
    (run_dir / "result_summary.json").write_text(
        json.dumps(
            {
                "best_val_score": 0.5,
                "val_aggregate_scores": [0.5, 0.5],
                "tied_best_indices": [0, 1],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "agent_logs" / "rollouts.jsonl").write_text("", encoding="utf-8")
    events = [
        {
            "iteration": 1,
            "status": "started",
            "components": ["main:system_prompt"],
            "proposal_dir": "proposals/0001",
        },
        {
            "iteration": 1,
            "status": "proposed",
            "components": ["main:system_prompt"],
            "proposal_dir": "proposals/0001",
        },
        {
            "iteration": 1,
            "status": "proposed",
            "components": ["main:system_prompt"],
            "proposal_dir": "proposals/0001",
        },
        {
            "iteration": 1,
            "status": "accepted",
            "components": ["main:system_prompt"],
            "proposal_dir": "proposals/0001",
        },
    ]
    (run_dir / "proposals" / "index.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    (run_dir / "rejected_proposals" / "index.jsonl").write_text(
        json.dumps({"iteration": 1, "proposal_dir": "rejected_proposals/0001"}) + "\n",
        encoding="utf-8",
    )
    for directory in [run_dir / "proposals" / "0001", run_dir / "rejected_proposals" / "0001"]:
        (directory / "proposal_rationale.json").write_text("{}", encoding="utf-8")

    summary = analyzer.summarize_run(run_dir)

    assert summary["proposal_event_count"] == 4
    assert summary["proposal_count"] == 1
    assert summary["proposal_statuses"] == {"accepted": 1}
    assert summary["proposed_components"] == {"main:system_prompt": 1}
    assert summary["proposal_rationale_files"] == 1
    assert any("accepted candidate tied" in note for note in summary["diagnosis"])


def test_run_analyzer_reports_unloadable_skill_and_held_out_regression(tmp_path):
    analyzer = _load_analyzer_module()
    run_dir = tmp_path / "run"
    (run_dir / "agent_logs" / "rollouts").mkdir(parents=True)
    (run_dir / "proposals").mkdir(parents=True)
    (run_dir / "rejected_proposals").mkdir(parents=True)
    (run_dir / "result_summary.json").write_text(
        json.dumps(
            {
                "best_idx": 1,
                "gepa_best_idx": 0,
                "selection_policy": "latest_accepted_on_validation_tie",
                "best_val_score": 1.0,
                "val_aggregate_scores": [1.0, 1.0],
                "tied_best_indices": [0, 1],
                "final_test": {"improvement": -0.5},
            }
        ),
        encoding="utf-8",
    )
    detail = {
        "constraints": [
            {
                "passed": False,
                "name": "skill:credit-risk-review:SKILL.md:frontmatter",
                "message": "missing",
                "severity": "hard",
            }
        ],
        "candidate_runtime_skipped": True,
        "fitness": {},
    }
    (run_dir / "agent_logs" / "rollouts" / "000000.json").write_text(
        json.dumps(detail),
        encoding="utf-8",
    )
    (run_dir / "agent_logs" / "rollouts.jsonl").write_text(
        json.dumps({"score": 0.0, "detail_file": "rollouts/000000.json"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "proposals" / "index.jsonl").write_text(
        json.dumps({"iteration": 1, "status": "accepted", "components": ["skill:credit-risk-review:SKILL.md"]}) + "\n",
        encoding="utf-8",
    )

    summary = analyzer.summarize_run(run_dir)

    assert summary["candidate_runtime_skipped_count"] == 1
    assert summary["unloadable_skill_failures"] == {"skill:credit-risk-review:SKILL.md:frontmatter": 1}
    assert any("Runtime-unloadable SKILL.md" in note for note in summary["diagnosis"])
    assert any("Held-out test regressed by 0.500" in note for note in summary["diagnosis"])
    assert any("legacy policy deployed the newest" in note for note in summary["diagnosis"])


def test_run_analyzer_excludes_no_failure_from_rejected_failure_class(tmp_path):
    analyzer = _load_analyzer_module()
    run_dir = tmp_path / "run"
    (run_dir / "agent_logs").mkdir(parents=True)
    (run_dir / "proposals").mkdir(parents=True)
    (run_dir / "rejected_proposals").mkdir(parents=True)
    (run_dir / "result_summary.json").write_text(
        json.dumps({"best_val_score": 0.5, "val_aggregate_scores": [0.5]}),
        encoding="utf-8",
    )
    (run_dir / "agent_logs" / "rollouts.jsonl").write_text("", encoding="utf-8")
    (run_dir / "proposals" / "index.jsonl").write_text("", encoding="utf-8")
    (run_dir / "rejected_proposals" / "index.jsonl").write_text(
        json.dumps(
            {
                "iteration": 1,
                "failure_classifications": ["NO_FAILURE", "NO_FAILURE", "EXECUTION_LAPSE"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = analyzer.summarize_run(run_dir)

    assert summary["rejected_failure_classes"] == {"EXECUTION_LAPSE": 1}
    assert any("EXECUTION_LAPSE (1)" in note for note in summary["diagnosis"])


def test_local_runner_forces_no_proxy_for_localhost(monkeypatch):
    runner = _load_runner_module()
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("NO_PROXY", "example.com")

    runner.configure_local_no_proxy("http://127.0.0.1:8080/v1")

    assert "HTTP_PROXY" not in runner.os.environ
    assert "HTTPS_PROXY" not in runner.os.environ
    assert "127.0.0.1" in runner.os.environ["NO_PROXY"]
    assert "localhost" in runner.os.environ["NO_PROXY"]
    assert "example.com" in runner.os.environ["NO_PROXY"]
    assert runner.os.environ["no_proxy"] == runner.os.environ["NO_PROXY"]


def _write_config(path: Path, project_root: Path, dataset_path: Path | None = None) -> None:
    dataset_section = ""
    if dataset_path is not None:
        dataset_section = f"""
[dataset]
source = "golden_jsonl"
path = "{dataset_path.as_posix()}"
"""
    path.write_text(
        f"""
[experiment]
name = "support-router-self-optimization"

[agent]
mode = "filesystem"
project_root = "{project_root.as_posix()}"
system_prompt = "You are a support router. Use skills before final answers."
memory = ["AGENTS.md"]
skills = ["skills"]
tools = ["tag_ticket"]

[[agent.subagents]]
name = "triage"
description = "Use for ambiguous support routing."
system_prompt = "You are a triage assistant."
tools = ["lookup_policy"]
skills = ["subagents/triage/skills"]

[surfaces.extra_memory]
kind = "file"
path = "AGENTS.md"
component = "memory:AGENTS.md"
source_type = "memory"

[[mcp.servers]]
name = "risk-docs"
transport = "stdio"
command = "risk-docs-mcp"

[[mcp.tools]]
name = "search_risk_docs"
server = "risk-docs"
description = "Search risk reference documents for due-diligence warnings."
{dataset_section}
""",
        encoding="utf-8",
    )


def test_config_driven_candidate_includes_project_and_mcp_surfaces(tmp_path):
    example = _load_example_module()
    project_root = tmp_path / "project"
    example.create_seed_workspace(project_root)
    config_path = tmp_path / "deepagents_gepa.toml"
    _write_config(config_path, project_root)

    config = example.load_deepagents_gepa_config(config_path)
    project = example.build_candidate_from_deep_agent_project(
        config,
        tool_registry={"tag_ticket": example.tag_ticket, "lookup_policy": example.lookup_policy},
    )

    assert project.config.agent_mode == "manual"
    assert "memory:AGENTS.md" in project.candidate
    assert "skill:support-router:SKILL.md" in project.candidate
    assert "subagent:triage:skill:triage-notes:SKILL.md" in project.candidate
    assert "main:tool:tag_ticket:description" in project.candidate
    assert "mcp:tool:search_risk_docs:description" in project.candidate
    assert all("scripts/" not in key for key in project.candidate)

    application = example.apply_candidate_to_deep_agent_project(project, project.candidate, tmp_path / "applied")

    assert application.mcp_servers[0].name == "risk-docs"
    assert application.mcp_tool_descriptions == {
        "search_risk_docs": "Search risk reference documents for due-diligence warnings."
    }
    assert (tmp_path / "applied" / "AGENTS.md").exists()


@pytest.mark.parametrize(
    "config_name",
    ["manual", "langgraph_cli"],
)
def test_repository_toml_examples_load_and_run(config_name, tmp_path):
    example = _load_example_module()
    config_path = (
        Path(__file__).parents[1] / "examples" / "langchain_adapter" / "deepagents_gepa_configs" / f"{config_name}.toml"
    )
    config = example.load_deepagents_gepa_config(config_path)
    if config.agent_mode in {"manual", "langgraph_cli"}:
        pytest.importorskip("deepagents", reason="DeepAgents modes require deepagents")
    project = example.build_candidate_from_deep_agent_project(config)
    train, val, test = example.load_dataset_from_config(config)
    billing_example = next(row for row in train if row.get("expected") == "billing")

    state = example.configured_rollout(
        project.candidate,
        billing_example,
        ToolFriendlyFakeChatModel(responses=["<route>billing</route>"] * 30),
        project,
        project.candidate,
    )

    assert state.get("error") is None
    expected_routes = {"billing", "account", "engineering", "product"}
    assert {row["expected"] for row in train} == expected_routes
    assert {row["expected"] for row in val} == expected_routes
    assert {row["expected"] for row in test} == expected_routes
    output_text = example.last_message_text(state)
    assert example.extract_route(output_text) == "billing"
    assert "memory:AGENTS.md" in project.candidate
    assert "main:tool:tag_ticket:description" in project.candidate
    assert "subagent:risk-reviewer:tool:lookup_policy:description" in project.candidate
    assert "skill:support-router:SKILL.md" in project.candidate
    assert "subagent:risk-reviewer:skill:risk-review:SKILL.md" in project.candidate
    assert "subagent:risk-reviewer:skill:risk-review:reference/risk_rules.md" in project.candidate
    assert "mcp:tool:search_routing_risks:description" in project.candidate
    assert all("scripts/" not in key for key in project.candidate)

    application = example.apply_candidate_to_deep_agent_project(
        project,
        project.candidate,
        tmp_path / "materialized",
    )
    assert application.mcp_servers[0].name == "routing-risk"
    assert application.mcp_tool_descriptions["search_routing_risks"].startswith("Search routing risk")
    assert (
        tmp_path
        / "materialized"
        / "subagents"
        / "risk-reviewer"
        / "skills"
        / "risk-review"
        / "scripts"
        / "check_route.py"
    ).exists()

    if config_name == "langgraph_cli":
        assert config.surfaces == ()
        assert config.langgraph_config == "langgraph.json"
        assert config.graph == "support_router"
        graph_entry = example._import_from_ref("./langgraph_agent.py:support_router", config.project_root)
        graph = graph_entry({})
        assert type(graph).__name__ == "CompiledStateGraph"
        assert not hasattr(graph, "gepa_deep_agent_spec")


def test_configured_rollout_skips_agent_creation_for_unloadable_skill(monkeypatch):
    example = _load_example_module()
    config_path = (
        Path(__file__).parents[1] / "examples" / "langchain_adapter" / "deepagents_gepa_configs" / "manual.toml"
    )
    project = example.build_candidate_from_deep_agent_project(example.load_deepagents_gepa_config(config_path))
    candidate = dict(project.candidate)
    candidate["skill:support-router:SKILL.md"] = "# Missing frontmatter"

    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("critical candidate should be rejected before agent or baseline execution")

    monkeypatch.setattr(example, "create_deep_agent_from_application", unexpected_call)
    monkeypatch.setattr(example, "run_configured_baseline_for_example", unexpected_call)
    monkeypatch.setattr(example, "configured_runtime_application", unexpected_call)

    state = example.configured_rollout(
        candidate,
        {"input": "I need an invoice."},
        ToolFriendlyFakeChatModel(responses=[]),
        project,
        project.candidate,
    )

    assert state["candidate_runtime_skipped"] is True
    assert "frontmatter" in state["candidate_runtime_skip_reason"]
    assert state["baseline_response"] == ""


def test_credit_approval_demo_loads_expert_risk_section_dataset():
    example = _load_example_module()
    config_path = (
        Path(__file__).parents[1]
        / "examples"
        / "langchain_adapter"
        / "deepagents_gepa_configs"
        / "credit_approval.toml"
    )

    config = example.load_deepagents_gepa_config(config_path)
    project = example.build_candidate_from_deep_agent_project(config)
    train, val, test = example.load_dataset_from_config(config)
    rows = train + val + test

    assert project.config.agent_mode == "manual"
    assert "skill:credit-risk-review:SKILL.md" in project.candidate
    assert "main:tool:lookup_financial_snapshot:description" in project.candidate
    assert "不查询具体企业事实" in project.candidate["main:tool:lookup_policy:description"]
    assert "利润、现金流、负债借款、票据和用信" in project.candidate["main:tool:lookup_financial_snapshot:description"]
    assert "skill:credit-risk-review:reference/financial_statement_analysis.md" in project.candidate
    assert "skill:credit-risk-review:reference/cashflow_and_repayment.md" in project.candidate
    assert "skill:credit-risk-review:reference/collateral_and_guarantee.md" in project.candidate
    assert "skill:credit-risk-review:reference/industry_management_and_warnings.md" in project.candidate
    assert "skill:credit-risk-review:reference/learned_expert_patterns.md" in project.candidate
    assert all("rubric" in row for row in rows)
    assert all("data" in row for row in rows)
    assert all("answer" not in row and "expected" not in row for row in rows)
    assert all("项目风险点" in row["data"] for row in rows)
    assert all("根据企业名称自主检索" in row["rubric"] for row in rows)
    assert all("未覆盖的 checkpoint 仍降低任务分数" in row["rubric"] for row in rows)
    assert all("不要求或奖励审批意见" in row["rubric"] for row in rows)
    assert len(rows) >= 8
    assert all(row["metadata"].get("checkpoints") for row in rows)
    assert all(row["metadata"].get("trace_expectations") for row in rows)
    assert all(row["metadata"].get("tool_coverage") in {"complete", "partial", "none"} for row in rows)
    assert all(row["metadata"].get("checkpoint_count") == len(row["metadata"]["checkpoints"]) for row in rows)
    assert all(checkpoint.get("evidence_expectations") for row in rows for checkpoint in row["metadata"]["checkpoints"])
    steel_row = next(row for row in rows if row["input"] == "华东钢铁集团有限公司")
    scope_checkpoint = next(
        checkpoint
        for checkpoint in steel_row["metadata"]["checkpoints"]
        if checkpoint["label"] == "集团内部信息不对称风险"
    )
    assert scope_checkpoint["evidence_mode"] == "any"
    assert scope_checkpoint["evidence_expectations"] == ["集团穿透信息获取", "财务口径限制信息获取"]
    scope_expectation = next(
        expectation
        for expectation in steel_row["metadata"]["trace_expectations"]
        if expectation["label"] == "财务口径限制信息获取"
    )
    assert scope_expectation["tool_names"] == ["lookup_financial_snapshot"]
    financial_expectations = [
        expectation
        for row in rows
        for expectation in row["metadata"]["trace_expectations"]
        if expectation["label"] in {"财务盈利质量信息获取", "债务结构信息获取", "财务现金流信息获取"}
    ]
    assert financial_expectations
    assert all(expectation["tool_names"] == ["lookup_financial_snapshot"] for expectation in financial_expectations)
    assert all(
        checkpoint["label"] != "授信压降和回款监管必要性"
        for row in rows
        for checkpoint in row["metadata"]["checkpoints"]
    )
    assert (
        "适用条件或信号 -> 重点关注 -> 可能后果"
        in project.candidate["skill:credit-risk-review:reference/learned_expert_patterns.md"]
    )
    assert "不重复常规财务分析" in project.candidate["skill:credit-risk-review:reference/learned_expert_patterns.md"]
    credit_skill = project.candidate["skill:credit-risk-review:SKILL.md"]
    assert "只输出当前信息能够支持的风险点" in credit_skill
    assert "缺少事实依据的维度直接不展开" in credit_skill
    assert "不输出审批意见" in credit_skill

    financial_tool = next(tool for tool in project.spec.tools if tool.name == "lookup_financial_snapshot")
    financial_snapshot = json.loads(financial_tool.invoke({"company_name": "华东钢铁集团有限公司"}))
    assert financial_snapshot["净利润"]["2024年"] == "17.84亿元"
    assert "子公司单体财务" in financial_snapshot["数据限制"]

    skill_constraints = example.skill_structure_constraints(
        "skill:credit-risk-review:SKILL.md",
        project.candidate["skill:credit-risk-review:SKILL.md"],
    )
    assert all(constraint.passed for constraint in skill_constraints)
    assert any(row["metadata"].get("scenario") == "钢铁集团授信_项目风险点对齐" for row in rows)
    assert any(row["input"] == "华东钢铁集团有限公司" for row in rows)
    steel = next(row for row in rows if row["input"] == "华东钢铁集团有限公司")
    steel_labels = {checkpoint["label"] for checkpoint in steel["metadata"]["checkpoints"]}
    assert {"盈利稳定性风险", "盈利质量风险"} <= steel_labels
    assert "盈利稳定性与盈利质量风险" not in steel_labels
    assert steel["metadata"]["tool_coverage"] == "partial"
    assert any(row["input"] == "华东钢铁集团有限公司" for row in train)


def test_configured_optimization_accepts_domain_override_hooks(tmp_path):
    pytest.importorskip("deepagents", reason="requires deepagents for real agent execution")
    example = _load_example_module()
    config_path = (
        Path(__file__).parents[1]
        / "examples"
        / "langchain_adapter"
        / "deepagents_gepa_configs"
        / "credit_approval.toml"
    )
    calls = {
        "dataset": 0,
        "evaluator": 0,
        "templates": 0,
        "selector": 0,
        "actionability": 0,
        "constraints": 0,
    }

    class DatasetProvider:
        def load(self):
            calls["dataset"] += 1
            row = {
                "input": "Borrower has negative operating cash flow and related-party receivables.",
                "rubric": "Expert opinion: identify repayment-capacity weakness and verification conditions.",
                "metadata": {"checkpoints": [{"label": "repayment capacity", "keywords": ["cash flow"]}]},
            }
            return [row], [row], [row]

    class Evaluator:
        def evaluate(self, example_row, state):
            del example_row, state
            calls["evaluator"] += 1
            return (
                0.6,
                "Override eval.\n"
                "- failure_classification: SKILL_DEFECT\n"
                "- suggested_component: skill:credit-risk-review:SKILL.md",
            )

    class TemplateRegistry:
        def templates_for(self, candidate):
            calls["templates"] += 1
            return {
                key: "Return only the selected component.\n<curr_param>\n<side_info>\n```" + value + "\n```"
                for key, value in candidate.items()
            }

    class Selector:
        def __call__(self, state, trajectories, subsample_scores, candidate_idx, candidate):
            del state, trajectories, subsample_scores, candidate_idx
            calls["selector"] += 1
            return ["skill:credit-risk-review:SKILL.md"] if "skill:credit-risk-review:SKILL.md" in candidate else []

    class ConstraintPolicy:
        def check(self, candidate, context):
            del candidate, context
            calls["constraints"] += 1
            return []

    class ActionabilityPolicy:
        def partition(self, examples, evaluation, *, regression_guard_limit):
            del evaluation, regression_guard_limit
            calls["actionability"] += 1
            return SimpleNamespace(
                actionable_indices=(0,),
                regression_guard_indices=(),
                tool_blocked_indices=(),
                satisfied_indices=(),
                other_indices=(),
                optimization_indices=tuple(range(len(examples))),
                fallback_to_unfiltered=False,
            )

    result = example.run_configured_skill_optimization(
        config_path,
        ToolFriendlyFakeChatModel(responses=["Credit risk review draft."] * 100),
        lambda _prompt: "```\n# 信贷审批风险审查\n\n审批前使用经过验证的现金流证据。\n```",
        dataset_provider=DatasetProvider(),
        evaluator=Evaluator(),
        template_registry=TemplateRegistry(),
        component_selector=Selector(),
        actionability_policy=ActionabilityPolicy(),
        constraint_policy=ConstraintPolicy(),
        max_metric_calls=5,
        reflection_minibatch_size=1,
        num_threads=1,
        use_reflection_judge=False,
        artifact_dir=tmp_path / "runs",
        artifact_run_name="credit-hooks",
    )

    assert result.best_candidate
    assert all(count > 0 for count in calls.values())
    artifact_summary = json.loads(
        (tmp_path / "runs" / "credit-hooks" / "result_summary.json").read_text(encoding="utf-8")
    )
    assert artifact_summary["preflight_actionability"]["evaluated_count"] == 1
    assert artifact_summary["overall_metric_calls"] > artifact_summary["total_metric_calls"]
    assert (
        artifact_summary["metric_calls_by_phase"]["gepa"]
        + artifact_summary["metric_calls_by_phase"]["preflight"]
        <= 5
    )
    assert artifact_summary["metric_budget_plan"]["proposal_budget_available"] is True


def test_golden_dataset_supports_rubric_without_expected(tmp_path):
    example = _load_example_module()
    project_root = tmp_path / "project"
    project_root.mkdir()
    dataset_path = project_root / "golden.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "input": "Check whether a receipt issue is billing.",
                        "data": "Evaluator-only expert material.",
                        "rubric": "Must route money issues.",
                    }
                ),
                json.dumps({"input": "Reset my password.", "expected": "account", "metadata": {"topic": "auth"}}),
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "deepagents_gepa.toml"
    _write_config(config_path, project_root, dataset_path)

    config = example.load_deepagents_gepa_config(config_path)
    rows = [record.as_example() for record in example.load_golden_jsonl(config)]

    assert rows[0]["rubric"] == "Must route money issues."
    assert rows[0]["data"] == "Evaluator-only expert material."
    assert "answer" not in rows[0]
    assert rows[1]["answer"] == "account"
    assert rows[1]["metadata"]["topic"] == "auth"


def test_dataset_level_rubric_is_applied_without_repeating_it_in_jsonl(tmp_path):
    example = _load_example_module()
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "golden.jsonl").write_text(
        "\n".join(json.dumps({"input": f"企业-{index}", "data": "专家风险点"}) for index in range(5)),
        encoding="utf-8",
    )
    config_path = tmp_path / "deepagents_gepa.toml"
    config_path.write_text(
        f"""
[agent]
project_root = "{project_root.as_posix()}"

[dataset]
source = "golden_jsonl"
path = "golden.jsonl"
rubric = "统一评价规则"
split_strategy = "random"
""",
        encoding="utf-8",
    )

    splits = example.load_dataset_from_config(example.load_deepagents_gepa_config(config_path))

    assert all(row["rubric"] == "统一评价规则" for split in splits for row in split)


def test_dataset_split_is_stratified_deterministic_and_disjoint():
    example = _load_example_module()
    rows = [
        {
            "input": f"企业-{index}",
            "metadata": {
                "difficulty": "hard" if index % 2 else "medium",
                "industry": "制造业" if index < 5 else "服务业",
            },
        }
        for index in range(10)
    ]

    first = example.split_examples(
        rows,
        stratify_by=("metadata.difficulty", "metadata.industry"),
        seed=17,
    )
    second = example.split_examples(
        list(reversed(rows)),
        stratify_by=("metadata.difficulty", "metadata.industry"),
        seed=17,
    )

    assert [len(split) for split in first] == [6, 2, 2]
    assert [{row["input"] for row in split} for split in first] == [{row["input"] for row in split} for split in second]
    assert not ({row["input"] for row in first[0]} & {row["input"] for row in first[1]})
    assert not ({row["input"] for row in first[0]} & {row["input"] for row in first[2]})
    assert all(row["metadata"]["dataset_split"] == "train" for row in first[0])
    assert all(row["metadata"]["dataset_stratum"] for split in first for row in split)


def test_dataset_split_distributes_tool_coverage_strata_across_splits():
    example = _load_example_module()
    rows = [
        {
            "input": f"{coverage}-{index}",
            "metadata": {
                "tool_coverage": coverage,
                **({"split": "train"} if coverage == "partial" and index == 0 else {}),
            },
        }
        for coverage in ("partial", "none")
        for index in range(4)
    ]

    train, val, test = example.split_examples(
        rows,
        stratify_by=("metadata.tool_coverage",),
        seed=17,
    )

    assert [len(split) for split in (train, val, test)] == [4, 2, 2]
    assert all(
        {row["metadata"]["tool_coverage"] for row in split} == {"partial", "none"} for split in (train, val, test)
    )


def test_dataset_split_honors_explicit_assignments():
    example = _load_example_module()
    rows = [
        {"input": "训练企业", "metadata": {"split": "train"}},
        {"input": "验证企业", "metadata": {"split": "val"}},
        {"input": "测试企业", "metadata": {"split": "test"}},
    ]

    train, val, test = example.split_examples(rows, split_strategy="explicit")

    assert [row["input"] for row in train] == ["训练企业"]
    assert [row["input"] for row in val] == ["验证企业"]
    assert [row["input"] for row in test] == ["测试企业"]


def test_golden_dataset_supports_evaluator_only_data_path(tmp_path):
    example = _load_example_module()
    project_root = tmp_path / "project"
    eval_dir = project_root / "evals"
    eval_dir.mkdir(parents=True)
    data_path = eval_dir / "risk_section.md"
    data_path.write_text("七、项目风险点\n1、行业周期性风险\n需要关注库存减值。", encoding="utf-8")
    dataset_path = eval_dir / "golden.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "input": "华东钢铁集团有限公司",
                "data_path": "risk_section.md",
                "rubric": "评价风险点覆盖。",
                "metadata": {
                    "trace_expectations": [{"label": "行业周期信息获取", "tool_intent_keywords": ["行业", "库存"]}]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "deepagents_gepa.toml"
    _write_config(config_path, project_root, dataset_path)

    config = example.load_deepagents_gepa_config(config_path)
    row = example.load_golden_jsonl(config)[0].as_example()
    messages = example.messages_for_example(row)

    assert row["data"].startswith("七、项目风险点")
    assert row["metadata"]["data_path"] == "risk_section.md"
    assert messages[0].content == "华东钢铁集团有限公司"
    assert "项目风险点" not in messages[0].content


def test_credit_risk_cleaner_extracts_project_risk_section(tmp_path):
    cleaner = _load_cleaner_module()
    source = tmp_path / "华东钢铁集团有限公司风险评价意见书.txt"
    source.write_text(
        "\n".join(
            [
                "六、项目情况",
                "企业主营钢铁冶炼加工。",
                "七、项目风险点",
                "1、钢铁行业周期性风险",
                "钢材和铁矿石价格波动, 库存减值压力持续存在。",
                "2、债务结构压力风险",
                "短期借款和应付票据规模较大, 财务费用侵蚀利润。",
                "八、风险评价人意见",
                "同意提交审议。",
            ]
        ),
        encoding="utf-8",
    )

    cleaned = cleaner.clean_one_file(source)

    assert cleaned is not None
    assert cleaned.company_name == "华东钢铁集团有限公司"
    assert cleaned.section_title == "项目风险点"
    assert "钢铁行业周期性风险" in cleaned.section_text
    assert "风险评价人意见" not in cleaned.section_text
    assert [item["label"] for item in cleaned.metadata["checkpoints"]] == [
        "钢铁行业周期性风险",
        "债务结构压力风险",
    ]
    assert any(item["label"] == "行业周期信息获取" for item in cleaned.metadata["trace_expectations"])


def test_credit_risk_cleaner_prefers_filename_and_validates_llm_tool_names(tmp_path):
    cleaner = _load_cleaner_module()
    source = tmp_path / "南城建材实业有限公司风险评价意见书.txt"
    source.write_text(
        "七、项目风险点\n1、地产链客户信用传导风险\n客户集中且应收账款回款放缓。\n八、审批意见",
        encoding="utf-8",
    )

    def extractor(_path, _section, _points, _inventory):
        return {
            "company_name": "LLM误识别企业",
            "checkpoints": [
                {
                    "label": "地产链客户信用传导风险",
                    "keywords": ["客户集中", "应收账款"],
                    "evidence_expectations": ["客户交易信息获取"],
                }
            ],
            "trace_expectations": [
                {
                    "label": "客户交易信息获取",
                    "tool_intent_keywords": ["客户", "回款"],
                    "tool_names": ["lookup_customers", "invented_tool"],
                }
            ],
        }

    cleaned = cleaner.clean_one_file(
        source,
        tool_inventory=[{"owner": "main", "name": "lookup_customers", "description": "查询企业客户集中度和回款信息。"}],
        metadata_extractor=extractor,
    )

    assert cleaned is not None
    assert cleaned.company_name == "南城建材实业有限公司"
    assert cleaned.metadata["trace_expectations"][0]["tool_names"] == ["lookup_customers"]
    assert cleaned.metadata["checkpoints"][0]["evidence_expectations"] == ["客户交易信息获取"]
    assert cleaned.metadata["tool_coverage"] == "complete"
    assert cleaned.metadata["tool_supported_checkpoint_count"] == 1
    assert cleaned.metadata["checkpoint_count"] == 1


def test_credit_risk_cleaner_excludes_action_only_checkpoints():
    cleaner = _load_cleaner_module()
    risk_points = [
        {
            "label": "地产链客户信用传导风险",
            "text": "客户信用恶化可能通过应收账款影响流动性。",
            "keywords": ["客户信用", "流动性"],
        },
        {
            "label": "授信压降和回款监管必要性",
            "text": "建议压降额度并加强回款监管。",
            "keywords": ["授信压降", "回款监管"],
        },
    ]

    checkpoints = cleaner.build_checkpoints(risk_points)
    expectations = cleaner.build_trace_expectations(risk_points)

    assert [item["label"] for item in checkpoints] == ["地产链客户信用传导风险"]
    assert all(item["label"] != "授信压降和回款监管必要性" for item in checkpoints)
    config_text = (
        Path(__file__).parents[1]
        / "examples"
        / "langchain_adapter"
        / "deepagents_gepa_configs"
        / "credit_approval.toml"
    ).read_text(encoding="utf-8")
    assert "未覆盖的 checkpoint 仍降低任务分数" in config_text
    assert "不要求或奖励审批意见" in config_text
    assert isinstance(expectations, list)


def test_credit_risk_cleaner_treats_explicit_consolidated_scope_limit_as_alternative_evidence(tmp_path):
    cleaner = _load_cleaner_module()
    source = tmp_path / "华东钢铁集团有限公司风险评价意见书.txt"
    source.write_text(
        "七、项目风险点\n"
        "1、集团内部信息不对称风险\n"
        "集团仅有合并口径报表, 未取得子公司单体财务和担保明细, 无法穿透各主体真实情况。\n"
        "八、审批意见",
        encoding="utf-8",
    )

    def extractor(_path, _section, _points, _inventory):
        return {
            "company_name": "LLM识别名称",
            "checkpoints": [
                {
                    "label": "集团内部信息不对称风险",
                    "keywords": ["信息不对称", "合并报表"],
                    "evidence_expectations": ["集团穿透信息获取"],
                    "evidence_mode": "all",
                }
            ],
            "trace_expectations": [
                {
                    "label": "集团穿透信息获取",
                    "tool_intent_keywords": ["集团", "子公司", "担保"],
                }
            ],
        }

    cleaned = cleaner.clean_one_file(
        source,
        tool_inventory=[
            {
                "owner": "main",
                "name": "lookup_financial_snapshot",
                "description": "查询企业财务快照以及数据口径和数据限制。",
            }
        ],
        metadata_extractor=extractor,
    )

    assert cleaned is not None
    scope_expectation = next(
        item for item in cleaned.metadata["trace_expectations"] if item["label"] == "财务口径限制信息获取"
    )
    assert scope_expectation["tool_names"] == ["lookup_financial_snapshot"]
    checkpoint = cleaned.metadata["checkpoints"][0]
    assert checkpoint["evidence_mode"] == "any"
    assert checkpoint["evidence_expectations"] == ["集团穿透信息获取", "财务口径限制信息获取"]
    assert cleaned.metadata["tool_coverage"] == "complete"


def test_langfuse_experience_mines_user_questions_without_trusting_final_answer(tmp_path):
    example = _load_example_module()
    config_path = tmp_path / "deepagents_gepa.toml"
    config_path.write_text(
        """
[agent]
project_root = "."

[dataset]
source = "langfuse_experience"
limit = 10
""",
        encoding="utf-8",
    )
    config = example.load_deepagents_gepa_config(config_path)
    traces = [
        {
            "id": "trace-1",
            "messages": [
                {"role": "user", "content": "Generate a due diligence report."},
                {"role": "assistant", "content": "Draft answer."},
                {"role": "user", "content": "不对, 继续验证供应商集中度风险。"},
            ],
            "output": "Possibly wrong final answer.",
        }
    ]

    rows = [record.as_example() for record in example.load_langfuse_records(config, langfuse_client=traces)]

    assert [row["input"] for row in rows] == [
        "Generate a due diligence report.",
        "不对, 继续验证供应商集中度风险。",
    ]
    assert all("answer" not in row for row in rows)
    assert rows[1]["metadata"]["experience_kind"] in {"correction", "risk_probe"}


def test_end_to_end_deep_agent_skill_self_optimization_from_config(tmp_path):
    pytest.importorskip("deepagents", reason="requires deepagents for real agent execution")
    example = _load_example_module()
    project_root = tmp_path / "project"
    example.create_seed_workspace(project_root)
    dataset_path = project_root / "golden.jsonl"
    dataset_path.write_text(
        json.dumps({"input": "I was charged twice for my invoice.", "expected": "billing"}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "deepagents_gepa.toml"
    _write_config(config_path, project_root, dataset_path)

    task_llm = ToolFriendlyFakeChatModel(responses=["<route>billing</route>"] * 100)

    def reflection_lm(prompt: str) -> str:
        assert "<curr_param>" not in prompt
        return "```\nYou are a support router. Always return a precise <route> tag.\n```"

    result = example.run_configured_skill_optimization(
        config_path,
        task_llm,
        reflection_lm,
        tool_registry={"tag_ticket": example.tag_ticket, "lookup_policy": example.lookup_policy},
        max_metric_calls=2,
        reflection_minibatch_size=1,
        num_threads=1,
    )

    assert result.best_candidate


def test_run_configured_optimization_writes_artifacts(tmp_path):
    pytest.importorskip("deepagents", reason="requires deepagents for real agent execution")
    example = _load_example_module()
    project_root = tmp_path / "project"
    example.create_seed_workspace(project_root)
    dataset_path = project_root / "golden.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps({"input": "I was charged twice for my invoice.", "expected": "billing"}),
                json.dumps({"input": "The invoice total is wrong.", "expected": "billing"}),
                json.dumps({"input": "Where can I download my receipt?", "expected": "billing"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "deepagents_gepa.toml"
    _write_config(config_path, project_root, dataset_path)

    task_llm = ToolFriendlyFakeChatModel(responses=["<route>billing</route>"] * 100)

    result = example.run_configured_skill_optimization(
        config_path,
        task_llm,
        lambda _prompt: "```\nYou are a support router. Always return a precise <route> tag.\n```",
        tool_registry={"tag_ticket": example.tag_ticket, "lookup_policy": example.lookup_policy},
        max_metric_calls=2,
        reflection_minibatch_size=1,
        num_threads=1,
        artifact_dir=tmp_path / "runs",
        artifact_run_name="artifact-test",
    )

    run_dir = tmp_path / "runs" / "artifact-test"
    assert result.best_candidate
    assert (run_dir / "result_summary.json").exists()
    assert (run_dir / "datasets" / "train.jsonl").exists()
    assert (run_dir / "candidates" / "0000" / "candidate.json").exists()
    assert (run_dir / "best_candidate" / "candidate.json").exists()
    assert (run_dir / "materialized_best_candidate" / "AGENTS.md").exists()
    rollout_rows = [
        json.loads(line)
        for line in (run_dir / "agent_logs" / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(row["fitness"] for row in rollout_rows)
    final_test = json.loads((run_dir / "final_test" / "summary.json").read_text(encoding="utf-8"))
    result_summary = json.loads((run_dir / "result_summary.json").read_text(encoding="utf-8"))
    assert final_test["count"] == 1
    assert result_summary["final_test"] == final_test
