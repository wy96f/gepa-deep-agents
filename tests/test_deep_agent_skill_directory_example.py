from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("langchain_core", reason="requires gepa[langchain] extra")
from langchain_core.language_models.fake_chat_models import FakeListChatModel


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


def test_boundary_gate_blocks_script_paths_outside_skill_definition(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path)
    candidate, surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)
    baseline_candidate = dict(candidate)
    candidate["main:system_prompt"] = "Use route_hint.py whenever routing is ambiguous. Return <route> tags."

    constraints = example.validate_candidate_constraints(candidate, baseline_candidate, surfaces)
    failure = next(
        constraint for constraint in constraints if constraint.name == "main:system_prompt:boundary:no_script_paths"
    )
    state = {
        "messages": [example.AIMessage(content="<route>billing</route>")],
        "baseline_response": "",
        "candidate_excerpt": candidate,
        "candidate_constraints": [constraint.__dict__ for constraint in constraints],
    }

    score, feedback = example.evaluate_response({"input": "I need my invoice.", "expected": "billing"}, state)

    assert failure.passed is False
    assert "route_hint.py" in failure.message
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
    assert "Optimize only the selected component" in template
    assert "rejected proposal lessons" in template
    assert "Proposal rationale" in template
    assert "Component boundary rules" in template
    assert "Selected component" in template
    assert "missing_rationale" in template
    assert "Do not start with a fenced code block" in template


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
            "fitness": {"hard": 1.0},
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
    assert (tmp_path / "run" / "proposals" / "0001" / "candidate.json").exists()
    assert (tmp_path / "run" / "proposals" / "0001" / "diff_against_parent.patch").exists()
    assert (tmp_path / "run" / "proposals" / "0001" / "proposal_rationale.json").exists()
    assert not (tmp_path / "run" / "proposals" / "0001" / "proposal_rationale_missing.json").exists()
    assert (tmp_path / "run" / "proposals" / "0001" / "prompts" / "memory__AGENTS.md.txt").exists()
    assert (tmp_path / "run" / "rejected_proposals" / "0001" / "candidate.json").exists()
    assert (tmp_path / "run" / "rejected_proposals" / "0001" / "diff_against_parent.patch").exists()
    assert (tmp_path / "run" / "rejected_proposals" / "0001" / "proposal_rationale.json").exists()
    assert "Recent rejected proposal lessons" in callback.rejected_history_prompt_block()


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
        Path(__file__).parents[1]
        / "examples"
        / "langchain_adapter"
        / "deepagents_gepa_configs"
        / f"{config_name}.toml"
    )
    config = example.load_deepagents_gepa_config(config_path)
    if config.agent_mode in {"manual", "langgraph_cli"}:
        pytest.importorskip("deepagents", reason="DeepAgents modes require deepagents")
    project = example.build_candidate_from_deep_agent_project(config)
    train, _val, _test = example.load_dataset_from_config(config)

    state = example.configured_rollout(
        project.candidate,
        train[0],
        ToolFriendlyFakeChatModel(responses=["<route>billing</route>"] * 30),
        project,
        project.candidate,
    )

    assert state.get("error") is None
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


def test_credit_approval_demo_loads_rubric_only_expert_dataset():
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
    assert "skill:credit-risk-review:reference/financial_statement_analysis.md" in project.candidate
    assert "skill:credit-risk-review:reference/cashflow_and_repayment.md" in project.candidate
    assert "skill:credit-risk-review:reference/collateral_and_guarantee.md" in project.candidate
    assert "skill:credit-risk-review:reference/industry_management_and_warnings.md" in project.candidate
    assert all("rubric" in row for row in rows)
    assert all("answer" not in row and "expected" not in row for row in rows)
    assert all("Expert risk opinion" in row["rubric"] for row in rows)


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
    calls = {"dataset": 0, "evaluator": 0, "templates": 0, "selector": 0, "constraints": 0}

    class DatasetProvider:
        def load(self):
            calls["dataset"] += 1
            row = {
                "input": "Borrower has negative operating cash flow and related-party receivables.",
                "rubric": "Expert opinion: identify repayment-capacity weakness and verification conditions.",
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
                key: "Return only the selected component.\n<curr_param>\n<side_info>\n```"
                + value
                + "\n```"
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

    result = example.run_configured_skill_optimization(
        config_path,
        ToolFriendlyFakeChatModel(responses=["Credit risk review draft."] * 100),
        lambda _prompt: "```\n# Credit Risk Review\n\nUse verified cash-flow evidence before approval.\n```",
        dataset_provider=DatasetProvider(),
        evaluator=Evaluator(),
        template_registry=TemplateRegistry(),
        component_selector=Selector(),
        constraint_policy=ConstraintPolicy(),
        max_metric_calls=2,
        reflection_minibatch_size=1,
        num_threads=1,
        use_reflection_judge=False,
        artifact_dir=tmp_path / "runs",
        artifact_run_name="credit-hooks",
    )

    assert result.best_candidate
    assert all(count > 0 for count in calls.values())


def test_golden_dataset_supports_rubric_without_expected(tmp_path):
    example = _load_example_module()
    project_root = tmp_path / "project"
    project_root.mkdir()
    dataset_path = project_root / "golden.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps({"input": "Check whether a receipt issue is billing.", "rubric": "Must route money issues."}),
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
    assert "answer" not in rows[0]
    assert rows[1]["answer"] == "account"
    assert rows[1]["metadata"]["topic"] == "auth"


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
        json.dumps({"input": "I was charged twice for my invoice.", "expected": "billing"}) + "\n",
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
