from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

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
                "feedback": "Scores:\n"
                "- failure_classification: TOOL_CAPABILITY_GAP\n"
                "- suggested_component: none",
            }
        ],
        [0.1],
        0,
        candidate,
    )

    assert selected == []


def test_default_evaluator_writes_fitness_back_to_original_state():
    example = _load_example_module()
    state = {"messages": []}

    def evaluate(_example, mutable_state):
        mutable_state["fitness"] = {"composite": 0.75}
        return 0.75, "ok"

    score, _feedback = example.DefaultEvaluator(evaluate).evaluate({"input": "x"}, state)

    assert score == 0.75
    assert state["fitness"] == {"composite": 0.75}


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
            "trace_expectations": [
                {"label": "行业周期信息获取", "tool_intent_keywords": ["钢铁行业周期", "库存减值"]}
            ]
        }
    }

    matched, missing, coverage = example.trace_expectation_results(row, state)

    assert matched == ["行业周期信息获取"]
    assert missing == []
    assert coverage == 1.0


def test_trace_expectation_rejects_ai_prose_and_failed_tool_results():
    example = _load_example_module()
    row = {
        "metadata": {
            "trace_expectations": [
                {"label": "司法工商信息获取", "tool_intent_keywords": ["司法", "被执行"]}
            ]
        }
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
    assert diagnostics["tool_capability_gaps"] == ["环保安监信息获取"]


def test_missing_data_tool_is_classified_as_tool_capability_gap():
    example = _load_example_module()
    row = {
        "input": "华东钢铁集团有限公司",
        "rubric": "核验环保处罚信息。",
        "metadata": {
            "trace_expectations": [
                {"label": "环保安监信息获取", "tool_intent_keywords": ["环保", "安全生产"]}
            ]
        },
    }
    state = {
        "messages": [example.AIMessage(content="目前缺少环保处罚数据。")],
        "baseline_response": "",
        "candidate_excerpt": {"skill:credit-risk-review:SKILL.md": "Review available evidence."},
        "candidate_constraints": [],
        "available_tools": [
            {"owner": "main", "name": "lookup_policy", "description": "查询内部授信政策。"}
        ],
    }

    _score, feedback = example.evaluate_response(row, state)

    assert state["fitness"]["failure_classification"] == "TOOL_CAPABILITY_GAP"
    assert state["fitness"]["tool_capability_gaps"] == ["环保安监信息获取"]
    assert "- failure_classification: TOOL_CAPABILITY_GAP" in feedback
    assert "- suggested_component: none" in feedback


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


def test_reflection_judge_score_is_capped_by_missing_rubric_checkpoints(tmp_path):
    example = _load_example_module()
    seed_spec = example.create_seed_workspace(tmp_path)
    candidate, surfaces = example.build_candidate_from_deep_agent_spec(seed_spec)
    constraints = example.validate_candidate_constraints(candidate, candidate, surfaces)
    state = {
        "messages": [
            example.AIMessage(
                content="该企业现金回款弱化, 放款前需要核验应收账款账龄。"
            )
        ],
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
    assert "- failure_classification: SKILL_DEFECT" in feedback
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
    assert "Expert data, rubrics, checkpoints" in template
    assert "under <side_info> is optimizer-only evidence" in template
    assert "A TOOL_CAPABILITY_GAP means no current text component" in template
    assert "Hidden-data boundary check" in template
    assert "Applicability scope and exclusions" in template
    assert "Cross-case regression risk" in template
    assert "trigger, evidence to obtain, analysis or comparison" in template
    assert "not one universal checklist" in template
    assert "Preserve the natural language used by the current component" in template


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
    assert "risk transmission" in template
    assert "borrower-specific acquisition plan" in template


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
        json.dumps({"best_val_score": 0.5, "val_aggregate_scores": [0.5]}),
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
    assert "skill:credit-risk-review:reference/financial_statement_analysis.md" in project.candidate
    assert "skill:credit-risk-review:reference/cashflow_and_repayment.md" in project.candidate
    assert "skill:credit-risk-review:reference/collateral_and_guarantee.md" in project.candidate
    assert "skill:credit-risk-review:reference/industry_management_and_warnings.md" in project.candidate
    assert "skill:credit-risk-review:reference/learned_expert_patterns.md" in project.candidate
    assert all("rubric" in row for row in rows)
    assert all("data" in row for row in rows)
    assert all("answer" not in row and "expected" not in row for row in rows)
    assert all("项目风险点" in row["data"] for row in rows)
    assert all("智能体仅根据企业名称自主检索" in row["rubric"] for row in rows)
    assert len(rows) >= 8
    assert all(row["metadata"].get("checkpoints") for row in rows)
    assert all(row["metadata"].get("trace_expectations") for row in rows)
    assert "取证计划" in project.candidate["skill:credit-risk-review:reference/learned_expert_patterns.md"]
    assert "同一触发条件" in project.candidate["skill:credit-risk-review:reference/learned_expert_patterns.md"]

    skill_constraints = example.skill_structure_constraints(
        "skill:credit-risk-review:SKILL.md",
        project.candidate["skill:credit-risk-review:SKILL.md"],
    )
    assert all(constraint.passed for constraint in skill_constraints)
    assert any(row["metadata"].get("scenario") == "钢铁集团授信_项目风险点对齐" for row in rows)
    assert any(row["input"] == "华东钢铁集团有限公司" for row in rows)


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
        lambda _prompt: "```\n# 信贷审批风险审查\n\n审批前使用经过验证的现金流证据。\n```",
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
    assert [{row["input"] for row in split} for split in first] == [
        {row["input"] for row in split} for split in second
    ]
    assert not ({row["input"] for row in first[0]} & {row["input"] for row in first[1]})
    assert not ({row["input"] for row in first[0]} & {row["input"] for row in first[2]})
    assert all(row["metadata"]["dataset_split"] == "train" for row in first[0])
    assert all(row["metadata"]["dataset_stratum"] for split in first for row in split)


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
                    "trace_expectations": [
                        {"label": "行业周期信息获取", "tool_intent_keywords": ["行业", "库存"]}
                    ]
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
