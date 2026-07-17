"""Protocol layer for Deep Agents GEPA projects.

These interfaces keep the example Deep Agents-specific while making the parts
that vary between projects explicit: datasets, evaluation, reflection templates,
component selection, constraints, materialization, and runner behavior.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

SOFTENER_PATTERN = re.compile(
    r"\b(consider|maybe|perhaps|flexibly|as appropriate|if possible|you may want to)\b",
    re.I,
)
SUGGESTED_COMPONENT_RE = re.compile(r"(?m)^-\s*suggested_component:\s*(?P<component>\S+)\s*$")


class DatasetProvider(Protocol):
    """Load train/val/test examples for an optimization run."""

    def load(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Return train, validation, and test splits."""


@dataclass
class DefaultDatasetProvider:
    """Default dataset provider backed by the config loader function."""

    config: Any
    load_dataset: Callable[..., tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]]
    langfuse_client: Any | None = None

    def load(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        return self.load_dataset(self.config, langfuse_client=self.langfuse_client)


class Evaluator(Protocol):
    """Score one rollout and produce feedback for reflection."""

    def evaluate(self, example: Mapping[str, Any], state: Mapping[str, Any]) -> tuple[float, str]:
        """Return a GEPA-compatible score and feedback string."""


@dataclass
class DefaultEvaluator:
    """Default evaluator adapter backed by a scoring function."""

    evaluate_fn: Callable[[dict[str, Any], dict[str, Any]], tuple[float, str]]

    def evaluate(self, example: Mapping[str, Any], state: Mapping[str, Any]) -> tuple[float, str]:
        mutable_state = state if isinstance(state, MutableMapping) else dict(state)
        return self.evaluate_fn(dict(example), mutable_state)


class ComponentSelector(Protocol):
    """Choose which text component GEPA should mutate next."""

    def __call__(
        self,
        state: Any,
        trajectories: list[Any],
        subsample_scores: list[float],
        candidate_idx: int,
        candidate: dict[str, str],
    ) -> list[str]:
        """Return one or more candidate keys to mutate."""


class DefaultFeedbackComponentSelector:
    """Pick evaluator-suggested components, with cooldown and fallback."""

    def __init__(self, cooldown_after: int = 2) -> None:
        self._fallback_offsets: dict[int, int] = {}
        self._last_selected: dict[int, str] = {}
        self._repeat_counts: dict[int, int] = {}
        self._cooldown_after = max(1, cooldown_after)

    def __call__(
        self,
        state: Any,
        trajectories: list[Any],
        subsample_scores: list[float],
        candidate_idx: int,
        candidate: dict[str, str],
    ) -> list[str]:
        del state, subsample_scores
        suggested: dict[str, dict[str, float]] = {}
        actionable_trajectories = 0
        capability_gap_trajectories = 0
        sorted_trajectories = sorted(
            trajectories or [],
            key=lambda item: float(item.get("score", 0.0)) if isinstance(item, dict) else 0.0,
        )
        for trajectory in sorted_trajectories:
            if not isinstance(trajectory, dict):
                continue
            score = float(trajectory.get("score", 0.0))
            feedback = str(trajectory.get("feedback", ""))
            if self._failure_classification(feedback) == "TOOL_CAPABILITY_GAP":
                capability_gap_trajectories += 1
                continue
            actionable_trajectories += 1
            component = self._component_from_feedback(feedback, candidate)
            if component is not None:
                vote = suggested.setdefault(component, {"count": 0.0, "weight": 0.0, "min_score": 1.0})
                vote["count"] += 1
                vote["weight"] += 1.0 + max(0.0, 1.0 - score)
                vote["min_score"] = min(vote["min_score"], score)
        if suggested:
            ranked_components = sorted(
                suggested,
                key=lambda component: (
                    suggested[component]["count"],
                    suggested[component]["weight"],
                    -suggested[component]["min_score"],
                ),
                reverse=True,
            )
            for component in ranked_components:
                if not self._is_cooled_down(candidate_idx, component):
                    return [self._record_selection(candidate_idx, component)]
            return [
                self._record_selection(
                    candidate_idx,
                    self._round_robin_fallback(candidate_idx, candidate, excluded=set(ranked_components)),
                )
            ]
        if capability_gap_trajectories and not actionable_trajectories:
            return []
        return [self._record_selection(candidate_idx, self._round_robin_fallback(candidate_idx, candidate))]

    @staticmethod
    def _failure_classification(feedback: str) -> str | None:
        match = re.search(r"(?m)^-\s*failure_classification:\s*(\S+)\s*$", feedback)
        return match.group(1).strip().rstrip(".,;") if match else None

    def _component_from_feedback(self, feedback: str, candidate: dict[str, str]) -> str | None:
        for match in SUGGESTED_COMPONENT_RE.finditer(feedback):
            raw_component = match.group("component").strip().rstrip(".,;")
            if raw_component in candidate:
                return raw_component
            matches = [key for key in candidate if raw_component.startswith(f"{key}:")]
            if matches:
                return max(matches, key=len)
        return None

    def _is_cooled_down(self, candidate_idx: int, component: str) -> bool:
        return self._last_selected.get(candidate_idx) == component and (
            self._repeat_counts.get(candidate_idx, 0) >= self._cooldown_after
        )

    def _record_selection(self, candidate_idx: int, component: str) -> str:
        if self._last_selected.get(candidate_idx) == component:
            self._repeat_counts[candidate_idx] = self._repeat_counts.get(candidate_idx, 0) + 1
        else:
            self._last_selected[candidate_idx] = component
            self._repeat_counts[candidate_idx] = 1
        return component

    def _round_robin_fallback(
        self,
        candidate_idx: int,
        candidate: dict[str, str],
        excluded: set[str] | None = None,
    ) -> str:
        excluded = excluded or set()
        keys = [key for key in candidate if key not in excluded]
        if not keys:
            keys = list(candidate)
        if not keys:
            raise ValueError("candidate must contain at least one component")
        offset = self._fallback_offsets.get(candidate_idx, 0)
        self._fallback_offsets[candidate_idx] = offset + 1
        return keys[offset % len(keys)]


class ReflectionTemplateRegistry(Protocol):
    """Provide reflection prompts for candidate surfaces."""

    def templates_for(self, candidate: Mapping[str, str]) -> dict[str, str]:
        """Return a reflection template per candidate key."""


class DefaultReflectionTemplateRegistry:
    """Default per-surface reflection templates for Deep Agents text surfaces."""

    def templates_for(self, candidate: Mapping[str, str]) -> dict[str, str]:
        templates: dict[str, str] = {}
        common_rules = (
            "General optimization rules:\n"
            "- Optimize only the selected component as a drop-in replacement.\n"
            "- Make the smallest change that plausibly fixes the observed failures.\n"
            "- Preserve the component's scope; do not paste unrelated skills, references, tool descriptions, or prompts.\n"
            "- Treat rejected proposal lessons, when present, as negative evidence: fix the failure pattern without copying rejected text.\n"
            "- Prefer reusable instructions, decision criteria, and guardrails over test-specific answers.\n"
            "- Expert data, rubrics, checkpoints, and evaluator feedback are hidden from the runtime agent. Distill "
            "reusable lessons from them, but never tell the runtime agent to read or respond to those hidden fields.\n"
            "- Evaluation data under <side_info> is optimizer-only evidence. It was not part of the runtime conversation. "
            "Do not claim the agent saw it, failed to read it, or should request it as a named runtime field.\n"
            "- A TOOL_CAPABILITY_GAP means no current text component can obtain the missing evidence. Do not invent a "
            "tool, imply that a prompt adds the capability, or encode unavailable facts as instructions.\n"
            "- Do not universalize a rule from one example. State observable applicability signals, relevant industries "
            "or business models, and non-applicable cases. Use examples to clarify scope, not as a closed hardcoded list.\n"
            "- Applicability signals are conditional observations, not one universal checklist. Evidence to obtain is a "
            "borrower-specific acquisition plan, not a fixed value known in advance. Adapt both to the current business "
            "model, mark unsupported acquisition as TOOL_CAPABILITY_GAP, and keep an unverified risk as a hypothesis.\n"
            "- Check cross-example regression risk. If a rule helps one scope but may hurt another, make the condition "
            "explicit instead of applying the rule globally.\n"
            "- Every added rule must be operational: include a trigger, evidence to obtain, analysis or comparison, "
            "risk transmission, and an approval action or verification consequence.\n"
            "- Avoid empty guidance such as 'analyze comprehensively', 'strengthen attention', or 'verify as needed' "
            "unless it is followed by concrete evidence, method, and decision criteria.\n"
            "- Keep names, paths, tool names, and output contracts consistent with the current component and feedback.\n"
            "- Preserve the natural language used by the current component unless the feedback explicitly requires a "
            "language change.\n"
            "- Respect size and growth constraints; remove duplication before adding new material."
        )
        for key in candidate:
            instruction = self._component_instruction(key)
            boundary_rules = self._component_boundary_rules(key)
            templates[key] = (
                f"{instruction}\n\n"
                "Use the full evaluation data as a global map of the Deep Agents project. It may include other "
                "component excerpts so you can diagnose where the failure belongs. Do not treat those excerpts as "
                "text to paste into the selected component.\n\n"
                f"{common_rules}\n\n"
                f"Component boundary rules for `{key}`:\n"
                f"{boundary_rules}\n\n"
                "Your response must have exactly two sections, in this order. Do not start with a fenced code block.\n\n"
                "Proposal rationale:\n"
                "- Failure pattern:\n"
                "- Evidence across examples:\n"
                "- Selected component:\n"
                "- Why this component:\n"
                "- Why not other components:\n"
                "- Applicability scope and exclusions:\n"
                "- Cross-case regression risk:\n"
                "- Operational rule shape:\n"
                "- Boundary checks:\n"
                "- Hidden-data boundary check:\n"
                "- Intended behavior change:\n\n"
                "Final replacement:\n"
                "Use exactly one fenced code block after `Final replacement:`. Do not use triple backticks anywhere "
                "before that final section. The fenced block must be the only fenced code block and must contain only "
                "the replacement text for the selected component. If you omit `Proposal rationale:`, the run artifact "
                "will mark this proposal as missing_rationale.\n\n"
                "Current component:\n```\n<curr_param>\n```\n\n"
                "Evaluation data and feedback:\n```\n<side_info>\n```\n\n"
                "Now write both required sections."
            )
        return templates

    def _component_instruction(self, key: str) -> str:
        if key.startswith("memory:"):
            return (
                "Return a complete AGENTS.md memory file. Keep it concise and focused on durable operating "
                "instructions, routing priorities, and when to use existing skills/tools. Do not copy SKILL.md, "
                "reference/*.md, tool descriptions, or subagent prompts into AGENTS.md; refer to them by name "
                "instead. Avoid large growth unless the feedback explicitly requires it."
            )
        if key.endswith(":SKILL.md"):
            return (
                "Return a complete SKILL.md file. Preserve valid YAML frontmatter with name and description. Keep the "
                "skill focused on invariant workflow, resource routing, failure modes, and guardrails. Put scoped "
                "industry or business-model knowledge in the most specific reference/*.md component instead."
            )
        if ":reference/" in key:
            return (
                "Return a complete Markdown reference file containing reusable scoped rules, facts, examples, or lookup "
                "tables. Each learned rule should state applicability signals, evidence, analysis method, risk "
                "transmission, decision use, and exclusions."
            )
        if ":tool:" in key and key.endswith(":description"):
            return "Return only a concise tool description. Explain when to call it, parameters, and boundaries."
        if key.startswith("subagent:") and key.endswith(":description"):
            return "Return only a concise delegation description for when the main agent should use this subagent."
        if key.startswith("subagent:") and key.endswith(":system_prompt"):
            return "Return a complete subagent system prompt with tool usage guidance and output expectations."
        if key == "main:system_prompt":
            return "Return a complete main agent system prompt focused on global behavior and routing strategy."
        return "Return a complete improved text component."

    def _component_boundary_rules(self, key: str) -> str:
        if key == "main:system_prompt":
            return (
                "- Scope: identity, global policy, tool/skill usage strategy, and final output contract.\n"
                "- Do not include YAML frontmatter, SKILL.md bodies, reference tables, or copied memory.\n"
                "- Point the agent toward existing skills/references/tools instead of embedding their full content.\n"
                "- Keep it short enough that the skill directory remains the source of task knowledge."
            )
        if key.startswith("memory:"):
            return (
                "- Scope: durable project memory and stable operating preferences.\n"
                "- Do not include YAML frontmatter, full SKILL.md content, reference files, tool descriptions, or subagent prompts.\n"
                "- Mention when to consult existing skills/tools; do not duplicate their implementation details."
            )
        if key.endswith(":SKILL.md"):
            return (
                "- Scope: reusable skill definition, workflow, failure modes, and guardrails.\n"
                "- Preserve YAML frontmatter with name and description.\n"
                "- You may reference local reference/*.md and scripts/*.py paths that actually belong to this skill.\n"
                "- Keep domain catalogs and industry-specific risk patterns in reference/*.md; route to them from the "
                "workflow instead of accumulating them in SKILL.md.\n"
                "- Add a domain rule here only when it is invariant across the skill's intended use cases.\n"
                "- Do not paste AGENTS.md, system prompts, tool descriptions, or unrelated subagent skills."
            )
        if ":reference/" in key:
            return (
                "- Scope: reusable domain facts, rules, rubrics, examples, and lookup material.\n"
                "- Do not write agent persona, delegation instructions, tool descriptions, or workflow prose here.\n"
                "- Keep references structured for lookup and reuse by a skill.\n"
                "- Scope each learned pattern by observable signals or business model. Include non-applicability "
                "conditions so an improvement for one sector does not become a global rule.\n"
                "- Make each rule concrete enough to execute: evidence source, comparison or calculation, risk "
                "transmission, and resulting verification or approval action."
            )
        if ":tool:" in key and key.endswith(":description"):
            return (
                "- Scope: one short description of when to call this tool, what parameters mean, and its boundaries.\n"
                "- Do not include workflows, examples, YAML frontmatter, or copied skill/reference text.\n"
                "- Keep it concise; the tool implementation is not being optimized."
            )
        if key.startswith("subagent:") and key.endswith(":description"):
            return (
                "- Scope: a short delegation trigger for the main agent.\n"
                "- Do not include the subagent system prompt, skills, references, or tool descriptions."
            )
        if key.startswith("subagent:") and key.endswith(":system_prompt"):
            return (
                "- Scope: subagent role, available tool/skill usage, decision criteria, and output expectations.\n"
                "- Do not copy full SKILL.md or reference files; tell the subagent when to consult them."
            )
        return "- Preserve the selected component's runtime role and do not paste unrelated components into it."


class Constraint(Protocol):
    """Validate a candidate or one of its surfaces."""

    def check(self, candidate: Mapping[str, str], context: Mapping[str, Any]) -> Sequence[Any]:
        """Return constraint results for the candidate."""


@dataclass
class DefaultConstraintSet:
    """Default constraint adapter backed by the example validator function."""

    validate_fn: Callable[..., Sequence[Any]]
    baseline_candidate: Mapping[str, str]
    surfaces: Mapping[str, Any]

    def check(self, candidate: Mapping[str, str], context: Mapping[str, Any]) -> Sequence[Any]:
        return self.validate_fn(
            dict(candidate),
            dict(self.baseline_candidate),
            dict(self.surfaces),
            materialized_root=context.get("materialized_root"),
        )


class CandidateMaterializer(Protocol):
    """Write an in-memory candidate into a runnable temporary project tree."""

    def materialize(self, candidate: Mapping[str, str], output_dir: Path) -> Any:
        """Return the runtime application/spec produced from the candidate."""


@dataclass
class DefaultCandidateMaterializer:
    """Default materializer backed by a project-specific apply function."""

    apply_fn: Callable[[Mapping[str, str], Path], Any]

    def materialize(self, candidate: Mapping[str, str], output_dir: Path) -> Any:
        return self.apply_fn(candidate, output_dir)
