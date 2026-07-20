"""Protocol layer for Deep Agents GEPA projects.

These interfaces keep the example Deep Agents-specific while making the parts
that vary between projects explicit: datasets, evaluation, reflection templates,
component selection, constraints, materialization, and runner behavior.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from math import isclose
from pathlib import Path
from typing import Any, Callable, Protocol

SOFTENER_PATTERN = re.compile(
    r"\b(consider|maybe|perhaps|flexibly|as appropriate|if possible|you may want to)\b",
    re.I,
)
SUGGESTED_COMPONENT_RE = re.compile(r"(?m)^-\s*suggested_component:\s*(?P<component>\S+)\s*$")
MUTATION_ELIGIBLE_RE = re.compile(r"(?mi)^-\s*mutation_eligible:\s*(?P<eligible>true|false)\s*$")
NON_ACTIONABLE_FAILURES = frozenset(
    {
        "TOOL_CAPABILITY_GAP",
        "INSUFFICIENT_RUNTIME_EVIDENCE",
        "NO_FAILURE",
    }
)


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
        non_actionable_trajectories = 0
        sorted_trajectories = sorted(
            trajectories or [],
            key=lambda item: float(item.get("score", 0.0)) if isinstance(item, dict) else 0.0,
        )
        for trajectory in sorted_trajectories:
            if not isinstance(trajectory, dict):
                continue
            score = float(trajectory.get("score", 0.0))
            feedback = str(trajectory.get("feedback", ""))
            failure_classification = self._failure_classification(feedback)
            if failure_classification in NON_ACTIONABLE_FAILURES or self._mutation_eligible(feedback) is False:
                non_actionable_trajectories += 1
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
                    selected = self._record_selection(candidate_idx, component)
                    return self._with_component_dependencies(selected, candidate)
            return [
                self._record_selection(
                    candidate_idx,
                    self._round_robin_fallback(candidate_idx, candidate, excluded=set(ranked_components)),
                )
            ]
        if non_actionable_trajectories and not actionable_trajectories:
            return []
        return [self._record_selection(candidate_idx, self._round_robin_fallback(candidate_idx, candidate))]

    @staticmethod
    def _with_component_dependencies(component: str, candidate: Mapping[str, str]) -> list[str]:
        """Add the owning workflow only when it does not yet route to the learned reference."""
        selected = [component]
        if ":reference/" not in component:
            return selected
        reference_name = component.rsplit(":reference/", maxsplit=1)[-1].lower()
        if not any(marker in reference_name for marker in ("learned", "expert", "experience")):
            return selected
        skill_component = component.split(":reference/", maxsplit=1)[0] + ":SKILL.md"
        skill_text = str(candidate.get(skill_component, "")).lower()
        if skill_component in candidate and reference_name not in skill_text:
            selected.append(skill_component)
        return selected

    @staticmethod
    def _failure_classification(feedback: str) -> str | None:
        match = re.search(r"(?m)^-\s*failure_classification:\s*(\S+)\s*$", feedback)
        return match.group(1).strip().rstrip(".,;") if match else None

    @staticmethod
    def _mutation_eligible(feedback: str) -> bool | None:
        match = MUTATION_ELIGIBLE_RE.search(feedback)
        if match is None:
            return None
        return match.group("eligible").lower() == "true"

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
            "- Use the trajectory diagnosis before proposing text: an available tool that was never called needs a "
            "scoped call trigger; an argument/validation error may need tool-description or call-guidance changes; a "
            "runtime/upstream tool failure needs tool or environment repair; a successful but insufficient result needs "
            "query/result review; and relevant successful evidence omitted from the answer may justify a skill/reference "
            "change. Do not collapse these cases into one generic skill defect.\n"
            "- Tool gaps can coexist with text-actionable failures. Improve missing reusable methodology or supported "
            "tool usage when feedback selects this component, while preserving each unavailable source as an explicit "
            "tool backlog item rather than pretending the replacement can retrieve it.\n"
            "- Do not universalize a rule from one example. State observable applicability signals, relevant industries "
            "or business models, and non-applicable cases. Use examples to clarify scope, not as a closed hardcoded list.\n"
            "- A company name or keyword is only a weak discovery clue. Do not use it alone to activate a risk "
            "conclusion; require business-model, transaction, financial, asset, or financing evidence.\n"
            "- Do not persist evaluator-only company names, dates, amounts, or invented numeric thresholds. A threshold "
            "must cite an applicable policy or evidence basis; otherwise express an entity-relative comparison or a "
            "clearly labeled adjustable stress scenario.\n"
            "- Applicability signals are conditional observations, not one universal checklist. Evidence to obtain is a "
            "task-specific acquisition plan, not a fixed value known in advance. Adapt both to the current business "
            "model and mark unsupported acquisition as TOOL_CAPABILITY_GAP. Follow the current output contract when "
            "deciding whether uncertainty should be omitted or briefly labeled; never present it as fact.\n"
            "- Check cross-example regression risk. If a rule helps one scope but may hurt another, make the condition "
            "explicit instead of applying the rule globally.\n"
            "- Add only the knowledge increment the runtime model is unlikely to recover reliably on its own. Prefer a "
            "short signal -> concern -> consequence reminder over a full textbook explanation.\n"
            "- Keep operational detail proportional to what is non-obvious. Include evidence, comparison, consequence, "
            "or a task-required decision criterion "
            "only when it changes how the agent should investigate or decide; do not expand every reminder into a "
            "fixed multi-section template.\n"
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
                "- Runtime trajectory diagnosis:\n"
                "- Recommended remediation category:\n"
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
                "Evaluation data and feedback:\n```\n<side_info>\n```\n\n"
                f"Authoritative target component: `{key}`\n"
                "Current target component (this is the only text you may replace):\n"
                "```\n<curr_param>\n```\n\n"
                f"Before answering, verify that the replacement is valid for `{key}` and not for any component "
                "suggested inside the evaluation evidence. Now write both required sections."
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
                "Return a complete Markdown reference file containing reusable scoped reminders, facts, examples, or "
                "lookup tables. For a learned reference, keep each new pattern compact: normally a heading plus one "
                "short paragraph or a few bullets describing when it matters, what to notice, and the likely "
                "consequence. Rely on the runtime model's general knowledge for standard analysis steps."
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
                "- When evaluation evidence points to a learned expert reference, add a concrete applicability trigger "
                "and lookup step for that reference without copying its domain rules into SKILL.md.\n"
                "- Add a domain rule here only when it is invariant across the skill's intended use cases.\n"
                "- Do not paste AGENTS.md, system prompts, tool descriptions, or unrelated subagent skills."
            )
        if ":reference/" in key:
            return (
                "- Scope: reusable domain facts, rules, rubrics, examples, and lookup material.\n"
                "- Do not write agent persona, delegation instructions, tool descriptions, or workflow prose here.\n"
                "- Keep references structured for lookup and reuse by a skill.\n"
                "- For a learned reference, merge evidence by shared economic mechanism and add at most a few focused "
                "reminders per proposal; do not append one section per evaluation example, company, or industry.\n"
                "- Scope each learned pattern by observable signals or business model. Include non-applicability "
                "conditions so an improvement for one sector does not become a global rule.\n"
                "- Treat entity-name keywords as discovery clues only, never sufficient applicability evidence.\n"
                "- Do not invent fixed cutoffs. Tie thresholds to policy/evidence or label them as adjustable stress "
                "assumptions.\n"
                "- Prefer compact signal -> concern -> consequence language. Add an evidence source, comparison, "
                "or exclusion only when it is non-obvious and materially changes execution. Do not add recommendations "
                "or decisions that the current agent output contract excludes.\n"
                "- Do not repeat generic finance, risk, or investigation knowledge that a capable runtime model already "
                "knows. Preserve context budget for genuinely learned expert cues."
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


@dataclass(frozen=True)
class ProposalReview:
    """Structured result from a pre-runtime proposal quality review."""

    decision: str
    issues: tuple[str, ...]
    reviewed_response: str | None
    raw_output: str


class ProposalReviewer(Protocol):
    """Review a reflected proposal before GEPA evaluates the candidate."""

    def review(
        self,
        *,
        reflection_prompt: str,
        proposal_response: str,
        review_lm: Callable[[str], str],
    ) -> ProposalReview:
        """Return ACCEPT, REVISE, or REJECT with optional corrected response."""


class DefaultProposalReviewer:
    """LLM reviewer for component ownership, learnability, and overfitting risk."""

    def review(
        self,
        *,
        reflection_prompt: str,
        proposal_response: str,
        review_lm: Callable[[str], str],
    ) -> ProposalReview:
        prompt = self._build_prompt(reflection_prompt, proposal_response)
        raw_output = str(review_lm(prompt))
        return self._parse_output(raw_output)

    @staticmethod
    def _build_prompt(reflection_prompt: str, proposal_response: str) -> str:
        return (
            "You are the final proposal reviewer for a Deep Agents text-surface optimizer. Review the proposed "
            "drop-in replacement before any expensive agent rollout.\n\n"
            "Judge the proposal holistically:\n"
            "1. Mutation eligibility: hidden evaluator facts may score behavior, but they may drive a text mutation "
            "only when runtime evidence, an available-but-skipped tool, or repeated independent examples support a "
            "reusable lesson. Missing external data alone is a tool backlog item, not knowledge to memorize.\n"
            "2. Trajectory attribution: distinguish a missing tool, a skipped tool, bad call arguments, runtime/upstream "
            "failure, insufficient tool results, and successful evidence omitted from analysis. REJECT a text mutation "
            "when the remediation belongs only to tool implementation, credentials, dependencies, or dataset mapping.\n"
            "3. Component ownership: keep global behavior in prompts/memory, invariant workflow in SKILL.md, compact "
            "domain cues in reference files, and invocation semantics in tool descriptions.\n"
            "4. Generalization: reject company-specific answers, entity-name-only triggers, closed industry lists, and "
            "rules extrapolated from one case without observable conditions.\n"
            "5. Evidence integrity: reject unsupported fixed thresholds, dates, amounts, facts, or tool capabilities.\n"
            "6. Output contract: preserve what the runtime agent is asked to produce. Reject or revise additions such "
            "as recommendations, decisions, approval opinions, or long missing-data lists when the current prompt/skill "
            "explicitly excludes them.\n"
            "7. Minimality: preserve existing good text and make the smallest coherent edit. For learned references, "
            "add at most a few compact signal -> concern -> consequence reminders. Do not produce long mechanism / "
            "evidence / analysis / status / transmission / action templates when the runtime model already knows "
            "those standard steps. For prompts and AGENTS.md, prefer one precise instruction over repeated prose.\n"
            "8. Regression risk: contain a scoped lesson with observable applicability signals; do not improve one "
            "sample by imposing a universal checklist on unrelated samples.\n\n"
            "Return exactly this format:\n"
            "Decision: ACCEPT | REVISE | REJECT\n"
            "Issues:\n"
            "- concise issue, or `none`\n"
            "Reviewed response:\n"
            "<for REVISE, a complete replacement response with `Proposal rationale:` and `Final replacement:`; "
            "otherwise write `same`>\n\n"
            "REJECT means the evidence does not justify any text mutation. REVISE means return a complete corrected "
            "proposal, not commentary or a patch.\n\n"
            "Original optimization prompt:\n"
            f"{reflection_prompt}\n\n"
            "Original proposal:\n"
            f"{proposal_response}"
        )

    @staticmethod
    def _parse_output(raw_output: str) -> ProposalReview:
        decision_match = re.search(r"(?mi)^Decision:\s*(ACCEPT|REVISE|REJECT)\s*$", raw_output)
        decision = decision_match.group(1).upper() if decision_match else "ACCEPT"
        issues_match = re.search(
            r"(?ms)^Issues:\s*(.*?)\nReviewed response:\s*(.*)\Z",
            raw_output,
        )
        issues_text = issues_match.group(1).strip() if issues_match else ""
        reviewed_text = issues_match.group(2).strip() if issues_match else ""
        issues = tuple(
            line.removeprefix("-").strip()
            for line in issues_text.splitlines()
            if line.removeprefix("-").strip().lower() not in {"", "none", "`none`"}
        )
        reviewed_response = None
        if decision == "REVISE" and reviewed_text.lower() != "same":
            if "Proposal rationale:" in reviewed_text and "Final replacement:" in reviewed_text:
                reviewed_response = reviewed_text
            else:
                decision = "ACCEPT"
                issues = (*issues, "reviewer revision was malformed; original proposal retained")
        return ProposalReview(
            decision=decision,
            issues=issues,
            reviewed_response=reviewed_response,
            raw_output=raw_output,
        )


def select_deployment_candidate_index(result: Any, *, score_tolerance: float = 1e-12) -> int | None:
    """Select the first validation winner, preserving the incumbent on a tie.

    A validation tie is not evidence that a newer candidate is better. All
    accepted candidates remain in the artifacts for review, but deployment
    follows GEPA's conservative first-maximum behavior.
    """
    scores = list(getattr(result, "val_aggregate_scores", []) or [])
    candidates = list(getattr(result, "candidates", []) or [])
    usable_count = min(len(scores), len(candidates))
    if usable_count == 0:
        return None
    best_score = max(float(score) for score in scores[:usable_count])
    tied_indices = [
        index
        for index, score in enumerate(scores[:usable_count])
        if isclose(float(score), best_score, rel_tol=0.0, abs_tol=max(0.0, score_tolerance))
    ]
    return min(tied_indices)


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
