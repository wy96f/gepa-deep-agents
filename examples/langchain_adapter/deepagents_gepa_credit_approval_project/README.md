# Credit Approval GEPA Demo Project

This demo shows expert-risk-section optimization for a Chinese corporate-credit
approval agent.

Each dataset row gives the agent only a borrower name as `input`. The
approval-officer "project risk points" section is stored in `data`, which is
visible to the evaluator and reflection step but not passed to the agent during
rollout. GEPA should use the gap between the agent trace/output and the expert
section to improve reusable methodology in `skills/credit-risk-review/SKILL.md`
and `reference/*.md`.

The evaluator checks three things:

- whether the trace shows enough relevant data acquisition
- whether the final output covers the expert risk points
- whether the output explains the risk facts and transmission logic

If a trace expectation is missed, the evaluator also checks the available tool
names and descriptions. Missing expectations with tool support point to
skill/prompt/tool-description optimization. Tool capability gaps point to
external work: add or connect a tool before expecting GEPA text edits to fix
that data gap.

The goal is experience distillation:

- strengthen risk dimensions
- add missing failure modes
- encode approval conditions
- preserve the separation between skill workflow and reference methodology

The tool functions are placeholders for a real credit platform. The example
optimizes their descriptions, not their implementation.

Use `examples/langchain_adapter/clean_credit_risk_dataset.py` to turn many
approval-opinion files into this JSONL shape.
