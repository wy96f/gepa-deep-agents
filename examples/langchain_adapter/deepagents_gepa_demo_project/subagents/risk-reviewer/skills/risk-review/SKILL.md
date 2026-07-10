---
name: risk-review
description: Review support-routing decisions for missing evidence and risky misroutes.
---

# Risk Review

Use this skill when a routing decision needs a second pass.

## Workflow

1. Read `reference/risk_rules.md`.
2. Run `python scripts/check_route.py` if the decision is ambiguous.
3. Identify one accepted route and one rejected alternative.

## Guardrails

- Never approve a route that contradicts the strongest user signal.
- Do not invent evidence that is not present in the ticket.
- Escalate ambiguous money or access issues instead of guessing.
