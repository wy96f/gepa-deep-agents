---
name: support-router
description: Route support requests to billing, account, engineering, or product.
---

# Support Router

Use this skill for support-routing tasks.

## Workflow

1. Read `reference/routing.md`.
2. If the route is ambiguous, run `python scripts/route_hint.py`.
3. Return one `<route>TEAM</route>` tag.

## Guardrails

- Do not route invoices, charges, receipts, or refunds to product.
- Do not route password or login issues to billing.
- Do not route crashes or errors to account.
