from __future__ import annotations

from langchain_core.tools import tool


@tool
def lookup_policy(topic: str) -> str:
    """Look up internal credit review policy by topic."""
    policies = {
        "cashflow": "Repayment capacity must be based on verified operating cash flow and debt service coverage.",
        "collateral": "Collateral mitigates loss severity but does not replace borrower repayment capacity.",
        "guarantee": "Guarantee value depends on legal enforceability, guarantor capacity, and independence.",
        "industry": "Industry cyclicality and policy restrictions should be reflected in tenor and covenants.",
    }
    return policies.get(topic.lower(), "No matching policy note.")


@tool
def record_review_note(application_id: str, risk_dimension: str, note: str) -> str:
    """Record a credit approval review note for an application."""
    return f"{application_id}:{risk_dimension}:{note}"
