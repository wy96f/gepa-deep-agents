from langchain_core.tools import tool


@tool
def tag_ticket(ticket: str, route: str) -> str:
    """Apply a support route label to a ticket."""
    return f"{ticket} -> {route}"


@tool
def lookup_policy(topic: str) -> str:
    """Look up routing policy evidence for a support topic."""
    policies = {
        "billing": "Invoices, refunds, duplicate charges, receipts, and plan changes route to billing.",
        "account": "Login, password, authentication, and locked-access issues route to account.",
        "engineering": "Crashes, bugs, errors, and broken features route to engineering.",
        "product": "Feature requests, integrations, and roadmap questions route to product.",
    }
    return policies.get(topic.lower(), "No policy evidence found.")
