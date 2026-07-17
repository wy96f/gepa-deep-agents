from __future__ import annotations

from langchain_core.tools import tool


@tool
def lookup_policy(topic: str) -> str:
    """按主题查询内部信贷审查政策; 主题可为现金流、抵质押、保证或行业."""
    policies = {
        "现金流": "偿债能力必须以经过验证的经营现金流和债务覆盖能力为基础.",
        "抵质押": "抵质押用于降低违约损失, 不能替代借款人的偿债能力.",
        "保证": "保证的缓释价值取决于法律可执行性、保证人偿付能力及其独立性.",
        "行业": "行业周期和政策限制应体现在授信期限、额度、条件和监控中.",
    }
    aliases = {
        "cashflow": "现金流",
        "cash flow": "现金流",
        "collateral": "抵质押",
        "guarantee": "保证",
        "industry": "行业",
    }
    normalized_topic = aliases.get(topic.strip().lower(), topic.strip())
    return policies.get(normalized_topic, "未找到匹配的政策说明.")


@tool
def record_review_note(application_id: str, risk_dimension: str, note: str) -> str:
    """按申请编号和风险维度记录一条信贷审批审查意见."""
    return f"{application_id}:{risk_dimension}:{note}"
