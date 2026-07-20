from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

FINANCIAL_SNAPSHOT_PATH = Path(__file__).with_name("resources") / "company_financial_snapshots.json"


@tool
def lookup_policy(topic: str) -> str:
    """仅按主题返回内部通用审查政策, 不查询具体企业事实; topic 可为现金流、抵质押、保证或行业."""
    policies = {
        "现金流": "偿债能力必须以经过验证的经营现金流和债务覆盖能力为基础.",
        "抵质押": "抵质押用于降低违约损失, 不能替代借款人的偿债能力.",
        "保证": "保证的缓释价值取决于法律可执行性、保证人偿付能力及其独立性.",
        "行业": "行业周期和政策限制可能通过需求、价格、毛利、库存和回款影响经营现金流与偿债能力.",
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
def lookup_financial_snapshot(company_name: str) -> str:
    """查询演示数据源中的企业财务快照; 覆盖利润、现金流、负债借款、票据和用信, 不提供其他外部信息."""
    snapshots = json.loads(FINANCIAL_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    snapshot = snapshots.get(company_name.strip())
    if snapshot is None:
        return f"ERROR: 演示财务数据源中未找到企业记录: {company_name}"
    return json.dumps(snapshot, ensure_ascii=False)


@tool
def record_review_note(application_id: str, risk_dimension: str, note: str) -> str:
    """仅保存调用方已取得且有事实依据的项目风险点, 不查询外部数据."""
    return f"{application_id}:{risk_dimension}:{note}"
