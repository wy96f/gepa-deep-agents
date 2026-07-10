from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from tools import lookup_policy, tag_ticket


class DemoChatModel(FakeListChatModel):
    def bind_tools(self, tools, **kwargs):
        del tools, kwargs
        return self


SUPPORT_ROUTER_SPEC = {
    "model": DemoChatModel(responses=["<route>billing</route>"] * 20),
    "system_prompt": "You are a support router loaded from langgraph.json.",
    "tools": [tag_ticket],
    "memory": ["AGENTS.md"],
    "skills": ["skills"],
    "subagents": [
        {
            "name": "risk-reviewer",
            "description": "Use to review ambiguous routing decisions before finalizing.",
            "system_prompt": "You review routing risk. Use lookup_policy and the risk-review skill.",
            "tools": [lookup_policy],
            "skills": ["subagents/risk-reviewer/skills"],
        }
    ],
}


def support_router(config=None):
    """Graph entry referenced by langgraph.json.

    LangGraph CLI graph entries normally return a StateGraph or
    CompiledStateGraph. This demo mirrors that shape by returning the graph
    produced by create_deep_agent(...). GEPA discovers the Deep Agents inputs by
    capturing this create_deep_agent call while loading the LangGraph graph.
    """
    del config
    return create_deep_agent(**SUPPORT_ROUTER_SPEC)
