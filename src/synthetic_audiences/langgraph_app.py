from __future__ import annotations

from typing import Any, TypedDict


class AgentApplicationState(TypedDict, total=False):
    agent_profile: dict[str, Any]
    question: dict[str, Any]
    artifact: dict[str, Any]
    messages: list[dict[str, str]]
    response: dict[str, Any]


def build_minimal_graph():
    """Optional LangGraph skeleton for future stateful orchestration.

    This MVP does not require LangGraph to run. Use this when you want to add
    memory/checkpointing, retries, human-in-the-loop review or parallel graph nodes.
    Install with: pip install -e '.[langgraph]'
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise ImportError("Install optional dependencies with: pip install -e '.[langgraph]' ") from exc

    from synthetic_audiences.prompts import build_prompt_agent

    def build_messages(state: AgentApplicationState) -> AgentApplicationState:
        state["messages"] = build_prompt_agent(
            state["agent_profile"],
            state["question"],
            state["artifact"],
        )
        return state

    graph = StateGraph(AgentApplicationState)
    graph.add_node("build_messages", build_messages)
    graph.add_edge(START, "build_messages")
    graph.add_edge("build_messages", END)
    return graph.compile()
