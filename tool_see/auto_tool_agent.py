"""
Automated tool-agent workflow for ToolSEE.
"""

import logging
from typing import Any, Callable, List
from typing_extensions import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.tools import tool, BaseTool, ToolRuntime
from langgraph.prebuilt.tool_node import (
    _get_runtime_arg,
    _get_state_args,
    _get_store_arg,
)

from tool_see.tool_searcher import select_tools_for_query
from tool_see.utils.llm_utils import llm
from tool_see.utils.tool_utils import ToolMemory, create_tool


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    # Add a module-level handler so we only show logs emitted from this module
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Ensure the module logger emits INFO and does not propagate to the root logger
logger.setLevel(logging.INFO)
logger.propagate = False


class AgentContext(TypedDict, total=False):
    tool_node: Any
    tool_memory: ToolMemory


class RuntimeToolExpansionMiddleware(AgentMiddleware[Any, AgentContext | None]):
    """Ensures the model sees newly registered tools during the same agent run.

    `search_tools` mutates the in-graph ToolNode (via `runtime.context['tool_node']`).
    This middleware refreshes the model's tool list before each model call so the LLM
    can actually invoke the newly registered tools.
    """

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> Any:
        ctx = getattr(request.runtime, "context", None)
        tool_node = ctx.get("tool_node") if isinstance(ctx, dict) else None
        if tool_node is None:
            return handler(request)

        # Keep provider-side (dict) tools, but refresh client-side BaseTool list
        # from the current ToolNode registry.
        provider_tools = [t for t in request.tools if isinstance(t, dict)]
        refreshed_tools = list(getattr(tool_node, "tools_by_name", {}).values())

        return handler(request.override(tools=refreshed_tools + provider_tools))


def register_tools(runtime: ToolRuntime, tools: List[BaseTool]) -> None:
    """Register dynamically created tools into the currently running agent.

    This function expects the agent's ToolNode instance to be provided in
    `runtime.context['tool_node']` (see `run_agent`). When available, it mutates the
    ToolNode's internal registries so newly registered tools can be executed.
    """

    ctx = getattr(runtime, "context", None)
    if not isinstance(ctx, dict):
        return

    tool_node = ctx.get("tool_node")
    if tool_node is None:
        # Best-effort: keep metadata for debugging even if we can't attach.
        pending = ctx.setdefault("pending_tools", [])
        pending.extend(tools)
        return

    # ToolNode keeps multiple internal maps; update all of them for consistent
    # injection behavior.
    tools_by_name = getattr(tool_node, "_tools_by_name", None)
    if not isinstance(tools_by_name, dict):
        raise TypeError(
            "runtime.context['tool_node'] does not look like a ToolNode instance"
        )

    for t in tools:
        tools_by_name[t.name] = t
        tool_node._tool_to_state_args[t.name] = _get_state_args(t)
        tool_node._tool_to_store_arg[t.name] = _get_store_arg(t)
        tool_node._tool_to_runtime_arg[t.name] = _get_runtime_arg(t)

    ctx.setdefault("registered_tool_names", set()).update([t.name for t in tools])


@tool
def search_tools(query: str, runtime: ToolRuntime) -> str:
    """Search for tools based on a query."""

    ctx = getattr(runtime, "context", None)
    if not isinstance(ctx, dict):
        return ""
    tool_memory = ctx.get("tool_memory", None)
    if tool_memory is None:
        return "No tool memory available."

    # Stream custom updates as the tool executes
    # runtime.stream_writer(f"Looking up tools for query: {query}")
    matched_tools = select_tools_for_query(
        query=query,
        top_k=5,
        score_threshold=0.35,
        tool_memory=tool_memory,
    )
    if not matched_tools:
        return "No matching tools found."

    fetched_tools: List[BaseTool] = []
    summaries: List[str] = []
    for matched_tool_data in matched_tools:
        tool = create_tool(matched_tool_data)
        if tool:
            logger.debug("search_tools: created tool: %s", tool)
            fetched_tools.append(tool)
            name = getattr(tool, "name", matched_tool_data.get("name", "unnamed"))
            desc = matched_tool_data.get("description", "")
            summaries.append(f"{name}: {desc}")

    # Register the created tool objects for the driver to pick up
    if fetched_tools:
        register_tools(runtime, fetched_tools)

    summary = ", ".join(summaries)
    # runtime.stream_writer(f"Found tools: {summary}")
    return "Registered tools: " + summary


def run_agent(prompt: str, tool_memory: ToolMemory) -> str:
    # `search_tools` is a LangChain tool created with the @tool decorator in tool_searcher
    tools = [search_tools]  # Default tools

    fetched_tools_data = select_tools_for_query(prompt, tool_memory=tool_memory)
    for tool_data in fetched_tools_data:
        tool_obj = create_tool(tool_data)
        if tool_obj:
            tools.append(tool_obj)

    agent = create_agent(
        llm,
        tools=tools,
        middleware=[RuntimeToolExpansionMiddleware()],
        context_schema=AgentContext,
        system_prompt=(
            "Be an agent that can use tools to solve the user query. "
            "Use `search_tools` tool to search for more tools if needed. "
            "Don't reject user query without trying to find tools first. "
        ),
    )

    # Provide the in-graph ToolNode so `search_tools` can register tools during runtime.
    tool_node = None
    try:
        tool_node = agent.nodes.get("tools").bound  # type: ignore[assignment]
    except Exception:
        tool_node = None

    logger.info("Running agent with tools=[search_tools] prompt:\n %s", prompt)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        context={"tool_node": tool_node, "tool_memory": tool_memory},
    )
    logger.debug("Agent result:\n %s", result)
    if not result:
        raise Exception("Empty response from the bot")

    # logger.debug("\n")
    # logger.debug("\n")
    # for message in result["messages"]:
    #     logger.debug("message %s %s", type(message), message)
    #     logger.debug("\n")
    # logger.debug("\n")
    # logger.debug("\n")

    response = result["messages"][-1].content
    # logger.info("Agent response:\n %s", response)
    return response
