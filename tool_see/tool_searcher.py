from typing import List, Dict, Any, Optional

from langchain.tools import tool, BaseTool, ToolRuntime

from tool_see.utils.tool_utils import tool_memory, create_tool


def select_tools_for_query(
    query: str,
    top_k: int = 5,
    score_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Query `tool_memory` and return a list of metadata for matching tools."""
    results = tool_memory.query(query, top_k=top_k)
    selected: List[Dict[str, Any]] = []

    for tid, metadata, score in results:
        if score_threshold and score < score_threshold:
            continue
        m = dict(metadata or {})
        m["_tool_id"] = tid
        m["_score"] = score
        selected.append(m)

    return selected


@tool
def search_tools(query: str, runtime: ToolRuntime) -> str:
    """Search for tools based on a query."""
    writer = runtime.stream_writer

    # Stream custom updates as the tool executes
    writer(f"Looking up tools for query: {query}")
    matched_tools = select_tools_for_query(query=query, top_k=5, score_threshold=0.35)
    if not matched_tools:
        return "No matching tools found."

    fetched_tools: List[BaseTool] = []
    summaries: List[str] = []
    for m in matched_tools:
        t = create_tool(m)
        if t:
            fetched_tools.append(t)
            name = getattr(t, "name", m.get("name", "unnamed"))
            desc = m.get("description", "")
            summaries.append(f"{name}: {desc}")

    # Register the created tool objects for the driver to pick up
    if fetched_tools:
        register_tools(runtime, fetched_tools)

    summary = ", ".join(summaries)
    writer(f"Found tools: {summary}")
    return "Registered tools: " + summary
