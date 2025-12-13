from typing import List, Dict, Any, Optional

from tool_see.utils.tool_utils import ToolMemory


def select_tools_for_query(
    query: str,
    tool_memory: ToolMemory,
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

