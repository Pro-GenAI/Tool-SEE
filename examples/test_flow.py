"""
Automated tool-agent workflow for ToolSEE.
"""

import logging
import statistics
import time
from typing import List, Dict, Any, Tuple

from tool_see.utils.tool_utils import ToolMemory, create_tool
from tool_see.auto_tool_agent import run_agent
from tool_see.tool_searcher import select_tools_for_query


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    # Add a module-level handler so we only show logs emitted from this module
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Ensure the module logger emits INFO and does not propagate to the root logger
logger.setLevel(logging.INFO)
logger.propagate = False

tool_memory = ToolMemory()


def sample_tools() -> List[Tuple[str, Dict[str, Any]]]:
    def search_files(query: str):
        """Tool to search files by query (demo stub)."""
        logger.info(f"➡️ Searching files for query: {query}")
        return f"Searching files for: {query}"

    def run_tests():
        """Tool to run tests (demo stub)."""
        logger.info("➡️ Running tests...")
        return "Tests run successfully."

    return [
        (
            "file_search",
            {
                "name": "file_search",
                "description": "Search project files by pattern.",
                "function": search_files,
            },
        ),
        (
            "test_runner",
            {
                "name": "test_runner",
                "description": "Run unit tests.",
                "function": run_tests,
            },
        ),
    ]


def populate_tool_memory(tools = None):
    if tools is None:
        tools = sample_tools()
    t0 = time.perf_counter()
    tool_memory.add_tools(tools)
    t1 = time.perf_counter()
    logger.info("Populated tool_memory with %d tools in %.4f seconds.", len(tools), t1 - t0)


def test_selection():
    q = "run tests and report failures"
    t0 = time.perf_counter()
    matches = select_tools_for_query(query=q, top_k=3, tool_memory=tool_memory)
    t1 = time.perf_counter()
    logger.info(f"Fetched {len(matches)} results in {t1 - t0:.4f} seconds.")
    logger.info(f"Top {len(matches)} matches:")
    for m in matches:
        logger.info(
            f"- id={m.get('_tool_id')} score={m.get('_score'):.4f} name={m.get('name')}"
        )
    return matches


def test_selection_latency():
    queries = [
        "run tests",
        "search files",
        "benchmark tool",
        "file search",
        "test runner",
    ]

    q_runs = 50
    query_times = []
    total_results = 0
    for i in range(q_runs):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        matches = select_tools_for_query(q, top_k=5, tool_memory=tool_memory)
        t1 = time.perf_counter()
        query_times.append(t1 - t0)
        total_results += len(matches)

    logger.info(
        "Selection latency stats (ms): mean=%.3f median=%.3f min=%.3f max=%.3f",
        statistics.mean(query_times) * 1000,
        statistics.median(query_times) * 1000,
        min(query_times) * 1000,
        max(query_times) * 1000,
    )
    logger.info("Total results fetched: %d", total_results)

def test_ingestion_latency():
    N = 100
    tools = sample_tools() * (N // 2)  # duplicate to reach N tools

    # Measure insertion (add_tools) latency
    runs = 3
    insert_times = []
    for r in range(runs):
        # ensure empty store for each run
        tool_memory._store = {}
        t0 = time.perf_counter()
        tool_memory.add_tools(tools)
        t1 = time.perf_counter()
        dt = t1 - t0
        insert_times.append(dt)
        logger.info("Run %d: inserted %d tools in %.4fs", r + 1, N, dt)

    logger.info(
        "Insertion stats (ms): mean=%.4f median=%.4f min=%.4f max=%.4f",
        statistics.mean(insert_times) * 1000,
        statistics.median(insert_times) * 1000,
        min(insert_times) * 1000,
        max(insert_times) * 1000,
    )
    logger.info(
        "Insertion time per tool (ms): mean=%.4f median=%.4f min=%.4f max=%.4f",
        (statistics.mean(insert_times) / N) * 1000,
        (statistics.median(insert_times) / N) * 1000,
        (min(insert_times) / N) * 1000,
        (max(insert_times) / N) * 1000,
    )


def test_create_tool_and_run(metadata: Dict[str, Any]):
    tool = create_tool(metadata)
    if tool is None:
        logger.warning("create_tool returned None for metadata: %s", metadata)
        return None

    logger.info("Created tool: %s", getattr(tool, "name", repr(tool)))
    # Try to run the tool with a sample input if possible
    try:
        # BaseTool typically exposes a `run` method that accepts a single string argument
        out = tool.run("example input")
        logger.info("Tool run output: %s", out)
    except Exception as e:
        logger.warning("Tool run raised an exception: %s", e)

    return tool


def test_agent():
    # prompt = "Search for a tool to run test cases."
    prompt = "Run tests."
    logger.info("Testing agent with prompt: %s", prompt)
    result = run_agent(prompt, tool_memory=tool_memory)
    logger.info("Agent returned result: %s", result)
    return result


def main():
    populate_tool_memory()

    # test create_tool on first matched metadata
    matches = test_selection()
    if matches:
        meta = matches[0]
        tool = test_create_tool_and_run(meta)
        if not tool:
            logger.warning("create_tool test failed.")
    else:
        logger.warning("No matches found for selection test; skipping create_tool test.")

    test_ingestion_latency()
    test_selection_latency()

    test_agent()


if __name__ == "__main__":
    main()
