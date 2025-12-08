"""
Automated tool-agent workflow for ToolSEE.
"""

import logging
from typing import List, Dict, Any

from langchain.agents import create_agent

from tool_see.tool_searcher import search_tools, select_tools_for_query
from tool_see.utils.tool_utils import tool_memory, create_tool
from tool_see.utils.llm_utils import llm


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


def sample_tools() -> List[tuple]:
    def search_files(query: str):
        """Tool to search files by query (demo stub)."""
        return f"Searching files for: {query}"

    def run_tests(_args: str):
        """Tool to run tests (demo stub)."""
        return "Running tests..."

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


def populate_tool_memory():
    tools = sample_tools()
    tool_memory.add_tools(tools)
    logger.debug("Populated tool_memory with %d tools", len(tools))


def test_selection():
    q = "run tests and report failures"
    matches = select_tools_for_query(query=q, top_k=3)
    logger.debug(f"Selection results for query: '{q}'")
    for m in matches:
        logger.debug(
            f"- id={m.get('_tool_id')} score={m.get('_score'):.4f} name={m.get('name')}"
        )
    return matches


def test_create_tool_and_run(metadata: Dict[str, Any]):
    tool = create_tool(metadata)
    if tool is None:
        logger.warning("create_tool returned None for metadata: %s", metadata)
        return None

    logger.debug("Created tool: %s", getattr(tool, "name", repr(tool)))
    # Try to run the tool with a sample input if possible
    try:
        # BaseTool typically exposes a `run` method that accepts a single string argument
        out = tool.run("example input")
        logger.debug("Tool run output: %s", out)
    except Exception as e:
        logger.warning("Tool run raised an exception: %s", e)

    return tool


def test_agent(prompt: str):
    # `search_tools` is a LangChain tool created with the @tool decorator in tool_searcher
    tools = [search_tools]

    agent = create_agent(
        llm,
        tools=tools,
        # system_prompt="You are a helpful assistant that can use tools.",
    )

    logger.debug("Running agent with tools=[search_tools] prompt:\n %s", prompt)
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": prompt},
            ]
        },
        context={"user_role": "beginner"},
    )
    logger.debug("Agent result:\n %s", result)
    if not result:
        raise Exception("Empty response from the bot")

    logger.debug("\n")
    logger.debug("\n")
    for message in result["messages"]:
        logger.debug("message %s %s", type(message), message)
        logger.debug("\n")
    logger.debug("\n")
    logger.debug("\n")

    response = result["messages"][-1].content
    logger.info("Agent response:\n %s", response)
    return response


def main():
    populate_tool_memory()

    # test create_tool on first matched metadata
    matches = test_selection()
    if matches:
        meta = matches[0]
        tool = test_create_tool_and_run(meta)
    else:
        print("No matches found for selection test; skipping create_tool test.")

    test_agent(
        "Search for tools to 'run tests' and return the names of matching tools."
    )


if __name__ == "__main__":
    main()
