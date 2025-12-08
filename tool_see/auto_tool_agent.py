"""
Automated tool-agent workflow for ToolSEE.
"""

import logging

from langchain.agents import create_agent

from tool_see.tool_searcher import search_tools
from tool_see.utils.llm_utils import llm


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def run_agent(prompt: str):
    # `search_tools` is a LangChain tool created with the @tool decorator in tool_searcher
    tools = [search_tools]

    agent = create_agent(
        llm,
        tools=tools,
        system_prompt="You are a helpful assistant that can use tools to solve the user query.",
    )

    logger.debug("Running agent with tools=[search_tools] prompt:\n %s", prompt)
    result = agent.invoke(
        {"messages": [
            {"role": "user", "content": prompt},
        ]},
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
    from tool_see.scripts.test_flow import populate_tool_memory
    populate_tool_memory()

    # Try to run a full agent flow if credentials are available
    logger.debug("--- ToolSEE auto_tool_agent demo ---")
    run_agent("Run tests.")


if __name__ == "__main__":
    main()
