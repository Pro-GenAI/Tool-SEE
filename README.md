# ToolSEE: Agent Tool Search Engine for Reliable Tool Selection

ToolSEE helps agents find the most appropriate tools for their tasks in real time.

## Problem

Agents often get overloaded with large collections of tools and struggle to choose the right one. Multiple tools — especially junk, redundant, or harmful tools — can confuse agents and increase hallucinations, reducing their effectiveness and reliability. This problem is often compounded by insufficient context engineering — not just prompt engineering — because without well-curated context (concise tool descriptions, provenance, and example usages), agents receive noisy or incomplete signals about tool capabilities, which further degrades tool selection and increases errors.

## Solution

ToolSEE implements a dynamic tool search engine that helps agents discover and select the most appropriate tools for a given task on the fly. By ranking and filtering candidate tools based on relevance, safety, and provenance, ToolSEE reduces noise and improves decision-making for agents.

__Key points__:
- Real-time search and ranking of tools relevant to the current task and context.
- Filtering out low-quality, redundant, or potentially harmful tools.
- Integrates as a lightweight component that agents can query to guide tool selection.

## How ToolSEE Works (high level)

- Index available tools with metadata (capabilities, inputs, outputs, safety notes).
- Given an agent request and runtime context, score candidate tools for relevance and safety.
- Return a ranked list of selected tools with short context snippets and example calls.
- Optionally, the agent can request more information or verification traces before invoking a tool.

## Why ToolSEE matters

Selecting the right tool at the right moment is the difference between a reliable agent and a hallucination-prone one. ToolSEE turns chaotic tool catalogs into clear, actionable choices:
- **Fewer mistakes:** Rank by task fit and safety to cut tool misuse.
- **Faster decisions:** Real-time scoring trims exploration time to seconds.
- **Higher trust:** Surface provenance and examples so agents know why a tool is chosen.
- **Drop-in integration:** Minimal changes to your agent loop—keep your architecture intact.

## Expected Outcomes
- **Reduced hallucinations:** Cleaner context and smarter selection lowers error rates.
- **Improved latency:** Less trial-and-error tool calling speeds up responses.
- **Better reliability:** Consistent, explainable tool choice builds confidence in production.


## Quick Start

Use the scripts to experience ToolSEE end-to-end:

```bash
# From the repository root
python -m tool_see.scripts.test_flow

# Optional: measure search latency characteristics
python -m tool_see.scripts.test_latency
```

Then explore the core components:
- `tool_see/tool_searcher.py`: Scoring, ranking, and filtering.
- `tool_see/auto_tool_agent.py`: Example agent loop integration.
- `tool_see/utils/llm_utils.py`: Prompt construction helpers and LLM glue.


## Integrate in Your Agent

1. **Define tool metadata:** Capabilities, inputs/outputs, and safety notes.
2. **Call ToolSEE before execution:** Request a ranked list for the current task.
3. **Execute with confidence:** Use the top-ranked tool, with provenance and example calls.

## Roadmap

- Deeper safety signals (risk heuristics, sandboxing hooks).
- Feedback-driven re-ranking based on agent outcomes.
- Pluggable backends for tool registries and tracing.

If you’re wrestling with tool overload and unreliable selection, ToolSEE is the fastest path to calmer, more capable agents. Try the scripts above and plug it into your loop.
