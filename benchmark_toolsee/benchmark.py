from pathlib import Path

import csv
import json
import time
import requests
import logging
import statistics
import random

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ToolCorrectnessMetric

from tool_see import ToolMemory, select_tools_for_query
from benchmark_toolsee.token_utils import count_tokens, count_tokens_for_tool_list


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


root = Path(__file__).parent.parent.parent
datasets = root / "eval_datasets"
datasets.mkdir(exist_ok=True)


def download_file(url: str):
    filename = url.split("/")[-1]
    dest_path = datasets / filename
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return
    logger.debug(f"Downloading {url} to {dest_path}...")
    response = requests.get(url)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(response.content)
    dest_path.chmod(0o444)  # Make file read-only


download_file(
    "https://raw.githubusercontent.com/HowieHwong/MetaTool/refs/heads/master/dataset/plugin_des.json"
)
download_file(
    "https://raw.githubusercontent.com/HowieHwong/MetaTool/refs/heads/master/dataset/data/all_clean_data.csv"
)
download_file(
    "https://raw.githubusercontent.com/HowieHwong/MetaTool/refs/heads/master/dataset/data/multi_tool_query_golden.json"
)


logger.info("============ Ingesting tools into ToolMemory... ============")

# check whether all rows of big_tool_des.json are in plugin_des.json
with open(datasets / "plugin_des.json", "r") as f:
    all_tools = json.load(f)

# Ingest tools into tool memory
tool_memory = ToolMemory()


def call_tool(tool_name: str, **kwargs):
    logger.info(f"Calling tool {tool_name} with args {kwargs}")
    return f"Called {tool_name} with input {kwargs}"


tool_list = []
available_tool_names = []
for tool_name, tool_description in all_tools.items():
    tool_list.append(
        (
            tool_name,
            {
                "name": tool_name,
                "description": tool_description,
                "function": lambda **kwargs: call_tool(tool_name, **kwargs),
            },
        )
    )
    available_tool_names.append(tool_name)

t0 = time.perf_counter()
tool_memory.add_tools(tool_list)
t1 = time.perf_counter()
time_ms = (t1 - t0) * 1000
logger.info(f"Inserted {len(tool_list)} tools into tool memory in {time_ms:.2f} ms")
avg_time_per_tool = time_ms / len(tool_list)
logger.info(f"Average insertion time per tool: {avg_time_per_tool:.2f} ms")

ALL_TOOLS_TOKENS = count_tokens_for_tool_list(tool_list)
logger.info(f"Total tokens for all tools: {ALL_TOOLS_TOKENS}")


# Evaluation setup

def evaluate_cases(test_cases: list[LLMTestCase]):
    scores = []
    metric = ToolCorrectnessMetric()
    result = evaluate(test_cases=test_cases, metrics=[metric])
    for case_result in result.test_results:
        scores.append(case_result.success)
    score = statistics.mean(scores) if scores else 0.0
    return score

def process_dataset(dataset: list[dict]):
    query_times = []
    test_cases = []
    tokens_used = []
    for item in dataset:
        query = item["query"]
        expected_tools = item["tool"]
        t0 = time.perf_counter()
        selected_tools = select_tools_for_query(
            query=query, top_k=len(expected_tools) + 5,
            tool_memory=tool_memory,
        )
        t1 = time.perf_counter()
        tokens_used.append(count_tokens(selected_tools))
        logger.debug(f"selected_tools: {selected_tools}")
        selected_tool_names = [tool["_tool_id"] for tool in selected_tools]
        logger.debug(f"selected_tool_names: {selected_tool_names}")
        query_times.append(t1 - t0)
        test_case = LLMTestCase(
            input=query,
            # actual_output="N/A",  # Not evaluated
            tools_called=[ToolCall(name=tid, input_parameters={}) for tid in selected_tool_names],
            expected_tools=[ToolCall(name=tid, input_parameters={}) for tid in expected_tools],
        )
        test_cases.append(test_case)

    score = evaluate_cases(test_cases)
    tokens_saved = [ALL_TOOLS_TOKENS - used for used in tokens_used]
    tokens_saved_ratio = [saved / ALL_TOOLS_TOKENS for saved in tokens_saved]

    logger.info("Tool selection accuracy: %.4f", score)
    logger.info("Latency: median=%.2f ms", statistics.median(query_times) * 1000)
    logger.info("Token savings: median=%.2f", statistics.median(tokens_saved_ratio))


logger.info("============ Running multi-tool selection benchmark... ============")

with open(datasets / "multi_tool_query_golden.json") as f:
    multi_tool_data = json.load(f)

if __name__ == "__main__":
    process_dataset(multi_tool_data)


logger.info("============ Running single-tool selection benchmark... ============")

# Load the data in same format as multi_tool_data
single_tool_data = []

with open(datasets / "all_clean_data.csv", "r", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader, None)  # skip header row
    query_idx, tool_idx = 0, 1
    if header:
        lowered = [str(col).strip().lower() for col in header]
        if "query" in lowered:
            query_idx = lowered.index("query")
        if "tool" in lowered:
            tool_idx = lowered.index("tool")

    for row in reader:
        if not row or max(query_idx, tool_idx) >= len(row):
            continue
        query = row[query_idx]
        tool = row[tool_idx]
        if not query or not tool:
            continue
        single_tool_data.append({"query": query, "tool": [tool]})

logger.info(f"Total single-tool data points: {len(single_tool_data)}")
random.seed(42)
single_tool_data = random.sample(single_tool_data, 500)
logger.info(f"Using random sample of 500 data points for single-tool benchmark.")

if __name__ == "__main__":
    process_dataset(single_tool_data)
