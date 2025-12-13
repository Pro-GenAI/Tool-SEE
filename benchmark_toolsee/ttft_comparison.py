
# This compares input latency of using all tools as context vs using only selected tools.

import json
import os
from pathlib import Path
import random
import statistics
import time
from typing import Dict

from dotenv import load_dotenv
from openai import OpenAI

from benchmark_toolsee.token_utils import count_tokens


load_dotenv()

client = OpenAI()

model = os.getenv("OPENAI_MODEL", "")
if not model:
    raise ValueError("OPENAI_MODEL environment variable not set")


def get_ttft_ms(tool_descriptions: str) -> int:
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"Given the following tool descriptions:\n{tool_descriptions}\n\n"
            },
            {
                "role": "user",
                "content": f"What's the weather like today in Seattle?",
            },
        ],
        max_tokens=1,
    )
    t1 = time.perf_counter()
    response = response.choices[0].message.content
    if not response or not response.strip():
        raise ValueError("Empty response from OpenAI API")
    return int((t1 - t0) * 1000)  # return time in milliseconds


# check whether all rows of big_tool_des.json are in plugin_des.json
root = Path(__file__).parent.parent
datasets = root / "eval_datasets"
with open(datasets / "plugin_des.json", "r") as f:
    all_tools: Dict[str, str] = json.load(f)


random.seed(42)
selected_keys = random.sample(list(all_tools.keys()), k=5)

selected_tools = {}
for key in selected_keys:
    selected_tools[key] = all_tools[key]


if __name__ == "__main__":

    # Calculate latency with all tools
    original_ttfts = []
    ATTEMPTS = 10

    print("Calculating TTFT with all tools...")
    for attempt in range(ATTEMPTS):
        print(f"Attempt {attempt + 1} of {ATTEMPTS}")
        original_ttft = get_ttft_ms(str(all_tools))
        original_ttfts.append(original_ttft)

    original_ttft_median = statistics.median(original_ttfts)
    print(f"TTFT with all tools: {original_ttft_median} ms")
    original_tokens = count_tokens(str(all_tools))
    print(f"Total tokens for all tools: {original_tokens}")


    # Calculate latency with only selected tools

    time.sleep(10)  # wait before making next set of requests

    print(f"Calculating TTFT with selected tools...")
    selected_ttfts = []
    for attempt in range(ATTEMPTS):
        print(f"Attempt {attempt + 1} of {ATTEMPTS}")
        selected_ttft = get_ttft_ms(str(selected_tools))
        selected_ttfts.append(selected_ttft)

    selected_ttft_median = statistics.median(selected_ttfts)
    print(f"TTFT with selected tools: {selected_ttft_median} ms")
    tokens = count_tokens(str(selected_tools))
    print(f"Total tokens for selected tools: {tokens}")



    print(f"TTFT difference: {original_ttft_median - selected_ttft_median} ms")
    reduction = (original_ttft_median - selected_ttft_median) / original_ttft_median * 100
    print(f"TTFT reduction: {reduction:.2f}%")

    print(f"Token difference: {original_tokens - tokens} tokens")
    token_saving = (original_tokens - tokens) / original_tokens * 100
    print(f"Token savings: {token_saving:.2f}%")
