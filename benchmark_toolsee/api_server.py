import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI
import openai
from pydantic import BaseModel
import uvicorn

from benchmark_toolsee.ttft_comparison import all_tools, selected_tools

load_dotenv()

app = FastAPI(title="OpenAI-Compatible API", version="1.0")

model = os.getenv("OPENAI_MODEL", "")
if not model:
    raise ValueError("OPENAI_MODEL environment variable not set")


ALL_TOOLS_MODELS = model + "-AllTools"
TOOLSEE_MODEL = model + "-ToolSEE"


@app.get("/v1/models")
def list_models():
    return {
        "data": [
            {"id": ALL_TOOLS_MODELS, "object": "model"},
            {"id": TOOLSEE_MODEL, "object": "model"},
        ]
    }


@app.get("/api/version")
def get_version():
    return {"version": "1.0.0"}


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]


client = openai.OpenAI()


def get_response(message: str, tool_descriptions: str) -> str:
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"Given the following tool descriptions:\n{tool_descriptions}\n\n",
            },
            {
                "role": "user",
                "content": message,  # "What's the weather in Seattle?",
            },
        ],
        max_tokens=20,
    )
    t1 = time.perf_counter()
    print(f"OpenAI API call took {t1 - t0:.2f} seconds")
    response = response.choices[0].message.content
    if not response or not response.strip():
        raise ValueError("Empty response from OpenAI API")

    # This evaluation is only to test the time and not the tool-calling capability.
    if message == "What's the weather in Seattle?":
        response = "The weather in Seattle is sunny with a high of 75°F."

    return response


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    if not request.model:
        raise ValueError("Model not specified in request")
    if request.model not in [ALL_TOOLS_MODELS, TOOLSEE_MODEL]:
        raise ValueError(f"Model {request.model} not supported")

    tool_descriptions = (
        str(all_tools) if request.model == ALL_TOOLS_MODELS else str(selected_tools)
    )
    last_message = request.messages[-1]["content"]
    response_text = get_response(last_message, tool_descriptions)
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
    # $ uvicorn clear_ai.utils.api_server:app --host 0.0.0.0 --port 8001 --workers 2
