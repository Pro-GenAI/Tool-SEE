from typing import Callable, Dict, List, Optional, Any, Tuple
import json
import math

from langchain.tools import tool, BaseTool

from tool_see.utils.llm_utils import embeddings


class ToolMemory:
    """In-memory storage for tool embeddings and full metadata.
    This avoids using a vector DB by keeping embeddings and metadata in a Python dict.
    Persist/restore is supported via JSON file (embeddings are stored as lists).
    """

    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = persist_path
        # internal store: tool_id -> {"metadata": {...}, "embedding": [...float]}
        self._store: Dict[str, Dict[str, Any]] = {}
        if persist_path:
            try:
                self.load(persist_path)
            except Exception:
                print(f"ToolMemory: could not load from {persist_path}, starting fresh.")

    def add_tools(
        self,
        tools: List[Tuple[str, Dict[str, Any]]],
        text_keys: Optional[List[str]] = None,
    ):
        texts = []
        ids = []
        for tool_id, metadata in tools:
            ids.append(tool_id)
            if text_keys is None:
                keys = ["name", "description"]
            else:
                keys = text_keys
            parts = [str(metadata.get(k)) for k in keys if metadata.get(k)]
            text = "Tool: " + " \n".join(parts) if parts else json.dumps(metadata)
            texts.append(text)

        try:
            embs = embeddings.embed_documents(texts)
        except Exception as e:
            print("ToolMemory.add_tools: embed_documents failed, trying embed_query:", e)
            raise e

        for tool_id, metadata, emb in zip(ids, [m for _, m in tools], embs):
            self._store[tool_id] = {"metadata": metadata, "embedding": list(emb)}

        if self.persist_path:
            self.save()

    def _cosine(self, a: List[float], b: List[float]) -> float:
        # Returns cosine similarity between vectors a and b
        # handle zero vectors defensively
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na == 0 or nb == 0:
            return 0.0
        return dot / (math.sqrt(na) * math.sqrt(nb))

    def query(
        self, query_text: str, top_k: int = 3
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """Query the memory and return top_k tools as (tool_id, metadata, score).
        Score is cosine similarity in [0,1].
        """
        query_embed = embeddings.embed_query(query_text)

        scores = []
        for tid, entry in self._store.items():
            score = self._cosine(query_embed, entry["embedding"])
            scores.append((tid, entry["metadata"], float(score)))

        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_k]

    def get_all_tools(self) -> Dict[str, Dict[str, Any]]:
        return self._store

    def save(self, path: Optional[str] = None):
        p = path or self.persist_path
        if not p:
            raise ValueError("persist_path not set")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self._store, f, indent=2)

    def load(self, path: Optional[str] = None):
        p = path or self.persist_path
        if not p:
            raise ValueError("persist_path not set")
        with open(p, encoding="utf-8") as f:
            self._store = json.load(f)


tool_memory = ToolMemory()


def create_tool(metadata: Dict[str, Any]) -> Optional[BaseTool]:
    """Convert metadata into a LangChain tool function using @tool wrapper.

    Expected metadata fields:
      - name: tool name (fallback _tool_id)
      - description: human-readable docstring
      - function or callable: Python callable to invoke

    Returns:
      A decorated tool function or None if metadata invalid.
    """

    # Determine function to wrap
    func = metadata.get("function")
    if not callable(func):
        return None

    # Determine name + description for the tool schema
    name = metadata.get("name", "")
    description = metadata.get("description", "")

    # Wrap the function with the tool decorator
    # Using dynamic decoration preserves signature
    decorated = tool(name_or_callable=name, description=description)(
        _ensure_doc(func, description)
    )

    return decorated


def _ensure_doc(func: Callable, desc: str) -> Callable:
    """Ensure the function has a proper docstring since LangChain uses it as description."""
    if not func.__doc__ or func.__doc__.strip() == "":
        func.__doc__ = desc
    return func
