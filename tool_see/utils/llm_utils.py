# Copyright (c) Praneeth Vadlapati

import os
from typing import Callable, List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI #, OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_community.cache import SQLiteCache
import openai

load_dotenv()


llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", ""),
    reasoning_effort="medium",
    max_retries=3,
#     api_key=lambda: api_key,
#     base_url=base_url,
    cache=SQLiteCache(database_path="llm_cache.db"),
)


class OpenAIEmbeddings(Embeddings):
    def __init__(self, api_key: Callable, base_url: Optional[str], model: str):
        self.embedding_client = openai.OpenAI(
            api_key=api_key(),
            base_url=base_url,
        )
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = self.embedding_client.embeddings.create(
            input=texts,
            model=self.model,
        )
        embeddings = [value.embedding for value in result.data]
        return embeddings

    def embed_query(self, text: str):
        return self.embed_documents([str(text)])[0]


embeddings = OpenAIEmbeddings(
    api_key=lambda: os.getenv("EMBED_API_KEY", ""),
    model=os.getenv("EMBED_MODEL", ""),
    base_url=os.getenv("EMBED_API_BASE"),
)
