from functools import lru_cache

from config import LLM_MODEL
from langchain_ollama import ChatOllama


@lru_cache(maxsize=1)
def get_llm()->ChatOllama:
    return ChatOllama(
        model=LLM_MODEL,
        temperature=0
    )