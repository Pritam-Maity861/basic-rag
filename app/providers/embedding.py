from functools import lru_cache

from config import EMBEDDING_MODEL
from langchain_ollama import OllamaEmbeddings


@lru_cache(maxsize=1)
def get_embedding_model()->OllamaEmbeddings:
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url="http://localhost:11434"
    )