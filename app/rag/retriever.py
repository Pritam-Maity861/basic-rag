from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from rag.vector_store import get_vector_store


def retrive_relevent_chunks(
    question: str,
    vector_store: QdrantVectorStore | None = None,
    k: int = 4,
)->list[Document]:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    if vector_store is None:
        vector_store = get_vector_store()

    documents=vector_store.similarity_search(
        query=question.strip(),
        k=k
    )

    return documents