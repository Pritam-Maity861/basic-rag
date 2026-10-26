from config import QDRANT_COLLECTION, QDRANT_URL
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from providers.embedding import get_embedding_model


def create_vector_store(
        chunks:list[Document]
)->QdrantVectorStore:
    if not chunks:
        raise ValueError("Cannot index an empty list of document chunks.")
    
    embedding_model = get_embedding_model()

    vector_store=QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION,
        force_recreate=True,  # always rebuild to avoid stale/mismatched collections
    )

    return vector_store


def get_vector_store()->QdrantVectorStore:
    return QdrantVectorStore.from_existing_collection(
        embedding=get_embedding_model(),
        url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION
    )