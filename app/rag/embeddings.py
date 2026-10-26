from langchain_core.documents import Document
from providers.embedding import get_embedding_model


def generate_embeddings(
    chunks: list[Document],
) -> list[list[float]]:
    if not chunks:
        raise ValueError("Cannot generate embeddings for empty chunks.")

    embedding_model = get_embedding_model()

    texts = [chunk.page_content for chunk in chunks]

    vectors = embedding_model.embed_documents(texts)

    if len(vectors) != len(chunks):
        raise RuntimeError(
            "The number of embeddings does not match the number of chunks."
        )

    return vectors


def generate_query_embedding(question: str) -> list[float]:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    embedding_model = get_embedding_model()

    return embedding_model.embed_query(question.strip())
