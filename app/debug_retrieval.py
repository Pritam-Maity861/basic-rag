"""
Debug script to diagnose RAG retrieval issues.
Run: uv run app/debug_retrieval.py
"""
import sys
import io
from pathlib import Path

from rag.document_loader import load_pdf
from rag.text_splitter import split_documents
from rag.vector_store import create_vector_store

# Force UTF-8 output so Unicode characters don't crash on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def safe_print(text: str, max_len: int = 200) -> str:
    """Trim and sanitize text for safe terminal printing."""
    return text[:max_len].strip().replace("\n", " \\n ")


def main() -> None:
    BASE_DIR = Path(__file__).resolve().parent
    pdf_path = BASE_DIR / "data" / "documents" / "my_resume.pdf"

    print("=== Step 1: Loading PDF ===")
    documents = load_pdf(str(pdf_path))
    print(f"  Pages loaded: {len(documents)}")
    for i, doc in enumerate(documents):
        print(f"  Page {i+1} preview: {safe_print(doc.page_content, 150)!r}")

    print("\n=== Step 2: Splitting into chunks ===")
    chunks = split_documents(documents, chunk_size=500, chunk_overlap=100)
    print(f"  Chunks created: {len(chunks)}")

    # Print all chunk previews so you can see what text is indexed
    print("\n=== Step 3: ALL Chunk contents ===")
    for i, chunk in enumerate(chunks):
        page = chunk.metadata.get("page", "?")
        print(f"\n  [Chunk {i+1} | Page {page}]")
        print(f"  {safe_print(chunk.page_content, 300)!r}")

    print("\n=== Step 4: Creating vector store (force fresh collection) ===")
    vector_store = create_vector_store(chunks)
    print("  Vector store created successfully.")

    print("\n=== Step 5: Test queries ===")
    test_queries = [
        "tech stack",
        "technologies used in projects",
        "programming languages",
        "frameworks",
        "skills",
    ]

    for query in test_queries:
        print(f"\n  Query: {query!r}")
        results = vector_store.similarity_search(query, k=3)
        if not results:
            print("  WARNING: No results returned!")
        for j, doc in enumerate(results):
            page = doc.metadata.get("page", "?")
            print(f"    Result {j+1} (page {page}): {safe_print(doc.page_content, 200)!r}")


if __name__ == "__main__":
    main()
