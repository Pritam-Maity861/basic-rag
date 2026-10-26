from pathlib import Path

from rag.chain import answer_question
from rag.document_loader import load_all_pdfs
from rag.text_splitter import split_documents
from rag.vector_store import create_vector_store

DOCUMENTS_DIR = Path(__file__).resolve().parent / "data" / "documents"

BANNER = """
RAG Chatbot CLI 
"""

MENU = """
  Please choose an option:
    1. Ingest documents                    
    2. Chat with bot                       
    3. Exit 
"""


def ingest_documents() -> object:
    """Load all PDFs from the documents folder, chunk, embed, and store them."""
    print(f"\n[Ingestion] Scanning: {DOCUMENTS_DIR}")
    try:
        documents = load_all_pdfs(str(DOCUMENTS_DIR))
    except (FileNotFoundError, ValueError) as e:
        print(f"[Error] {e}")
        return None

    chunks = split_documents(documents, chunk_size=500, chunk_overlap=100)
    print(f"[Ingestion] Loaded {len(documents)} page(s) → {len(chunks)} chunk(s).")
    print("[Ingestion] Building vector store … (this may take a moment)")

    vector_store = create_vector_store(chunks)
    print("[Ingestion] Documents ingested successfully!\n")
    return vector_store


def chat_loop(vector_store) -> None:
    """Interactive Q&A loop using the provided vector store."""
    print("\n[Chat] RAG Chatbot is ready!  Type 'back' to return to the menu.\n")
    while True:
        question = input("You: ").strip()

        if question.lower() == "back":
            break

        if not question:
            continue

        answer = answer_question(
            question=question,
            vector_store=vector_store,
            k=6,
        )
        print(f"\nBot: {answer}\n")


def main() -> None:
    print(BANNER)
    vector_store = None  # will be set after ingestion

    while True:
        print(MENU)
        try:
            choice = input("Enter your choice (1/2/3): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye!")
            break

        if choice == "1":
            vector_store = ingest_documents()

        elif choice == "2":
            if vector_store is None:
                print(
                    "\n[Warning] No documents ingested yet. "
                    "Please choose option 1 first.\n"
                )
            else:
                chat_loop(vector_store)

        elif choice == "3":
            print("\nBot: Goodbye! \n")
            break

        else:
            print("\n[Warning] Invalid choice. Please enter 1, 2, or 3.\n")


if __name__ == "__main__":
    main()
