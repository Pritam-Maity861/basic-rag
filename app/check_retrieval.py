from pathlib import Path

from rag.chain import answer_question
from rag.document_loader import load_pdf
from rag.text_splitter import split_documents
from rag.vector_store import create_vector_store


def main() -> None:
    BASE_DIR = Path(__file__).resolve().parent
    pdf_path = BASE_DIR / "data" / "documents" / "my_resume.pdf"
    documents = load_pdf(str(pdf_path))
    chunks = split_documents(documents)

    vector_store = create_vector_store(chunks)

    print(f"Indexed {len(chunks)} chunks.")

    while True:
        question = input("\nAsk a question (or type 'exit'): ").strip()

        if question.lower() == "exit":
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            answer=answer_question(
                question=question,
                vector_store=vector_store,
                k=4
            )
            print(f"\nBot: {answer}")
        except (ValueError, KeyError, RuntimeError) as error:
            print(f"\nError while answering: {error}")


if __name__ == "__main__":
    main()
