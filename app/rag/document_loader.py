from builtins import FileNotFoundError
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(file_path: str) -> list[Document]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    loader = PyPDFLoader(str(path))
    documents = loader.load()
    documents = [doc for doc in documents if doc.page_content.strip()]

    if not documents:
        raise ValueError("No extractable text found in the PDF.")

    return documents


def load_all_pdfs(folder_path: str) -> list[Document]:
    """Load every PDF file found inside *folder_path* and return all pages."""
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Documents folder not found: {folder}")

    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in: {folder}")

    all_documents: list[Document] = []
    for pdf_file in pdf_files:
        print(f"  Loading: {pdf_file.name}")
        docs = load_pdf(str(pdf_file))
        all_documents.extend(docs)

    return all_documents