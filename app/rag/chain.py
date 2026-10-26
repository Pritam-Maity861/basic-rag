from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from providers.llm import get_llm
from rag.retriever import retrive_relevent_chunks
from pathlib import Path


PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions based on content from PDF documents.
Below are relevant excerpts retrieved from the documents. Each chunk is labeled with its source file.
Rules:
1. Read ALL context chunks carefully.
2. Pay attention to the "Source file" label — do NOT mix facts from different source files unless the question explicitly asks to compare them.
3. Only use what is present in the document context. Do NOT add external knowledge.
4. If the context does not contain relevant information, say: "I could not find this information in the document."
5. Be specific: list technologies, skills, projects, or facts as they appear in the context.
Document context (retrieved excerpts):
{context}
User question:
{question}
Answer:
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


def format_documents(documents: list[Document]) -> str:
    formatted_chunks = []

    for document in documents:
        page = document.metadata.get("page")
        source = document.metadata.get("source","unknown")
        page_label = page + 1 if isinstance(page, int) else "unknown"
        source_name = Path(source).name if source != "unknown" else "unknown"

        formatted_chunks.append(
            f"Source file: {source_name} | Page: {page_label}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n---\n\n".join(formatted_chunks)


def answer_question(
    question: str,
    vector_store: QdrantVectorStore | None = None,
    k: int = 6,
) -> str:
    """Retrieve relevant PDF chunks and generate an answer."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    documents = retrive_relevent_chunks(
        question=question,
        vector_store=vector_store,
        k=k,
    )

    if not documents:
        return "I could not find relevant information in the document."

    context = format_documents(documents)

    chain = prompt | get_llm() | StrOutputParser()

    answer = chain.invoke(
        {
            "context": context,
            "question": question.strip(),
        }
    )

    return answer.strip()
