from typing import List, Tuple
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from config import embedding_model, reranker_model, CHROMA_DIR

# -------------------------------
# Document Loaders
# -------------------------------
# Each function loads one file type and returns a list of LangChain Documents.


def load_pdf(file_path: str) -> List[Document]:
    """Load and return text content from a PDF file, one Document per page."""
    return PyPDFLoader(file_path).load()


def load_docx(file_path: str) -> List[Document]:
    """Load and return text content from a Word (.docx) file."""
    return Docx2txtLoader(file_path).load()


def load_csv(file_path: str) -> List[Document]:
    """Load and return rows from a CSV file as documents."""
    return CSVLoader(file_path).load()


def load_excel(file_path: str) -> List[Document]:
    """Load and return content from an Excel (.xlsx) file."""
    return UnstructuredExcelLoader(file_path).load()


# -------------------------------
# Text Splitting
# -------------------------------

def split_documents(
    documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 150
) -> List[Document]:
    """Split loaded documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


# -------------------------------
# Vector Store (Chroma) - persistent, reused across calls
# -------------------------------

def get_vectorstore(persist_directory: str = CHROMA_DIR) -> Chroma:
    """
    Open the existing persistent Chroma collection (or create an empty one).
    Never rebuilds - this is the single vector store the whole app shares.
    """
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
    )


def add_chunks_to_store(vectorstore: Chroma, chunks: List[Document]) -> None:
    """Embed and persist new chunks into the existing vector store."""
    if not chunks:
        return
    vectorstore.add_documents(chunks)


def delete_document_chunks(vectorstore: Chroma, document_id: str) -> None:
    """Delete every chunk belonging to one document_id from the vector store."""
    vectorstore._collection.delete(where={"document_id": document_id})


# -------------------------------
# BM25 Keyword Search
# -------------------------------

def build_bm25_retriever(chunks: List[Document], k: int = 5) -> BM25Retriever:
    """Create a keyword-based (BM25) retriever from document chunks."""
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k
    return retriever


# -------------------------------
# Vector Search (with similarity scores, for the debug expander)
# -------------------------------

def vector_search_with_scores(
    vectorstore: Chroma, query: str, k: int = 5
) -> List[Tuple[Document, float]]:
    """Search ChromaDB, returning (chunk, similarity_score) pairs."""
    return vectorstore.similarity_search_with_relevance_scores(query, k=k)


# -------------------------------
# Hybrid Search (Vector + BM25)
# -------------------------------

def hybrid_search(
    vectorstore: Chroma, bm25_retriever: BM25Retriever, query: str, k: int = 5
) -> List[Tuple[Document, dict]]:
    """
    Combine vector search and BM25 keyword search results.
    Returns (chunk, score_info) pairs so scores can be shown in the UI.
    Duplicate chunks (same page content) are removed, keeping the first hit.
    """
    vector_hits = vector_search_with_scores(vectorstore, query, k=k)
    keyword_hits = bm25_retriever.invoke(query)

    seen = set()
    merged: List[Tuple[Document, dict]] = []

    for doc, score in vector_hits:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            merged.append((doc, {"vector_score": round(score, 4), "bm25_score": None}))

    for doc in keyword_hits:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            merged.append((doc, {"vector_score": None, "bm25_score": "matched"}))

    return merged


# -------------------------------
# Re-ranking (cross-encoder)
# -------------------------------

def rerank_chunks(
    query: str, hits: List[Tuple[Document, dict]], top_n: int = 4
) -> List[Tuple[Document, dict]]:
    """
    Re-score hybrid search hits with a cross-encoder and keep only the
    most relevant `top_n`. Preserves each chunk's original score_info and
    adds a `rerank_score` to it.
    """
    if not hits:
        return []

    pairs = [(query, doc.page_content) for doc, _ in hits]
    rerank_scores = reranker_model.predict(pairs)

    scored = []
    for (doc, score_info), rerank_score in zip(hits, rerank_scores):
        score_info = {**score_info, "rerank_score": round(float(rerank_score), 4)}
        scored.append((doc, score_info))

    scored.sort(key=lambda pair: pair[1]["rerank_score"], reverse=True)
    return scored[:top_n]
