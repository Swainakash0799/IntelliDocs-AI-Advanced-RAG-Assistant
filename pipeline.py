import time
from typing import List, Dict, Any, Optional

from tools import build_bm25_retriever, hybrid_search, rerank_chunks, get_vectorstore
from ingestion import ingest_files
from agents import retriever_agent, answer_agent
from config import TOP_K, RERANK_TOP_K, get_logger

logger = get_logger(__name__)


# -------------------------------
# Ingestion entrypoint
# -------------------------------

def run_ingestion(file_paths: List[str]) -> Dict[str, Any]:
    """
    Add uploaded files to the persistent knowledge base. Safe to call
    repeatedly - files already ingested (matched by content hash) are
    skipped rather than re-embedded.
    """
    logger.info("Ingesting %d file(s)", len(file_paths))
    return ingest_files(file_paths)


# -------------------------------
# Query entrypoint
# -------------------------------

def _format_chat_history(chat_history: Optional[List[Dict[str, str]]], max_turns: int = 3) -> str:
    """Render the last few turns as plain text for prompt context."""
    if not chat_history:
        return "(none)"
    recent = chat_history[-max_turns:]
    lines = [f"Q: {turn['question']}\nA: {turn['answer']}" for turn in recent]
    return "\n\n".join(lines)


def _tag_source(doc) -> str:
    """Build a '[source: filename, page X]' tag from a chunk's metadata."""
    filename = doc.metadata.get("filename", "unknown")
    page = doc.metadata.get("page", "-")
    return f"[source: {filename}, page {page}]"


def run_query(question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Answer a question against the existing knowledge base:
    rewrite query -> hybrid retrieve -> rerank -> answer, with citations.
    Never re-embeds documents - the vector store is reused as-is.
    """
    state: Dict[str, Any] = {}
    history_text = _format_chat_history(chat_history)

    # Step 1 - Rewrite the question into a standalone search query
    t0 = time.time()
    search_query = retriever_agent.invoke({"question": question, "chat_history": history_text})
    state["search_query"] = search_query

    # Step 2 - Hybrid retrieval (vector + BM25) against the persistent store
    vectorstore = get_vectorstore()
    # BM25 needs the corpus in memory; pull it from the same persisted chunks
    all_chunks = vectorstore._collection.get(include=["documents", "metadatas"])
    from langchain_core.documents import Document
    corpus = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(all_chunks["documents"], all_chunks["metadatas"])
    ]
    bm25_retriever = build_bm25_retriever(corpus, k=TOP_K) if corpus else None

    hits = hybrid_search(vectorstore, bm25_retriever, search_query, k=TOP_K) if bm25_retriever else []
    retrieval_latency = round(time.time() - t0, 3)

    # Step 3 - Re-rank and keep only the most relevant chunks
    t1 = time.time()
    reranked_hits = rerank_chunks(search_query, hits, top_n=RERANK_TOP_K)
    rerank_latency = round(time.time() - t1, 3)

    context = "\n\n".join(f"{_tag_source(doc)}\n{doc.page_content}" for doc, _ in reranked_hits)
    citations = [
        {"filename": doc.metadata.get("filename", "unknown"), "page": doc.metadata.get("page", "-")}
        for doc, _ in reranked_hits
    ]
    state["context"] = context
    state["citations"] = citations
    state["debug"] = [{"source": _tag_source(doc), **scores} for doc, scores in reranked_hits]

    # Step 4 - Generate the final answer
    t2 = time.time()
    final_answer = answer_agent.invoke({
        "context": context if context else "(no relevant documents found)",
        "question": question,
        "chat_history": history_text,
    })
    generation_latency = round(time.time() - t2, 3)
    state["answer"] = final_answer

    state["latencies"] = {
        "retrieval_seconds": retrieval_latency,
        "rerank_seconds": rerank_latency,
        "generation_seconds": generation_latency,
    }

    logger.info(
        "query=%r rewritten=%r retrieval=%.3fs rerank=%.3fs generation=%.3fs",
        question, search_query, retrieval_latency, rerank_latency, generation_latency,
    )

    return state


if __name__ == "__main__":
    paths = input("Enter file paths separated by commas: ").split(",")
    paths = [p.strip() for p in paths if p.strip()]
    if paths:
        print(run_ingestion(paths))

    user_question = input("Enter your question: ")
    result = run_query(user_question)
    print("\nFinal Answer:\n", result["answer"])
    print("\nCitations:", result["citations"])
