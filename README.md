# EnterpriseRAG

A hybrid (vector + BM25) RAG assistant with a **persistent knowledge base** —
documents are embedded once, reused for every future question.

## Structure

```
config.py       LLM, embedding model, reranker, paths, logging - all in one place
tools.py        Loaders, splitter, vector store, BM25, hybrid search, reranking
ingestion.py    Hashing/dedup, UUID + metadata, add/delete documents (manifest.json)
prompts.py      Retriever + answer prompt templates (memory-aware)
agents.py       Two LCEL chains: query rewriter, answer generator
pipeline.py     Two entrypoints: run_ingestion() and run_query()
app.py          Streamlit UI - Knowledge Base tab + Ask (chat) tab
```

No `data/` folder is required — everything the knowledge base needs lives
inside `chroma_db/` (vectors + `manifest.json`). `uploads/` is just a scratch
folder used while a file is being ingested.

## What changed from the original version

- **Ingestion and Q&A are separate.** Uploading a document embeds it once;
  asking a question never re-embeds anything.
- **Duplicate uploads are skipped** via a SHA-256 hash check against
  `chroma_db/manifest.json`.
- **Every chunk carries metadata** (filename, page, document_id, chunk_index,
  upload date, file type) — used for citations and document deletion.
- **Answers are cited**: each response shows which file/page it came from.
- **Retrieved chunks are reranked** with a cross-encoder before being sent
  to the LLM, instead of using raw hybrid search hits directly.
- **Conversation memory**: follow-up questions reuse recent chat history to
  resolve context, both when rewriting the search query and generating the
  answer.
- **Logging**: every query logs the rewritten query and retrieval / rerank /
  generation latency to `logs/app.log`.
- **Retrieval details are visible** in a small expander under each answer
  (rewritten query, per-chunk scores, latencies) — no separate debug file
  or evaluation library needed.

## Run

```bash
pip install -r requirements.txt   # your existing deps + sentence-transformers
streamlit run app.py
```

Add documents in the **Knowledge Base** tab first, then ask questions in
the **Ask** tab. Re-running with the same files is safe — duplicates are
detected and skipped automatically.
