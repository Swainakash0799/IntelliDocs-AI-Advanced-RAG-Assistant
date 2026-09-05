# 📚 IntelliDocs AI — Advanced Hybrid RAG Assistant

> An advanced document question-answering system using **Hybrid RAG, BM25, Vector Search, Cross-Encoder Reranking, Query Rewriting, Conversational Memory, and Persistent ChromaDB**.

IntelliDocs AI lets users upload documents, build a persistent knowledge base, and ask questions using natural language. It combines semantic and keyword retrieval with reranking to provide relevant, source-cited answers.

---

## ✨ Features

* 📄 **Multi-format Document Support** — PDF, DOCX, CSV, XLSX
* 🔎 **Hybrid Retrieval** — Vector Search + BM25
* 🎯 **Cross-Encoder Reranking** — improves retrieved context relevance
* 🧠 **Query Rewriting** — converts conversational questions into focused queries
* 💾 **Persistent ChromaDB** — embeddings are stored and reused
* ♻️ **Duplicate Detection** — SHA-256 file hashing prevents re-ingestion
* 💬 **Conversational Memory** — supports contextual follow-up questions
* 📌 **Source Citations** — displays document and page information
* 🗂️ **Document Management** — upload, view and delete documents
* 📊 **Retrieval Debugging** — view retrieval, reranking and generation latency
* 🎨 **Streamlit UI** — clean document management and chat interface

---

## 🏗️ Architecture

```text
                    User
                     │
          ┌──────────┴──────────┐
          │                     │
       Upload                Question
          │                     │
          ▼                     ▼
   SHA-256 Hashing       Query Rewriting
          │                     │
    Duplicate Check             ▼
          │              Hybrid Retrieval
          ▼                ┌─────┴─────┐
   Document Loading        │           │
          │             Vector        BM25
          ▼                │           │
     Chunking              └─────┬─────┘
          │                      ▼
          ▼               Cross-Encoder
      Embeddings             Reranking
          │                       │
          ▼                       ▼
      ChromaDB              Top Relevant
                               Chunks
                                  │
                                  ▼
                             Groq LLM
                                  │
                                  ▼
                         Answer + Citations
```

---

## 🔄 RAG Pipeline

### 1. Document Ingestion

```text
Upload
  ↓
Hash & Duplicate Check
  ↓
Document Loader
  ↓
Text Chunking
  ↓
Metadata Enrichment
  ↓
Hugging Face Embeddings
  ↓
Persistent ChromaDB
```

Documents are split using:

```python
chunk_size = 1000
chunk_overlap = 150
```

---

### 2. Hybrid Retrieval

The system combines:

**Vector Search**

* Semantic similarity
* Understands meaning and context

**BM25**

* Keyword-based retrieval
* Useful for exact names, terms and technical keywords

```text
Query
  ↓
Vector Search + BM25
  ↓
Combined Results
  ↓
Cross-Encoder Reranking
  ↓
Top Relevant Chunks
```

---

### 3. Query Rewriting

Conversational queries are rewritten before retrieval.

Example:

```text
User:
"What are its limitations?"

        ↓

Rewritten Query:
"What are the limitations of the proposed approach?"
```

This improves retrieval for follow-up questions.

---

### 4. Reranking

Retrieved candidates are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

This helps select the most relevant chunks before sending context to the LLM.

---

## 💾 Persistent Knowledge Base

Documents are embedded once and stored locally in ChromaDB.

```text
Documents
    ↓
Embeddings
    ↓
ChromaDB
    ↓
Future Queries
```

This avoids repeatedly processing and embedding the same documents.

Document metadata and hashes are maintained through:

```text
chroma_db/manifest.json
```

---

## 🛠️ Tech Stack

| Category            | Technology                         |
| ------------------- | ---------------------------------- |
| Frontend            | Streamlit                          |
| Language            | Python                             |
| LLM                 | Groq / Qwen                        |
| Framework           | LangChain                          |
| Vector DB           | ChromaDB                           |
| Embeddings          | Hugging Face Sentence Transformers |
| Embedding Model     | `all-MiniLM-L6-v2`                 |
| Keyword Search      | BM25                               |
| Reranker            | Cross-Encoder                      |
| Reranker Model      | `ms-marco-MiniLM-L-6-v2`           |
| Document Processing | PyPDF, Docx2txt, Pandas            |
| Package Manager     | uv                            |

---

## 📂 Project Structure

```text
IntelliDocs-AI-Advanced-RAG-Assistant/
│
├── app.py                 # Streamlit UI
├── config.py              # LLM, embeddings, reranker and system configuration
├── tools.py               # Loaders, chunking, ChromaDB, BM25 and reranking
├── ingestion.py           # Document ingestion & deduplication
├── pipeline.py            # End-to-end ingestion and query pipeline 
├── agents.py              # Query rewriting and answer-generation chains
├── prompts.py             # Retrieval and answer-generation prompts
│
├── sample_data/           # Sample documents
├── chroma_db/             # Persistent vector database 
├── uploads/               # Uploaded files
├── logs/                  # Application logs
│
├── pyproject.toml         # Project configuration and dependencies
├── requirements.txt       # Python dependencies
├── uv.lock                # Locked dependency versions
└── README.md              # Project documentation
```

---

## 🚀 Installation

### 1. Clone

```bash
git clone https://github.com/Swainakash0799/IntelliDocs-AI-Advanced-RAG-Assistant.git

cd IntelliDocs-AI-Advanced-RAG-Assistant
```

### 3. Install Dependencies

```bash
uv sync
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

### 4. Configure API Key

Create `.env`:

```env
GROQ_API_KEY=your_groq_api_key
```

### 5. Run

```bash
streamlit run app.py
```

---

## 💬 Example

Upload a research paper and ask:

```text
What are the main findings of the paper?
```

Then continue:

```text
What are the limitations?

How does the proposed method compare with the previous approach?
```

The system uses conversation history, query rewriting, hybrid retrieval and reranking to construct the final context.

---

## 🎯 Why Hybrid RAG?

Traditional vector-only RAG can struggle with exact terminology.

IntelliDocs combines:

```text
Semantic Search
      +
Keyword Search
      ↓
Better Candidate Recall
      ↓
Cross-Encoder Reranking
      ↓
More Relevant Context
      ↓
LLM Answer
```

This provides a stronger retrieval pipeline than a basic vector-only RAG system.

---

## 📊 Observability

The application provides retrieval information including:

* Rewritten query
* Retrieved documents
* Retrieval scores
* Reranking scores
* Retrieval latency
* Reranking latency
* Generation latency

This helps analyze and improve RAG performance.

---

## 🔐 Security

Store API keys in `.env`:

```env
GROQ_API_KEY=your_api_key
```
---

## 🚧 Future Improvements

* [ ] RAGAS evaluation
* [ ] Streaming responses
* [ ] Metadata filtering
* [ ] Multi-query retrieval
* [ ] Query expansion
* [ ] Persistent chat history
* [ ] Authentication & multi-user support
* [ ] Qdrant / cloud vector database
* [ ] Docker & CI/CD

---

## 🧠 Concepts Demonstrated

**RAG • Hybrid Search • Vector Databases • BM25 • Embeddings • Cross-Encoder Reranking • Query Rewriting • Prompt Engineering • Conversational Retrieval • Document Chunking • Metadata Management • LLM Applications • LangChain • ChromaDB**

---

## 👨‍💻 Author

### Akash Swain
