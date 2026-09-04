import streamlit as st
import tempfile
import os
from pipeline import run_ingestion, run_query
from ingestion import list_documents, delete_document

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Advanced RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #e8e4dc;
}

.stApp {
    background: #0a0a0f;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(80,140,255,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(80,200,255,0.08) 0%, transparent 55%);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }

.hero { text-align: center; padding: 2.5rem 0 1.5rem; }
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #4da3ff;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 800;
    line-height: 1.0;
    letter-spacing: -0.03em;
    color: #f0ebe0;
    margin: 0 0 1rem;
}
.hero h1 span { color: #4da3ff; }
.hero-sub {
    font-size: 1.0rem;
    font-weight: 300;
    color: #a09890;
    max-width: 560px;
    margin: 0 auto;
    line-height: 1.6;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(80,163,255,0.3), transparent);
    margin: 1.8rem 0;
}

.input-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(80,163,255,0.15);
    border-radius: 16px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.5rem;
}

.stButton > button {
    background: linear-gradient(135deg, #4da3ff 0%, #1a6bff 100%) !important;
    color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    width: 100%;
}

.doc-row {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 0.6rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
}

.answer-panel {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(80,163,255,0.2);
    border-radius: 16px;
    padding: 1.8rem 2.2rem;
    margin-top: 0.5rem;
}
.panel-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #4da3ff;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(80,163,255,0.15);
}
.citation-chip {
    display: inline-block;
    background: rgba(80,163,255,0.1);
    border: 1px solid rgba(80,163,255,0.25);
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    margin: 0.2rem 0.3rem 0 0;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #4da3ff;
}
.section-heading {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #f0ebe0;
    margin: 1.5rem 0 1rem;
}
.notice {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #605850;
    text-align: center;
    margin-top: 2.5rem;
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"question": ..., "answer": ...}
if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Advanced RAG Assistant</div>
    <h1>IntelliDocs<span>AI</span></h1>
    <p class="hero-sub">
        A persistent knowledge base you build once - then ask as many
        questions as you like without re-embedding a thing.
    </p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


tab_kb, tab_chat = st.tabs(["📁 Knowledge Base", "💬 Ask"])


# ── Tab 1: Knowledge Base management ─────────────────────────────────────────
with tab_kb:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "csv", "xlsx"],
        accept_multiple_files=True,
    )
    ingest_btn = st.button("⚡  Add to Knowledge Base", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if ingest_btn:
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            temp_dir = tempfile.mkdtemp(dir="uploads") if os.path.isdir("uploads") else tempfile.mkdtemp()
            saved_paths = []
            for f in uploaded_files:
                path = os.path.join(temp_dir, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                saved_paths.append(path)

            with st.spinner("Checking for duplicates and embedding new documents…"):
                summary = run_ingestion(saved_paths)

            if summary["added"]:
                st.success(f"Added: {', '.join(summary['added'])}")
            if summary["skipped_duplicates"]:
                st.info(f"Already in knowledge base, skipped: {', '.join(summary['skipped_duplicates'])}")

    st.markdown('<div class="section-heading">Documents in knowledge base</div>', unsafe_allow_html=True)
    docs = list_documents()
    if not docs:
        st.caption("No documents ingested yet.")
    else:
        for doc in docs:
            col_info, col_del = st.columns([5, 1])
            with col_info:
                st.markdown(
                    f"""<div class="doc-row">
                    {doc['filename']} · {doc['chunk_count']} chunks · {doc['file_type']} ·
                    uploaded {doc['upload_date'][:10]}
                    </div>""",
                    unsafe_allow_html=True,
                )
            with col_del:
                if st.button("Delete", key=f"del_{doc['document_id']}"):
                    delete_document(doc["document_id"])
                    st.rerun()


# ── Tab 2: Ask (chat, with memory) ───────────────────────────────────────────
with tab_chat:
    if not list_documents():
        st.warning("Add at least one document in the Knowledge Base tab before asking questions.")

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])

    question = st.chat_input("Ask a question about your documents…")

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating answer…"):
                result = run_query(question, chat_history=st.session_state.chat_history)
            st.markdown(result["answer"])

            if result["citations"]:
                chips = "".join(
                    f'<span class="citation-chip">{c["filename"]} · p.{c["page"]}</span>'
                    for c in result["citations"]
                )
                st.markdown(chips, unsafe_allow_html=True)

            with st.expander("🔍 Retrieval details"):
                st.caption(f"Rewritten search query: \"{result['search_query']}\"")
                st.json(result["debug"])
                st.caption(
                    f"Retrieval: {result['latencies']['retrieval_seconds']}s · "
                    f"Rerank: {result['latencies']['rerank_seconds']}s · "
                    f"Generation: {result['latencies']['generation_seconds']}s"
                )

        st.session_state.chat_history.append({"question": question, "answer": result["answer"]})
        st.session_state.last_result = result


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="notice">
    Advanced RAG · Persistent Hybrid Vector + BM25 Retrieval · Powered by Groq & ChromaDB
</div>
""", unsafe_allow_html=True)
