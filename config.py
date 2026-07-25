import os
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv()

# -------------------------------
# API Key
# -------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -------------------------------
# LLM Setup
# -------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY,
    temperature=0
)

# -------------------------------
# Embedding Model Setup
# -------------------------------
# This converts text chunks into vectors for ChromaDB.
# Loaded once here and imported everywhere else -> no repeated model loads.

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# Reranker Setup
# -------------------------------
# Cross-encoder that re-scores retrieved chunks against the query.
# Uses the same sentence-transformers dependency the embedding model needs,
# so no new library is introduced.

from sentence_transformers import CrossEncoder  # noqa: E402

reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# -------------------------------
# Storage Paths
# -------------------------------
# Everything the knowledge base needs lives inside chroma_db/ (vectors +
# manifest.json for document metadata). uploads/ is only a scratch folder
# for files mid-ingestion - nothing permanent is kept there.

CHROMA_DIR = "chroma_db"
UPLOADS_DIR = "uploads"
MANIFEST_PATH = os.path.join(CHROMA_DIR, "manifest.json")

os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# -------------------------------
# Retrieval Settings
# -------------------------------

TOP_K = 5           # chunks pulled by each of vector search / BM25
RERANK_TOP_K = 4    # chunks kept after reranking, sent to the LLM

# -------------------------------
# Logging
# -------------------------------
# One shared logger, writing to logs/app.log. Every module imports this
# instead of configuring logging itself.

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log")),
        logging.StreamHandler(),
    ],
)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger that writes to logs/app.log."""
    return logging.getLogger(name)
