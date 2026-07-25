import os
import json
import uuid
import hashlib
import datetime
from typing import List, Dict, Any

from tools import (
    load_pdf,
    load_docx,
    load_csv,
    load_excel,
    split_documents,
    get_vectorstore,
    add_chunks_to_store,
    delete_document_chunks,
)
from config import MANIFEST_PATH, get_logger

logger = get_logger(__name__)


# -------------------------------
# File loading (by extension)
# -------------------------------

def load_any_file(file_path: str):
    """Detect file type by extension and load it with the right loader."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    elif ext == ".csv":
        return load_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        return load_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# -------------------------------
# Hashing (duplicate detection)
# -------------------------------

def compute_file_hash(file_path: str) -> str:
    """Compute a SHA-256 hash of a file's contents."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()


# -------------------------------
# Manifest (document registry: filename, hash, doc_id, chunk_count, ...)
# -------------------------------
# Chroma stores per-chunk vectors; this small JSON file is the one place
# that tracks "which documents exist" so we can check duplicates and list
# / delete whole documents without scanning the vector store.

def load_manifest() -> Dict[str, Any]:
    """Load the document manifest, or an empty one if it doesn't exist yet."""
    if not os.path.exists(MANIFEST_PATH):
        return {}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: Dict[str, Any]) -> None:
    """Persist the document manifest to disk."""
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def list_documents() -> List[Dict[str, Any]]:
    """Return metadata for every document currently in the knowledge base."""
    manifest = load_manifest()
    return list(manifest.values())


# -------------------------------
# Ingestion
# -------------------------------

def ingest_files(file_paths: List[str]) -> Dict[str, Any]:
    """
    Ingest one or more uploaded files into the persistent knowledge base:
    hash each file, skip anything already ingested, otherwise load -> split
    -> embed -> store, and record it in the manifest.

    Returns a summary dict: {"added": [...], "skipped_duplicates": [...]}
    """
    manifest = load_manifest()
    vectorstore = get_vectorstore()

    added, skipped = [], []

    for path in file_paths:
        filename = os.path.basename(path)
        file_hash = compute_file_hash(path)

        already_ingested = any(
            entry["file_hash"] == file_hash for entry in manifest.values()
        )
        if already_ingested:
            logger.info("Skipping duplicate upload: %s", filename)
            skipped.append(filename)
            continue

        document_id = str(uuid.uuid4())
        upload_date = datetime.datetime.utcnow().isoformat()
        file_type = os.path.splitext(filename)[1].lower().lstrip(".")

        raw_docs = load_any_file(path)
        chunks = split_documents(raw_docs)

        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "filename": filename,
                "document_id": document_id,
                "upload_date": upload_date,
                "chunk_index": i,
                "file_type": file_type,
                # "page" comes from the loader already (PDF loader sets it);
                # default to "-" for formats without page concept
                "page": chunk.metadata.get("page", "-"),
            })

        add_chunks_to_store(vectorstore, chunks)

        manifest[document_id] = {
            "document_id": document_id,
            "filename": filename,
            "file_hash": file_hash,
            "upload_date": upload_date,
            "chunk_count": len(chunks),
            "file_type": file_type,
        }
        added.append(filename)
        logger.info("Ingested %s as document_id=%s (%d chunks)", filename, document_id, len(chunks))

    save_manifest(manifest)
    return {"added": added, "skipped_duplicates": skipped}


def delete_document(document_id: str) -> bool:
    """Delete a document's chunks from the vector store and its manifest entry."""
    manifest = load_manifest()
    if document_id not in manifest:
        return False

    vectorstore = get_vectorstore()
    delete_document_chunks(vectorstore, document_id)

    filename = manifest[document_id]["filename"]
    del manifest[document_id]
    save_manifest(manifest)
    logger.info("Deleted document %s (document_id=%s)", filename, document_id)
    return True