"""
EPC Intelligence Core — ChromaDB Vector Store Interface

Manages ChromaDB collections for storing and querying document embeddings:
  - 'standards': Pre-indexed specification PDFs (TIA-942, BIS, etc.)
  - 'project_{id}': Per-project submittals, specs, RFI logs
"""

import logging
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

from backend.config import get_settings

logger = logging.getLogger("epc_intelligence.vector_store")

# Module-level client cache
_client: Optional[chromadb.PersistentClient] = None
_embedding_fn = None


def get_chroma_client() -> chromadb.PersistentClient:
    """Get or create the persistent ChromaDB client."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = chromadb.PersistentClient(path=settings.chroma_db_path)
        logger.info(f"ChromaDB client initialized at: {settings.chroma_db_path}")
    return _client


def get_embedding_function():
    """Get the sentence-transformer embedding function for ChromaDB."""
    global _embedding_fn
    if _embedding_fn is None:
        settings = get_settings()
        _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model,
        )
        logger.info(f"Embedding function loaded: {settings.embedding_model}")
    return _embedding_fn


def get_or_create_collection(collection_name: str) -> chromadb.Collection:
    """Get or create a ChromaDB collection with the configured embedding function."""
    client = get_chroma_client()
    ef = get_embedding_function()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )
    logger.info(
        f"Collection '{collection_name}': {collection.count()} documents."
    )
    return collection


def add_documents(
    chunks: list[dict],
    collection_name: str,
) -> int:
    """
    Add document chunks to a ChromaDB collection.

    Args:
        chunks: List of dicts with 'text' and 'metadata' keys
        collection_name: Target collection name

    Returns:
        Number of chunks added
    """
    if not chunks:
        logger.warning("No chunks to add.")
        return 0

    collection = get_or_create_collection(collection_name)

    # Prepare batch data
    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        source = chunk["metadata"].get("source", "unknown")
        page = chunk["metadata"].get("page_number", 0)
        chunk_idx = chunk["metadata"].get("chunk_index", i)
        doc_id = f"{source}_p{page}_c{chunk_idx}"

        ids.append(doc_id)
        documents.append(chunk["text"])
        metadatas.append(chunk["metadata"])

    # ChromaDB has a batch limit; process in batches of 500
    batch_size = 500
    total_added = 0
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        total_added += len(ids[start:end])

    logger.info(
        f"Added {total_added} chunks to collection '{collection_name}'. "
        f"Total in collection: {collection.count()}"
    )
    return total_added


def delete_collection(collection_name: str) -> bool:
    """Delete a collection entirely (for re-indexing)."""
    client = get_chroma_client()
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted collection '{collection_name}'.")
        return True
    except Exception as e:
        logger.warning(f"Failed to delete collection '{collection_name}': {e}")
        return False


def list_collections() -> list[str]:
    """List all collection names in the ChromaDB instance."""
    client = get_chroma_client()
    collections = client.list_collections()
    return [c.name if hasattr(c, "name") else str(c) for c in collections]


def collection_count(collection_name: str) -> int:
    """Return the number of documents in a collection."""
    try:
        collection = get_or_create_collection(collection_name)
        return collection.count()
    except Exception:
        return 0
