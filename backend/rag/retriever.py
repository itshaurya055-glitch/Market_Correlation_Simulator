"""
EPC Intelligence Core — RAG Retriever

Performs similarity search across ChromaDB collections to find
relevant document chunks for a given query. Supports:
  - Multi-collection search (e.g., standards + project docs)
  - Metadata filtering (by doc_type, project_id)
  - Top-K retrieval with relevance scores
"""

import logging
from typing import Optional

from backend.config import get_settings
from backend.rag.vector_store import get_or_create_collection

logger = logging.getLogger("epc_intelligence.retriever")


def _safe_query(collection, query_texts: list[str], n_results: int, where: Optional[dict] = None) -> Optional[dict]:
    """
    Execute a ChromaDB query safely, handling the common edge-case where
    n_results exceeds the number of documents that match the `where` filter.

    ChromaDB raises InvalidArgumentError (or NotEnoughElementsException in newer
    versions) when n_results > len(filtered_results). We progressively reduce
    n_results down to 1 before giving up.
    """
    # Clamp n_results to total collection size first
    total = collection.count()
    n_results = min(n_results, total)
    if n_results == 0:
        return None

    for attempt_n in range(n_results, 0, -1):
        try:
            kwargs = {"query_texts": query_texts, "n_results": attempt_n}
            if where:
                kwargs["where"] = where
            result = collection.query(**kwargs)
            if attempt_n < n_results:
                logger.debug(
                    f"ChromaDB query succeeded with n_results={attempt_n} "
                    f"(reduced from {n_results}) — where filter matched fewer docs."
                )
            return result
        except Exception as e:
            err_lower = str(e).lower()
            # Retry only on the well-known "not enough elements" errors
            if any(kw in err_lower for kw in ("not enough", "brute force", "n_results", "greater than")):
                logger.debug(f"Reducing n_results from {attempt_n} due to: {e}")
                continue
            # Any other error — surface it immediately
            raise

    logger.warning(
        f"ChromaDB _safe_query: could not retrieve any results with where={where}. "
        "The where filter may not match any stored metadata."
    )
    return None


def retrieve(
    query: str,
    collection_names: list[str],
    top_k: Optional[int] = None,
    where_filter: Optional[dict] = None,
) -> list[dict]:
    """
    Perform similarity search across one or more ChromaDB collections.

    Args:
        query: The search query text
        collection_names: List of collection names to search
        top_k: Number of results per collection (defaults to config TOP_K_RETRIEVAL)
        where_filter: Optional ChromaDB metadata filter dict
            e.g. {"doc_type": "standard"} or {"project_id": "42"}

    Returns:
        List of result dicts, sorted by relevance (best first):
          - text: the chunk content
          - metadata: source, page_number, doc_type, etc.
          - distance: cosine distance (lower = more relevant)
          - collection: which collection it came from
    """
    settings = get_settings()
    if top_k is None:
        top_k = settings.top_k_retrieval

    all_results = []

    for collection_name in collection_names:
        try:
            collection = get_or_create_collection(collection_name)

            total_docs = collection.count()
            if total_docs == 0:
                logger.info(f"Collection '{collection_name}' is empty, skipping.")
                continue

            # Use the safe query helper to handle the n_results > filtered_count
            # edge-case that causes ChromaDB to fail silently or raise.
            results = _safe_query(
                collection=collection,
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
            )

            if results is None:
                logger.warning(
                    f"No results from '{collection_name}' with where_filter={where_filter}. "
                    f"Collection has {total_docs} total docs."
                )
                # Diagnostic: try without filter to confirm data is present
                if where_filter:
                    diag = _safe_query(collection, [query], n_results=1, where=None)
                    if diag and diag["documents"] and diag["documents"][0]:
                        sample_meta = diag["metadatas"][0][0] if diag["metadatas"] else {}
                        logger.warning(
                            f"Diagnostic: collection '{collection_name}' HAS data but the "
                            f"where_filter {where_filter} matched nothing. "
                            f"Sample stored metadata: {sample_meta}"
                        )
                continue

            # Parse results
            if results and results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    result = {
                        "text": doc,
                        "metadata": (
                            results["metadatas"][0][i]
                            if results["metadatas"]
                            else {}
                        ),
                        "distance": (
                            results["distances"][0][i]
                            if results["distances"]
                            else 0.0
                        ),
                        "collection": collection_name,
                    }
                    all_results.append(result)

            retrieved_count = len(results["documents"][0]) if results and results["documents"] else 0
            logger.info(
                f"Retrieved {retrieved_count} chunks from '{collection_name}' "
                f"(total in collection: {total_docs}, where_filter: {where_filter})."
            )

        except Exception as e:
            logger.error(f"Retrieval error in '{collection_name}': {e}", exc_info=True)

    # Sort by distance (cosine — lower is better)
    all_results.sort(key=lambda x: x["distance"])

    # Limit total results to top_k
    if len(all_results) > top_k:
        all_results = all_results[:top_k]

    logger.info(
        f"Total retrieved: {len(all_results)} chunks from "
        f"{len(collection_names)} collections."
    )
    return all_results


def format_context(results: list[dict]) -> str:
    """
    Format retrieved chunks into a context string for LLM consumption.

    Each chunk is prefixed with its source document and page number
    for citation tracking.
    """
    if not results:
        return "No relevant context found."

    context_parts = []
    for i, result in enumerate(results):
        source = result["metadata"].get("source", "Unknown")
        page = result["metadata"].get("page_number", "?")
        context_parts.append(
            f"[Source {i+1}: {source}, Page {page}]\n{result['text']}"
        )

    return "\n\n---\n\n".join(context_parts)
