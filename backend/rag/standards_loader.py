"""
EPC Intelligence Core — Standards Loader

Pre-indexes all specification PDFs in data/standards/ into the
'standards' ChromaDB collection. Idempotent — checks if the
collection already contains documents before re-processing.

Usage:
    # From CLI
    uv run python -m backend.rag.standards_loader

    # Or called programmatically on app startup
    from backend.rag.standards_loader import load_standards
    load_standards()
"""

import logging
from pathlib import Path

from backend.config import get_settings
from backend.rag.document_ingestion import ingest_pdf
from backend.rag.vector_store import (
    add_documents,
    collection_count,
    get_or_create_collection,
)

logger = logging.getLogger("epc_intelligence.standards_loader")

STANDARDS_COLLECTION = "standards"


def load_standards(force_reindex: bool = False) -> dict:
    """
    Index all PDFs in the standards directory into ChromaDB.

    Args:
        force_reindex: If True, re-index even if collection already has documents.

    Returns:
        Dict with indexing results: {total_files, total_chunks, skipped, errors}
    """
    settings = get_settings()
    standards_dir = Path(settings.standards_dir)

    if not standards_dir.exists():
        logger.warning(f"Standards directory not found: {standards_dir}")
        return {"total_files": 0, "total_chunks": 0, "skipped": 0, "errors": []}

    # Find all PDF files
    pdf_files = list(standards_dir.glob("*.pdf"))
    if not pdf_files:
        logger.info("No PDF files found in standards directory.")
        return {"total_files": 0, "total_chunks": 0, "skipped": 0, "errors": []}

    logger.info(f"Found {len(pdf_files)} PDF files in standards directory.")

    # Check if already indexed
    existing_count = collection_count(STANDARDS_COLLECTION)
    if existing_count > 0 and not force_reindex:
        logger.info(
            f"Standards collection already contains {existing_count} chunks. "
            "Skipping re-indexing. Use force_reindex=True to override."
        )
        return {
            "total_files": len(pdf_files),
            "total_chunks": existing_count,
            "skipped": len(pdf_files),
            "errors": [],
            "status": "already_indexed",
        }

    # Index each PDF
    total_chunks = 0
    errors = []
    indexed_files = []

    for pdf_path in pdf_files:
        try:
            logger.info(f"Indexing standard: {pdf_path.name}")
            chunks = ingest_pdf(
                file_path=str(pdf_path),
                doc_type="standard",
                project_id=None,  # Standards are global, not project-specific
            )
            if chunks:
                added = add_documents(chunks, STANDARDS_COLLECTION)
                total_chunks += added
                indexed_files.append(pdf_path.name)
                logger.info(f"  → {added} chunks indexed from {pdf_path.name}")
            else:
                logger.warning(f"  → No content extracted from {pdf_path.name}")
        except Exception as e:
            error_msg = f"Failed to index {pdf_path.name}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    result = {
        "total_files": len(pdf_files),
        "indexed_files": indexed_files,
        "total_chunks": total_chunks,
        "skipped": 0,
        "errors": errors,
        "status": "completed",
    }

    logger.info(
        f"Standards indexing complete: {len(indexed_files)}/{len(pdf_files)} files, "
        f"{total_chunks} total chunks."
    )
    return result


# Allow running as a standalone script
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    result = load_standards(force_reindex=True)
    print(f"\nIndexing result: {result}")
