"""
EPC Intelligence Core — Document Management API Routes

Endpoints:
  POST /api/documents/ingest  — Upload and ingest a PDF document
  GET  /api/documents/list    — List documents for a project
  POST /api/standards/index   — Trigger standards re-indexing
"""

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.db.models import DocType, Document, Project, get_db
from backend.rag.document_ingestion import ingest_pdf
from backend.rag.standards_loader import load_standards
from backend.rag.vector_store import add_documents, list_collections

logger = logging.getLogger("epc_intelligence.api.documents")

router = APIRouter(prefix="/api/documents", tags=["Documents"])


@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    doc_type: str = Form(...),
    project_id: int = Form(...),
    db: Session = Depends(get_db),
):
    """
    Upload a PDF document, extract text, chunk it, and store embeddings.

    - **file**: PDF file to upload
    - **doc_type**: One of 'submittal', 'spec', 'standard', 'rfi_log', 'schedule'
    - **project_id**: ID of the project this document belongs to

    Returns the document ID and number of chunks created.
    """
    # Validate doc_type
    try:
        doc_type_enum = DocType(doc_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid doc_type '{doc_type}'. Must be one of: {[e.value for e in DocType]}",
        )

    # Validate file is PDF
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Validate project exists
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found.")

    # Save uploaded file
    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / f"proj{project_id}_{file.filename}"
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"Saved upload: {file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Ingest and chunk the PDF — pass original filename so the ChromaDB
    # 'source' metadata key matches doc.filename stored in the SQL record.
    # The file on disk is prefixed (e.g. proj2_ups_submittal_fail.pdf) but the
    # DB record and all downstream queries use the un-prefixed original name.
    try:
        chunks = ingest_pdf(
            file_path=str(file_path),
            doc_type=doc_type,
            project_id=project_id,
            original_filename=file.filename,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    # Store chunks in ChromaDB
    collection_name = f"project_{project_id}"
    chunks_added = 0
    if chunks:
        chunks_added = add_documents(chunks, collection_name)

    # Save document record to database
    doc = Document(
        project_id=project_id,
        filename=file.filename,
        doc_type=doc_type_enum,
        chunk_count=chunks_added,
        file_path=str(file_path),
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    return {
        "doc_id": doc.id,
        "filename": doc.filename,
        "doc_type": doc_type,
        "project_id": project_id,
        "chunk_count": chunks_added,
        "collection": collection_name,
        "status": "ingested",
    }


@router.get("/list")
async def list_documents(
    project_id: int,
    db: Session = Depends(get_db),
):
    """List all documents for a project."""
    documents = (
        db.query(Document)
        .filter(Document.project_id == project_id)
        .order_by(Document.upload_date.desc())
        .all()
    )

    return {
        "project_id": project_id,
        "documents": [
            {
                "id": doc.id,
                "filename": doc.filename,
                "doc_type": doc.doc_type.value if doc.doc_type else None,
                "upload_date": doc.upload_date.isoformat() if doc.upload_date else None,
                "chunk_count": doc.chunk_count,
            }
            for doc in documents
        ],
    }


@router.get("/debug")
async def debug_chromadb(
    project_id: int,
    filename: str = None,
):
    """
    Diagnostic endpoint: inspect what is actually stored in ChromaDB for a
    given project collection.

    - **project_id**: The project whose ChromaDB collection to inspect.
    - **filename**: Optional. If provided, show only chunks whose metadata
      'source' matches this filename. If omitted, show all chunks (up to 50).

    Use this to diagnose 'source' key mismatches when the compliance agent
    queries by filename but finds nothing.
    """
    from backend.rag.vector_store import get_or_create_collection

    collection_name = f"project_{project_id}"
    try:
        collection = get_or_create_collection(collection_name)
        total = collection.count()

        if total == 0:
            return {
                "collection": collection_name,
                "total_chunks": 0,
                "chunks": [],
                "note": "Collection is empty — PDF was not ingested into ChromaDB.",
            }

        # Peek at all stored chunks (up to 50 to avoid huge payloads)
        peek = collection.get(
            limit=min(50, total),
            include=["metadatas", "documents"],
        )

        chunks = []
        for i, doc_text in enumerate(peek.get("documents") or []):
            meta = (peek.get("metadatas") or [{}])[i]
            chunks.append({
                "id": (peek.get("ids") or [None])[i],
                "text_preview": doc_text[:120] if doc_text else "",
                "metadata": meta,
            })

        # Filter client-side if filename requested
        if filename:
            matched = [c for c in chunks if c["metadata"].get("source") == filename]
            return {
                "collection": collection_name,
                "total_chunks": total,
                "filter_filename": filename,
                "matched_chunks": len(matched),
                "chunks": matched,
                "all_source_values": list({c["metadata"].get("source") for c in chunks}),
                "note": (
                    "MISMATCH: 'source' metadata does not match the filename "
                    "used by the compliance agent's where_filter."
                    if not matched and chunks else ""
                ),
            }

        return {
            "collection": collection_name,
            "total_chunks": total,
            "shown": len(chunks),
            "all_source_values": list({c["metadata"].get("source") for c in chunks}),
            "chunks": chunks,
        }

    except Exception as e:
        logger.error(f"ChromaDB debug error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"ChromaDB debug failed: {e}")


# ── Standards Management ───────────────────────────────────────────────────────

standards_router = APIRouter(prefix="/api/standards", tags=["Standards"])


@standards_router.post("/index")
async def index_standards(force: bool = False):
    """
    Trigger indexing of all PDFs in data/standards/.

    - **force**: If true, re-index even if already indexed.
    """
    try:
        result = load_standards(force_reindex=force)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Standards indexing failed: {e}")


@standards_router.get("/status")
async def standards_status():
    """Check the status of the standards collection."""
    collections = list_collections()
    from backend.rag.vector_store import collection_count

    return {
        "collections": collections,
        "standards_chunks": collection_count("standards"),
    }


# ── Project Management ────────────────────────────────────────────────────────

projects_router = APIRouter(prefix="/api/projects", tags=["Projects"])


@projects_router.post("/create")
async def create_project(
    name: str = Form(...),
    location: str = Form(""),
    tier_level: str = Form("III"),
    db: Session = Depends(get_db),
):
    """Create a new EPC project."""
    project = Project(name=name, location=location, tier_level=tier_level)
    db.add(project)
    db.commit()
    db.refresh(project)

    return {
        "project_id": project.id,
        "name": project.name,
        "location": project.location,
        "tier_level": project.tier_level,
        "created_at": project.created_at.isoformat() if project.created_at else None,
    }


@projects_router.get("/list")
async def list_projects(db: Session = Depends(get_db)):
    """List all projects."""
    projects = db.query(Project).order_by(Project.created_at.desc()).all()
    return {
        "projects": [
            {
                "id": p.id,
                "name": p.name,
                "location": p.location,
                "tier_level": p.tier_level,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in projects
        ]
    }
