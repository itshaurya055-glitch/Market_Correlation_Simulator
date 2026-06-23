"""
EPC Intelligence Core — Compliance API Routes

Endpoints:
  POST /api/compliance/check      — Run spec compliance check (SSE streaming)
  GET  /api/ncr/list               — List NCRs for a project
  PUT  /api/ncr/{ncr_id}/status    — Update NCR status
  GET  /api/ncr/summary            — NCR summary by severity/status
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.agents.spec_compliance_agent import (
    check_compliance,
    check_compliance_streaming,
)
from backend.db.models import Document, get_db
from backend.db.ncr_store import (
    create_ncrs_from_compliance_result,
    get_ncr_summary,
    list_ncrs,
    update_ncr_status,
)
from backend.rag.document_ingestion import ingest_pdf
from backend.rag.retriever import retrieve, format_context

logger = logging.getLogger("epc_intelligence.api.compliance")

router = APIRouter(tags=["Compliance"])


# ── Request Models ─────────────────────────────────────────────────────────────


class ComplianceCheckRequest(BaseModel):
    doc_id: int
    project_id: int
    spec_ids: list[int] = []  # Optional specific spec doc IDs


class ComplianceTextRequest(BaseModel):
    text: str
    project_id: int


class NCRStatusUpdate(BaseModel):
    status: str  # "open", "resolved", "waived"


# ── Compliance Endpoints ───────────────────────────────────────────────────────


@router.post("/api/compliance/check")
async def run_compliance_check(
    request: ComplianceCheckRequest,
    db: Session = Depends(get_db),
):
    """
    Run a spec compliance check on an uploaded document.

    Retrieves the document's chunks from ChromaDB, compares against
    specification clauses using the Groq LLM agent, and saves any
    NCRs to the database.

    Returns the full compliance analysis with deviations.
    """
    # Get document record
    doc = db.query(Document).filter(Document.id == request.doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {request.doc_id} not found.")

    # Retrieve the document's content from ChromaDB
    collection_name = f"project_{request.project_id}"
    doc_chunks = retrieve(
        query="full document content",
        collection_names=[collection_name],
        top_k=20,
        where_filter={"source": doc.filename},
    )

    if not doc_chunks:
        raise HTTPException(
            status_code=400,
            detail=f"No indexed content found for document '{doc.filename}'. Please re-ingest.",
        )

    # Combine all chunks into submittal text
    submittal_text = "\n\n".join([chunk["text"] for chunk in doc_chunks])

    # Run compliance check
    try:
        result = await check_compliance(
            submittal_text=submittal_text,
            project_id=request.project_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compliance check failed: {e}")

    # Save NCRs to database
    if result.get("deviations"):
        created_ncrs = create_ncrs_from_compliance_result(
            db=db,
            project_id=request.project_id,
            doc_id=request.doc_id,
            compliance_result=result,
        )
        result["ncrs_created"] = len(created_ncrs)

    return result


@router.post("/api/compliance/check-stream")
async def run_compliance_check_stream(
    request: ComplianceTextRequest,
):
    """
    Streaming compliance check — sends results via SSE as the LLM generates them.

    Use this endpoint for real-time UI updates.
    """
    return StreamingResponse(
        check_compliance_streaming(
            submittal_text=request.text,
            project_id=request.project_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── NCR Endpoints ─────────────────────────────────────────────────────────────


@router.get("/api/ncr/list")
async def get_ncr_list(
    project_id: int,
    severity: str = None,
    status: str = None,
    db: Session = Depends(get_db),
):
    """
    List all NCRs for a project, sorted by severity (critical first).

    Optional filters: severity (critical/major/minor), status (open/resolved/waived)
    """
    ncrs = list_ncrs(
        db=db,
        project_id=project_id,
        severity_filter=severity,
        status_filter=status,
    )
    return {
        "project_id": project_id,
        "total": len(ncrs),
        "ncrs": ncrs,
    }


@router.put("/api/ncr/{ncr_id}/status")
async def change_ncr_status(
    ncr_id: int,
    update: NCRStatusUpdate,
    db: Session = Depends(get_db),
):
    """Update the status of an NCR."""
    try:
        result = update_ncr_status(db, ncr_id, update.status)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if result is None:
        raise HTTPException(status_code=404, detail=f"NCR {ncr_id} not found.")

    return result


@router.get("/api/ncr/summary")
async def ncr_summary(
    project_id: int,
    db: Session = Depends(get_db),
):
    """Get a summary of NCRs by severity and status for a project."""
    return get_ncr_summary(db, project_id)
