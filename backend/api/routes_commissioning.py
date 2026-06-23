"""
EPC Intelligence Core — Commissioning API Routes

Endpoints:
  POST /api/commissioning/start          — Start a commissioning session
  POST /api/commissioning/submit-result  — Submit a measured value
  GET  /api/commissioning/export/{id}    — Export test record as PDF
  GET  /api/commissioning/session/{id}   — Get session state and records
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.agents.commissioning_agent import (
    get_session_state,
    start_session,
    submit_measurement,
    submit_measurement_streaming,
)
from backend.db.models import CommissioningSession, SystemType, get_db
from backend.db.test_record_store import (
    add_test_result,
    complete_session,
    create_session as create_db_session,
    export_test_record_pdf,
    get_session_records,
)

logger = logging.getLogger("epc_intelligence.api.commissioning")

router = APIRouter(prefix="/api/commissioning", tags=["Commissioning"])


# ── Request Models ─────────────────────────────────────────────────────────────


class StartSessionRequest(BaseModel):
    system_type: str  # "ups", "generator", "cooling", "fire_suppression", "bms"
    project_id: int


class SubmitResultRequest(BaseModel):
    session_id: int
    measured_value: str
    notes: Optional[str] = None
    stream: bool = False  # If True, use SSE streaming


class SaveTestRecordRequest(BaseModel):
    session_id: int
    step_number: int
    procedure: str
    expected_range: Optional[str] = None
    measured_value: Optional[float] = None
    measured_value_text: Optional[str] = None
    result: Optional[str] = None  # "pass", "fail", "skipped"
    notes: Optional[str] = None


# ── Endpoints ──────────────────────────────────────────────────────────────────


@router.post("/start")
async def start_commissioning_session(
    request: StartSessionRequest,
    db: Session = Depends(get_db),
):
    """
    Initialize a commissioning test session for a system type.

    Creates a DB record and starts the AI agent session.
    Returns session_id and the first test step.
    """
    # Validate system type
    try:
        SystemType(request.system_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid system_type '{request.system_type}'. "
            f"Must be one of: {[s.value for s in SystemType]}",
        )

    # Create DB session record
    db_session = create_db_session(
        db=db,
        project_id=request.project_id,
        system_type=request.system_type,
    )

    # Start the AI agent session
    try:
        first_step = await start_session(
            session_id=str(db_session.id),
            system_type=request.system_type,
            project_id=request.project_id,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start agent session: {e}"
        )

    return {
        "session_id": db_session.id,
        "system_type": request.system_type,
        "project_id": request.project_id,
        "status": "in_progress",
        "first_step": first_step,
    }


@router.post("/submit-result")
async def submit_test_result(
    request: SubmitResultRequest,
    db: Session = Depends(get_db),
):
    """
    Submit a measured value for the current test step.

    The AI agent evaluates pass/fail and streams the next step
    or NCR recommendation.
    """
    session_id_str = str(request.session_id)

    if request.stream:
        # SSE streaming response
        return StreamingResponse(
            submit_measurement_streaming(
                session_id=session_id_str,
                measured_value=request.measured_value,
                notes=request.notes,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # Synchronous response
        try:
            result = await submit_measurement(
                session_id=session_id_str,
                measured_value=request.measured_value,
                notes=request.notes,
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Measurement evaluation failed: {e}")

        return result


@router.post("/save-record")
async def save_test_record(
    request: SaveTestRecordRequest,
    db: Session = Depends(get_db),
):
    """Save a test record to the database (called after agent evaluation)."""
    # Verify session exists
    session = (
        db.query(CommissioningSession)
        .filter(CommissioningSession.id == request.session_id)
        .first()
    )
    if not session:
        raise HTTPException(
            status_code=404, detail=f"Session {request.session_id} not found."
        )

    record = add_test_result(
        db=db,
        session_id=request.session_id,
        system_type=session.system_type.value,
        step_number=request.step_number,
        procedure=request.procedure,
        expected_range=request.expected_range,
        measured_value=request.measured_value,
        measured_value_text=request.measured_value_text,
        result=request.result,
        notes=request.notes,
    )

    return {
        "record_id": record.id,
        "session_id": request.session_id,
        "step_number": record.step_number,
        "result": record.result.value if record.result else None,
    }


@router.get("/export/{session_id}")
async def export_session_pdf(
    session_id: int,
    db: Session = Depends(get_db),
):
    """Generate and download the as-commissioned test record PDF."""
    try:
        pdf_path = export_test_record_pdf(db, session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF export failed: {e}")

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"test_record_session_{session_id}.pdf",
    )


@router.get("/session/{session_id}")
async def get_session_details(
    session_id: int,
    db: Session = Depends(get_db),
):
    """Get the current state and all test records for a session."""
    # DB session info
    session = (
        db.query(CommissioningSession)
        .filter(CommissioningSession.id == session_id)
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    # Test records
    records = get_session_records(db, session_id)

    # Agent state (if active)
    agent_state = get_session_state(str(session_id))

    return {
        "session_id": session.id,
        "project_id": session.project_id,
        "system_type": session.system_type.value if session.system_type else None,
        "status": session.status.value if session.status else None,
        "started_at": session.started_at.isoformat() if session.started_at else None,
        "completed_at": session.completed_at.isoformat() if session.completed_at else None,
        "test_records": records,
        "agent_state": agent_state,
    }


@router.post("/complete/{session_id}")
async def complete_commissioning_session(
    session_id: int,
    db: Session = Depends(get_db),
):
    """Mark a commissioning session as completed."""
    session = complete_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    return {
        "session_id": session.id,
        "status": session.status.value,
        "completed_at": session.completed_at.isoformat() if session.completed_at else None,
    }
