"""
EPC Intelligence Core — RFI & Schedule API Routes

Endpoints:
  POST /api/rfi/ask             — RAG Q&A with source citations
  POST /api/schedule/analyse    — Schedule risk analysis
  POST /api/orchestrate         — Auto-route query to correct agent
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.agents.orchestrator import orchestrated_query
from backend.agents.rfi_rag_agent import ask_rfi, ask_rfi_streaming
from backend.agents.schedule_risk_agent import analyse_schedule

logger = logging.getLogger("epc_intelligence.api.rfi")

router = APIRouter(tags=["RFI & Schedule"])


# ── Request Models ─────────────────────────────────────────────────────────────


class RFIRequest(BaseModel):
    question: str
    project_id: int
    history: list[dict] = []  # Previous Q&A pairs for context
    stream: bool = False


class ScheduleRequest(BaseModel):
    schedule_data: dict | str  # JSON schedule or raw text
    project_id: int


class OrchestrateRequest(BaseModel):
    message: str
    project_id: int


# ── RFI Endpoints ──────────────────────────────────────────────────────────────


@router.post("/api/rfi/ask")
async def rfi_question(request: RFIRequest):
    """
    Ask a technical question — answered using RAG over all project documents.

    Returns answer with source citations, confidence level, and related standards.
    """
    if request.stream:
        return StreamingResponse(
            ask_rfi_streaming(
                question=request.question,
                project_id=request.project_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        result = await ask_rfi(
            question=request.question,
            project_id=request.project_id,
            history=request.history if request.history else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RFI query failed: {e}")

    return result


# ── Schedule Endpoints ─────────────────────────────────────────────────────────


@router.post("/api/schedule/analyse")
async def schedule_analysis(request: ScheduleRequest):
    """
    Analyse a project schedule for critical path risks, float consumption,
    and procurement delays. Returns risk items with severity and mitigation actions.
    """
    try:
        result = await analyse_schedule(schedule_data=request.schedule_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Schedule analysis failed: {e}")

    return result


# ── Orchestrator Endpoint ──────────────────────────────────────────────────────


@router.post("/api/orchestrate")
async def orchestrate(request: OrchestrateRequest):
    """
    Auto-route a message to the correct agent based on intent classification.

    The orchestrator classifies the message and dispatches it to:
    - Spec Compliance Agent
    - Commissioning Copilot
    - RFI Knowledge Agent
    - Schedule Risk Agent
    """
    try:
        result = await orchestrated_query(
            user_input=request.message,
            project_id=request.project_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orchestration failed: {e}")

    return result
