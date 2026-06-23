"""
EPC Intelligence Core — Commissioning QA Copilot Agent

Guides engineers through integrated system testing (IST) for data centre
MEP systems. For each test step, provides:
  - The procedure to follow
  - Expected measurement range
  - Pass/fail criteria per TIA-942 Annex H
  - Safety precautions

When given a measured value, evaluates pass/fail and advises on next steps
or triggers NCR procedures.

Supports system types: UPS, Generator, Cooling, Fire Suppression, BMS
"""

import json
import logging
from typing import AsyncGenerator, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.config import get_settings
from backend.rag.retriever import format_context, retrieve

logger = logging.getLogger("epc_intelligence.agents.commissioning")

COMMISSIONING_SYSTEM_PROMPT = """You are a commissioning engineer guiding integrated system testing (IST) for a Tier III/IV data centre. You have deep expertise in TIA-942 Annex H commissioning procedures, Uptime Institute Tier requirements, and industry best practices for MEP system testing.

Your role is to guide an on-site engineer through step-by-step testing of data centre critical systems.

For each test step, you must provide:
1. **Procedure**: Clear, numbered instructions for the test
2. **Expected Range**: The acceptable measurement range per specifications
3. **Pass/Fail Criteria**: Specific criteria per TIA-942 Annex H
4. **Safety Precautions**: Any safety warnings relevant to this step
5. **Equipment Required**: Instruments needed for the measurement

When the engineer submits a measured value, you must:
1. Compare it against the acceptance criteria
2. If PASS: Acknowledge, record the result, and provide the NEXT test step
3. If FAIL: Explain the deviation, recommend immediate corrective action, and advise on NCR procedure
4. If MARGINAL: Flag for review with the lead commissioning agent

RESPONSE FORMAT — Always respond with valid JSON:
{
  "step_number": <int>,
  "test_name": "Name of this specific test",
  "procedure": ["Step 1...", "Step 2...", ...],
  "expected_range": "e.g., 230V ±2% (225.4V - 234.6V)",
  "pass_fail_criteria": "Specific criteria statement",
  "safety_precautions": ["Warning 1...", "Warning 2..."],
  "equipment_required": ["Instrument 1...", ...],
  "status": "pending|pass|fail|marginal",
  "measured_value": null or the submitted value,
  "evaluation": null or "Assessment of the measured value",
  "corrective_action": null or "Required action if failed",
  "next_action": "What the engineer should do next"
}

SYSTEM-SPECIFIC TEST SEQUENCES:

UPS System Tests:
1. Input voltage and frequency measurement
2. Battery bank voltage verification
3. Load transfer test (normal → battery)
4. Transfer time measurement (must be ≤10ms for Tier III)
5. Battery runtime verification under load
6. Retransfer test (battery → normal)
7. Bypass operation test
8. Harmonic distortion measurement (THD)

Generator System Tests:
1. Fuel system inspection and level verification
2. Battery start system test
3. Engine start time measurement (must be ≤10s for Tier III)
4. Voltage and frequency stabilization time
5. Load step test (0% → 25% → 50% → 75% → 100%)
6. Governor response test
7. Emergency stop test
8. Coolant temperature and oil pressure verification

Cooling System Tests:
1. Chilled water supply/return temperature differential
2. CRAH unit airflow measurement
3. Hot aisle / cold aisle temperature differential
4. Humidity level verification (40-60% RH per ASHRAE)
5. Redundancy switchover test (N+1)
6. Leak detection system test
7. BMS integration verification

Fire Suppression Tests:
1. Smoke detection sensitivity test
2. Pre-action valve operation test
3. Gas suppression agent weight/pressure verification
4. Abort switch functionality test
5. HVAC shutdown integration test
6. Audible/visual alarm test
7. Clean agent concentration verification

BMS (Building Management System) Tests:
1. Sensor calibration verification (temperature, humidity)
2. Alarm threshold configuration test
3. Trending and data logging verification
4. Remote monitoring connectivity test
5. Escalation procedure test
6. Integration with all MEP systems
7. Failover/redundancy test

Return ONLY the JSON object. Do not include any text before or after the JSON."""


# Session state storage (in-memory for MVP, could move to Redis/DB)
_active_sessions: dict[str, dict] = {}


def _get_llm() -> ChatGroq:
    """Create the Groq LLM instance for commissioning guidance."""
    settings = get_settings()
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.2,  # Slightly higher than compliance for more natural guidance
        max_tokens=4096,
    )


async def start_session(
    session_id: str,
    system_type: str,
    project_id: int,
) -> dict:
    """
    Initialize a commissioning session for a specific system type.

    Returns the first test step to perform.
    """
    # Retrieve relevant standards context
    spec_results = retrieve(
        query=f"{system_type} commissioning test procedure data centre TIA-942",
        collection_names=["standards"],
        top_k=5,
    )
    spec_context = format_context(spec_results)

    # Build initial prompt
    user_prompt = f"""Initialize a commissioning test sequence for a {system_type.upper()} system in a Tier III data centre.

Relevant specification context:
{spec_context}

Provide the FIRST test step in the sequence. Return JSON format."""

    llm = _get_llm()
    messages = [
        SystemMessage(content=COMMISSIONING_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    response_text = response.content.strip()

    # Parse JSON
    try:
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            json_lines = [l for l in lines if not l.startswith("```")]
            response_text = "\n".join(json_lines)
        result = json.loads(response_text)
    except json.JSONDecodeError:
        result = {
            "step_number": 1,
            "test_name": f"{system_type.upper()} Initial Test",
            "status": "pending",
            "raw_response": response_text[:1000],
        }

    # Store session state
    _active_sessions[session_id] = {
        "system_type": system_type,
        "project_id": project_id,
        "current_step": result.get("step_number", 1),
        "history": [
            SystemMessage(content=COMMISSIONING_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
            AIMessage(content=response.content),
        ],
        "test_results": [],
    }

    logger.info(
        f"Commissioning session '{session_id}' started for {system_type}. "
        f"First step: {result.get('test_name', 'N/A')}"
    )
    return result


async def submit_measurement(
    session_id: str,
    measured_value: str,
    notes: Optional[str] = None,
) -> dict:
    """
    Submit a measured value for the current test step.

    The agent evaluates pass/fail and returns the next step or NCR advice.
    """
    session = _active_sessions.get(session_id)
    if not session:
        raise ValueError(f"Session '{session_id}' not found. Start a new session.")

    # Build measurement submission prompt
    user_prompt = f"""The engineer has submitted a measured value for the current test step.

Measured value: {measured_value}
{f"Notes: {notes}" if notes else ""}

Evaluate this measurement against the acceptance criteria.
- If PASS: Record the result and provide the NEXT test step in the sequence.
- If FAIL: Explain the deviation, provide corrective action, and advise on NCR procedure.
- If MARGINAL: Flag for review.

Return the evaluation AND the next step (if pass) as JSON."""

    # Add to conversation history
    session["history"].append(HumanMessage(content=user_prompt))

    llm = _get_llm()
    response = llm.invoke(session["history"])

    # Update history
    session["history"].append(AIMessage(content=response.content))

    # Parse response
    response_text = response.content.strip()
    try:
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            json_lines = [l for l in lines if not l.startswith("```")]
            response_text = "\n".join(json_lines)
        result = json.loads(response_text)
    except json.JSONDecodeError:
        result = {
            "status": "error",
            "evaluation": response_text[:1000],
        }

    # Track result
    session["test_results"].append({
        "step": session["current_step"],
        "measured_value": measured_value,
        "result": result.get("status", "unknown"),
        "notes": notes,
    })

    if result.get("step_number"):
        session["current_step"] = result["step_number"]

    logger.info(
        f"Session '{session_id}' step {session['current_step']}: "
        f"value={measured_value}, result={result.get('status', 'unknown')}"
    )
    return result


async def submit_measurement_streaming(
    session_id: str,
    measured_value: str,
    notes: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Streaming version of submit_measurement for SSE."""
    session = _active_sessions.get(session_id)
    if not session:
        yield f"data: {json.dumps({'type': 'error', 'message': 'Session not found'})}\n\n"
        return

    user_prompt = f"""The engineer has submitted a measured value for the current test step.

Measured value: {measured_value}
{f"Notes: {notes}" if notes else ""}

Evaluate this measurement against the acceptance criteria.
If PASS, provide the NEXT test step. If FAIL, advise on corrective action and NCR.
Return as JSON."""

    session["history"].append(HumanMessage(content=user_prompt))

    llm = _get_llm()
    full_response = ""

    async for chunk in llm.astream(session["history"]):
        if chunk.content:
            full_response += chunk.content
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk.content})}\n\n"

    session["history"].append(AIMessage(content=full_response))

    # Parse final result
    try:
        clean = full_response.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join([l for l in lines if not l.startswith("```")])
        parsed = json.loads(clean)
        session["test_results"].append({
            "step": session["current_step"],
            "measured_value": measured_value,
            "result": parsed.get("status", "unknown"),
        })
        if parsed.get("step_number"):
            session["current_step"] = parsed["step_number"]
        yield f"data: {json.dumps({'type': 'result', 'data': parsed})}\n\n"
    except json.JSONDecodeError:
        yield f"data: {json.dumps({'type': 'result', 'data': {'raw': full_response[:1000]}})}\n\n"

    yield "data: [DONE]\n\n"


def get_session_state(session_id: str) -> Optional[dict]:
    """Get the current state of a commissioning session."""
    session = _active_sessions.get(session_id)
    if not session:
        return None
    return {
        "session_id": session_id,
        "system_type": session["system_type"],
        "project_id": session["project_id"],
        "current_step": session["current_step"],
        "test_results": session["test_results"],
        "total_steps_completed": len(session["test_results"]),
    }
