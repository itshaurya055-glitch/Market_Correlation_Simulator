"""
EPC Intelligence Core — LangGraph Multi-Agent Orchestrator

Routes user intent to the correct agent using a LangGraph StateGraph.
Nodes: spec_compliance, commissioning, rfi, schedule_risk
The router node classifies intent and dispatches to the appropriate agent.
"""

import json
import logging
from typing import Any, Literal, TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from backend.config import get_settings

logger = logging.getLogger("epc_intelligence.agents.orchestrator")


# ── Agent State ────────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    """Shared state passed between LangGraph nodes."""
    user_input: str
    intent: str
    project_id: int
    context: dict[str, Any]
    result: dict[str, Any]


# ── Intent Classifier ─────────────────────────────────────────────────────────

INTENT_CLASSIFIER_PROMPT = """You are an intent classifier for an EPC data centre intelligence platform. Given a user message, classify the intent into exactly one of these categories:

- "spec_compliance": User wants to check a submittal, verify specifications, find non-conformances, or review vendor documentation
- "commissioning": User wants to run a commissioning test, start a test procedure, submit test measurements, or review test records
- "rfi": User has a technical question about the project, wants to search documents, or needs information from specs/standards
- "schedule": User wants to analyse project schedule, check critical path, review delays, or assess procurement timelines

Return ONLY a JSON object with the classified intent:
{"intent": "spec_compliance|commissioning|rfi|schedule", "confidence": 0.0-1.0, "reasoning": "brief explanation"}"""


def _get_classifier_llm() -> ChatGroq:
    """Lightweight LLM for intent classification."""
    settings = get_settings()
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.0,
        max_tokens=200,
    )


async def classify_intent(user_input: str) -> dict:
    """Classify user intent into one of the four agent categories."""
    llm = _get_classifier_llm()
    messages = [
        SystemMessage(content=INTENT_CLASSIFIER_PROMPT),
        HumanMessage(content=user_input),
    ]

    response = llm.invoke(messages)
    response_text = response.content.strip()

    try:
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            json_lines = [l for l in lines if not l.startswith("```")]
            response_text = "\n".join(json_lines)
        result = json.loads(response_text)
    except json.JSONDecodeError:
        # Default to RFI for general questions
        result = {"intent": "rfi", "confidence": 0.5, "reasoning": "fallback"}

    logger.info(
        f"Intent classified: {result.get('intent')} "
        f"(confidence={result.get('confidence')})"
    )
    return result


# ── Route by Intent ────────────────────────────────────────────────────────────


def route_by_intent(
    state: AgentState,
) -> Literal["spec_agent", "cx_agent", "rfi_agent", "schedule_agent"]:
    """Route to the correct agent node based on classified intent."""
    intent = state.get("intent", "rfi")

    route_map = {
        "spec_compliance": "spec_agent",
        "commissioning": "cx_agent",
        "rfi": "rfi_agent",
        "schedule": "schedule_agent",
    }

    return route_map.get(intent, "rfi_agent")


# ── Orchestrated Query ─────────────────────────────────────────────────────────


async def orchestrated_query(
    user_input: str,
    project_id: int,
) -> dict:
    """
    High-level orchestration: classify intent → route to agent → return result.

    This is a simplified orchestration for the MVP. The full LangGraph
    StateGraph implementation would add:
    - Multi-turn conversation management
    - Agent chaining (e.g., compliance check → auto-generate NCR)
    - Parallel agent queries
    - Memory/context persistence
    """
    # Step 1: Classify intent
    intent_result = await classify_intent(user_input)
    intent = intent_result.get("intent", "rfi")

    # Step 2: Route to agent
    result = {"intent": intent, "confidence": intent_result.get("confidence")}

    if intent == "spec_compliance":
        from backend.agents.spec_compliance_agent import check_compliance
        agent_result = await check_compliance(
            submittal_text=user_input,
            project_id=project_id,
        )
        result["agent"] = "spec_compliance"
        result["data"] = agent_result

    elif intent == "commissioning":
        # For commissioning, we need a session — return guidance
        result["agent"] = "commissioning"
        result["data"] = {
            "message": (
                "To start a commissioning session, use POST /api/commissioning/start "
                "with your system_type (ups/generator/cooling/fire_suppression/bms)."
            ),
        }

    elif intent == "rfi":
        from backend.agents.rfi_rag_agent import ask_rfi
        agent_result = await ask_rfi(
            question=user_input,
            project_id=project_id,
        )
        result["agent"] = "rfi"
        result["data"] = agent_result

    elif intent == "schedule":
        from backend.agents.schedule_risk_agent import analyse_schedule
        agent_result = await analyse_schedule(schedule_data=user_input)
        result["agent"] = "schedule_risk"
        result["data"] = agent_result

    return result
