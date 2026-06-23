"""
EPC Intelligence Core — Schedule Risk Agent

Analyses project schedules (P6/JSON export) to identify:
  - Critical path risks
  - Float consumption
  - Activities at risk of causing cascade delays
  - Procurement-driven schedule risks

For each risk, suggests 2-3 mitigation actions with estimated recovery time.
"""

import json
import logging

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from backend.config import get_settings

logger = logging.getLogger("epc_intelligence.agents.schedule_risk")

SCHEDULE_RISK_SYSTEM_PROMPT = """You are an EPC scheduler analyst specialising in data centre construction projects. You have deep expertise in Primavera P6, critical path method (CPM), and EPC procurement cycles.

Given a project schedule with task durations, dependencies, and procurement status, you must:

1. IDENTIFY critical path risks — activities on or near the critical path that show schedule pressure
2. ANALYSE float consumption — tasks consuming float faster than expected
3. FLAG cascade risks — activities whose delay would cause ripple effects across multiple work packages
4. ASSESS procurement risks — long-lead items at risk of late delivery

For each risk identified, provide:
- The specific activity or work package
- Risk severity (critical/high/medium/low)
- Current status and trend
- 2-3 specific mitigation actions with estimated recovery time
- Impact if unmitigated (days of delay)

RESPONSE FORMAT — Return valid JSON:
{
  "schedule_health": "green|yellow|red",
  "summary": "Overall schedule assessment (2-3 sentences)",
  "critical_path_status": "Description of critical path health",
  "total_risks": <int>,
  "risks": [
    {
      "activity": "Activity name/WBS code",
      "risk_type": "critical_path|float_consumption|cascade|procurement",
      "severity": "critical|high|medium|low",
      "description": "What the risk is",
      "current_float_days": <float or null>,
      "impact_days": <estimated days of delay if unmitigated>,
      "trend": "worsening|stable|improving",
      "mitigation_actions": [
        {
          "action": "Specific action to take",
          "recovery_days": <estimated days recovered>,
          "owner": "Responsible party"
        }
      ]
    }
  ],
  "recommendations": ["Top-level recommendation 1", ...]
}

Return ONLY the JSON object."""


def _get_llm() -> ChatGroq:
    """Create the Groq LLM instance for schedule analysis."""
    settings = get_settings()
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.2,
        max_tokens=4096,
    )


async def analyse_schedule(schedule_data: dict | str) -> dict:
    """
    Analyse a project schedule for risks.

    Args:
        schedule_data: Either a dict (parsed JSON schedule) or
                       a string (raw schedule text/CSV)

    Returns:
        Parsed JSON dict with risk analysis
    """
    if isinstance(schedule_data, dict):
        schedule_text = json.dumps(schedule_data, indent=2)
    else:
        schedule_text = str(schedule_data)

    user_prompt = f"""Analyse the following project schedule data for a data centre EPC project.
Identify all critical path risks, float consumption issues, cascade risks, and procurement delays.

SCHEDULE DATA:
{schedule_text[:8000]}

Provide a comprehensive risk analysis with mitigation actions."""

    llm = _get_llm()
    messages = [
        SystemMessage(content=SCHEDULE_RISK_SYSTEM_PROMPT),
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
            "schedule_health": "unknown",
            "summary": response_text[:500],
            "risks": [],
            "parse_error": "Response was not valid JSON",
        }

    logger.info(
        f"Schedule analysis complete: health={result.get('schedule_health')}, "
        f"risks={result.get('total_risks', len(result.get('risks', [])))}"
    )
    return result
