"""
EPC Intelligence Core — Spec Compliance Agent

The intellectual core of the platform. Compares vendor submittals against
indexed specification clauses and identifies technical deviations.

System Prompt: "You are a data centre QA engineer expert in TIA-942-B,
BIS IS 3043, and Uptime Institute Tier specifications..."

Returns structured JSON with clause references, deviation details,
severity levels, and remediation recommendations.
"""

import json
import logging
from typing import AsyncGenerator, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from backend.config import get_settings
from backend.rag.retriever import format_context, retrieve

logger = logging.getLogger("epc_intelligence.agents.spec_compliance")

SPEC_COMPLIANCE_SYSTEM_PROMPT = """You are a data centre QA engineer expert in TIA-942-B, BIS IS 3043, BICSI 002, and Uptime Institute Tier specifications.

Your role is to review vendor submittals against the relevant specification clauses and identify any technical deviations or non-conformances.

When given a vendor submittal excerpt and the relevant specification clauses, you must:

1. Compare each technical parameter in the submittal against the specification requirements.
2. Identify ANY deviations — including missing information, incorrect values, insufficient ratings, or non-compliant materials.
3. For each deviation found, return a structured JSON object.

CRITICAL RULES:
- Be thorough and precise. Missing a non-conformance in a data centre can cause outages or safety incidents.
- Always reference the specific clause or section number from the specification.
- If the submittal meets all requirements, return an empty deviations array.
- If you cannot determine compliance due to missing information, flag it as a deviation with severity "minor" and deviation_type "insufficient_information".

Return your analysis as valid JSON in this exact format:
{
  "summary": "Brief overall assessment (1-2 sentences)",
  "compliant": true/false,
  "deviations": [
    {
      "clause": "Specific clause reference (e.g., TIA-942-B Section 5.3.4.2)",
      "requirement": "What the spec requires",
      "submittal_value": "What the submittal states or provides",
      "required_value": "What should have been provided per the spec",
      "deviation_type": "One of: value_mismatch | missing_data | below_rating | material_non_compliance | design_deviation | documentation_gap",
      "severity": "One of: critical | major | minor",
      "recommendation": "Specific corrective action required"
    }
  ]
}

SEVERITY GUIDELINES:
- critical: Safety risk, Tier certification failure, or could cause data centre outage
- major: Specification non-compliance that affects performance or reliability
- minor: Documentation gaps, marginal values, or minor discrepancies

Return ONLY the JSON object. Do not include any text before or after the JSON."""


def _get_llm() -> ChatGroq:
    """Create the Groq LLM instance for compliance checking."""
    settings = get_settings()
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.1,  # Low temperature for deterministic compliance analysis
        max_tokens=4096,
    )


async def check_compliance(
    submittal_text: str,
    project_id: int,
    spec_doc_ids: Optional[list[str]] = None,
) -> dict:
    """
    Run a compliance check on submittal text against indexed specifications.

    Args:
        submittal_text: The vendor submittal content to check
        project_id: Project ID for context
        spec_doc_ids: Optional list of specific spec doc IDs to check against

    Returns:
        Parsed JSON dict with compliance analysis results
    """
    # Retrieve relevant specification clauses from ChromaDB
    collections_to_search = ["standards"]
    project_collection = f"project_{project_id}"
    collections_to_search.append(project_collection)

    # Use the submittal text as the query to find relevant spec clauses
    spec_results = retrieve(
        query=submittal_text[:1000],  # Use first 1000 chars as query
        collection_names=collections_to_search,
        top_k=8,
        where_filter={"doc_type": "standard"} if spec_doc_ids is None else None,
    )

    spec_context = format_context(spec_results)

    # Build the prompt
    user_prompt = f"""## VENDOR SUBMITTAL TO CHECK:
{submittal_text}

## RELEVANT SPECIFICATION CLAUSES:
{spec_context}

Analyse the submittal against the specification clauses above. Identify ALL technical deviations and non-conformances. Return your analysis as structured JSON."""

    # Call Groq LLM
    llm = _get_llm()
    messages = [
        SystemMessage(content=SPEC_COMPLIANCE_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        response = llm.invoke(messages)
        response_text = response.content.strip()

        # Parse JSON from response (handle markdown code blocks)
        if response_text.startswith("```"):
            # Strip markdown code block
            lines = response_text.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            response_text = "\n".join(json_lines)

        result = json.loads(response_text)
        logger.info(
            f"Compliance check complete. Compliant: {result.get('compliant')}, "
            f"Deviations: {len(result.get('deviations', []))}"
        )
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.debug(f"Raw response: {response_text[:500]}")
        return {
            "summary": "Analysis completed but response parsing failed.",
            "compliant": None,
            "deviations": [],
            "raw_response": response_text[:2000],
            "parse_error": str(e),
        }
    except Exception as e:
        logger.error(f"Compliance check failed: {e}")
        raise


async def check_compliance_streaming(
    submittal_text: str,
    project_id: int,
) -> AsyncGenerator[str, None]:
    """
    Streaming version of compliance check — yields SSE-formatted chunks.

    Used by the /api/compliance/check endpoint for real-time frontend updates.
    """
    # Retrieve spec context
    collections_to_search = ["standards", f"project_{project_id}"]
    spec_results = retrieve(
        query=submittal_text[:1000],
        collection_names=collections_to_search,
        top_k=8,
    )
    spec_context = format_context(spec_results)

    user_prompt = f"""## VENDOR SUBMITTAL TO CHECK:
{submittal_text}

## RELEVANT SPECIFICATION CLAUSES:
{spec_context}

Analyse the submittal against the specification clauses above. Identify ALL technical deviations and non-conformances. Return your analysis as structured JSON."""

    llm = _get_llm()
    messages = [
        SystemMessage(content=SPEC_COMPLIANCE_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    full_response = ""
    async for chunk in llm.astream(messages):
        if chunk.content:
            full_response += chunk.content
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk.content})}\n\n"

    # After streaming, try to parse the complete response
    try:
        clean_text = full_response.strip()
        if clean_text.startswith("```"):
            lines = clean_text.split("\n")
            json_lines = [
                l for l in lines if not l.startswith("```")
            ]
            clean_text = "\n".join(json_lines)

        parsed = json.loads(clean_text)
        yield f"data: {json.dumps({'type': 'result', 'data': parsed})}\n\n"
    except json.JSONDecodeError:
        yield f"data: {json.dumps({'type': 'result', 'data': {'raw_response': full_response[:2000]}})}\n\n"

    yield "data: [DONE]\n\n"
