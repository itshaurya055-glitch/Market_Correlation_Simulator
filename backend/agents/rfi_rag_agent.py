"""
EPC Intelligence Core — RFI Knowledge Agent

RAG-powered Q&A agent for answering technical questions about a data centre
EPC project. Uses all indexed documents (specs, submittals, RFI logs, standards)
as context and always cites source documents with page numbers.
"""

import json
import logging
from typing import AsyncGenerator

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from backend.config import get_settings
from backend.rag.retriever import format_context, retrieve

logger = logging.getLogger("epc_intelligence.agents.rfi_rag")

RFI_SYSTEM_PROMPT = """You are a technical advisor on a data centre EPC project. You have access to project specifications, vendor submittals, RFI logs, meeting minutes, and industry standards (TIA-942-B, BIS IS 3043, BICSI 002, Uptime Institute).

CRITICAL RULES:
1. Answer questions using ONLY the provided document context. Do not fabricate information.
2. ALWAYS cite the source document and page number for every claim. Format: [Source: document_name, Page X]
3. If the question was answered in a previous RFI, reference that RFI number.
4. If the context is insufficient to answer the question, say so explicitly: "The available documents do not contain sufficient information to answer this question. Consider submitting a formal RFI to [relevant party]."
5. For technical questions, provide the specific clause or section reference from the applicable standard.
6. Be precise and concise. EPC engineers need actionable answers, not lengthy explanations.

RESPONSE FORMAT:
{
  "answer": "Your detailed, cited answer",
  "citations": [
    {"source": "document_name", "page": page_number, "relevant_text": "brief excerpt"},
    ...
  ],
  "confidence": "high|medium|low",
  "related_standards": ["TIA-942-B Section X.Y", ...],
  "suggested_follow_up": "Any recommended follow-up action or additional RFI"
}

Return ONLY the JSON object."""


def _get_llm() -> ChatGroq:
    """Create the Groq LLM instance for RFI Q&A."""
    settings = get_settings()
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.2,
        max_tokens=4096,
    )


async def ask_rfi(
    question: str,
    project_id: int,
    history: list[dict] | None = None,
) -> dict:
    """
    Answer a technical question using RAG over all project documents.

    Args:
        question: The engineer's question
        project_id: Project ID to search within
        history: Optional conversation history for context

    Returns:
        Parsed JSON dict with answer, citations, and confidence
    """
    # Retrieve from both standards and project-specific collections
    collections = ["standards", f"project_{project_id}"]
    results = retrieve(
        query=question,
        collection_names=collections,
        top_k=8,
    )

    context = format_context(results)

    # Build prompt with history context
    history_text = ""
    if history:
        history_parts = []
        for h in history[-5:]:  # Last 5 exchanges
            history_parts.append(f"Q: {h.get('question', '')}\nA: {h.get('answer', '')}")
        history_text = f"\n\nPREVIOUS Q&A HISTORY:\n{'---'.join(history_parts)}\n"

    user_prompt = f"""QUESTION: {question}
{history_text}
DOCUMENT CONTEXT:
{context}

Answer the question using ONLY the document context provided. Cite all sources."""

    llm = _get_llm()
    messages = [
        SystemMessage(content=RFI_SYSTEM_PROMPT),
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
            "answer": response_text,
            "citations": [],
            "confidence": "low",
            "parse_note": "Response was not valid JSON",
        }

    logger.info(
        f"RFI answered: confidence={result.get('confidence', 'unknown')}, "
        f"citations={len(result.get('citations', []))}"
    )
    return result


async def ask_rfi_streaming(
    question: str,
    project_id: int,
) -> AsyncGenerator[str, None]:
    """Streaming version of ask_rfi for SSE."""
    collections = ["standards", f"project_{project_id}"]
    results = retrieve(query=question, collection_names=collections, top_k=8)
    context = format_context(results)

    user_prompt = f"""QUESTION: {question}

DOCUMENT CONTEXT:
{context}

Answer the question using ONLY the document context provided. Cite all sources."""

    llm = _get_llm()
    messages = [
        SystemMessage(content=RFI_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    full_response = ""
    async for chunk in llm.astream(messages):
        if chunk.content:
            full_response += chunk.content
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk.content})}\n\n"

    # Parse final response
    try:
        clean = full_response.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join([l for l in lines if not l.startswith("```")])
        parsed = json.loads(clean)
        yield f"data: {json.dumps({'type': 'result', 'data': parsed})}\n\n"
    except json.JSONDecodeError:
        yield f"data: {json.dumps({'type': 'result', 'data': {'answer': full_response}})}\n\n"

    yield "data: [DONE]\n\n"
