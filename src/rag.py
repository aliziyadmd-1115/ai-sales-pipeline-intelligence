from __future__ import annotations

import os
from typing import Any

import requests

from .retrieval import Retriever


def build_grounded_context(query: str, retriever: Retriever, k: int = 3) -> tuple[str, list[dict[str, Any]]]:
    hits = retriever.search(query, k=k)
    citations = []
    blocks = []
    for hit in hits:
        citations.append({
            "opportunity_id": hit.opportunity_id,
            "industry": hit.industry,
            "product_line": hit.product_line,
            "outcome": hit.outcome,
            "similarity": hit.score,
        })
        blocks.append(
            f"[{hit.opportunity_id}] Industry={hit.industry}; Product={hit.product_line}; Outcome={hit.outcome}\n"
            f"Opportunity notes: {hit.notes}\nClose reason: {hit.close_reason}"
        )
    return "\n\n".join(blocks), citations


def answer_query(query: str, retriever: Retriever, use_llm: bool = False, k: int = 3) -> dict:
    context, citations = build_grounded_context(query, retriever, k=k)
    if not use_llm:
        return {
            "answer": (
                "LLM synthesis is disabled. Review the retrieved historical opportunities below as "
                "grounded evidence for the current sales analysis.\n\n" + context
            ),
            "citations": citations,
            "llm_used": False,
        }

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    prompt = f"""You are a sales analytics copilot for a business analyst.
Use ONLY the historical opportunities below. If the evidence is insufficient, say so.
Provide: (1) relevant historical patterns, (2) win/loss risk factors, and (3) recommended next analytical or commercial actions.
Cite opportunity IDs in brackets. Do not invent customer facts.

Current opportunity/question:
{query}

Historical opportunities:
{context}
"""
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=90,
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
        return {"answer": answer, "citations": citations, "llm_used": True, "model": model}
    except requests.RequestException as exc:
        return {
            "answer": "LLM API was unavailable; returning grounded retrieval context instead.\n\n" + context,
            "citations": citations,
            "llm_used": False,
            "warning": str(exc),
        }
