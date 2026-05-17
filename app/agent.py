"""Agent orchestrator: intent gate → retrieval → LLM → validate/refill."""
from __future__ import annotations

import logging

from .catalog import Catalog, get_catalog
from .intent import Intent, classify
from .llm import call_llm
from .models import ChatResponse, Recommendation
from .prompts import (
    CLARIFIER_TEMPLATES,
    COMPARISON_SYSTEM_PROMPT,
    END_REPLY,
    GENERIC_ERROR,
    REFUSAL_INJECTION,
    REFUSAL_OFFTOPIC,
    SYSTEM_PROMPT,
    render_pair_context,
    render_retrieval_context,
)
from .retrieval import HybridRetriever, get_retriever
from .validators import validate_recommendations

log = logging.getLogger(__name__)


def _empty(reply: str, end: bool = False) -> ChatResponse:
    return ChatResponse(reply=reply, recommendations=[], end_of_conversation=end)


def _pick_clarifier(messages: list[dict]) -> str:
    last = next((m["content"].lower() for m in reversed(messages) if m["role"] == "user"), "")
    if any(w in last for w in ("dev", "engineer", "manager", "analyst", "sales", "consult", "lead", "intern")):
        return CLARIFIER_TEMPLATES["seniority"]
    if any(w in last for w in ("senior", "junior", "mid", "executive", "graduate", "entry")):
        return CLARIFIER_TEMPLATES["role"]
    return CLARIFIER_TEMPLATES["generic"]


def _handle_comparison(
    subjects: list[str],
    messages: list[dict],
    retriever: HybridRetriever,
    catalog: Catalog,
) -> ChatResponse:
    a, b = retriever.compare(subjects[0], subjects[1])
    if not a or not b:
        # Fall back to standard retrieval if we can't pin down both
        log.warning("Comparison: couldn't resolve subjects %r — falling back to retrieval", subjects)
        return _handle_concrete_or_refinement(messages, retriever, catalog)
    context = render_pair_context(a, b)
    parsed = call_llm(messages, COMPARISON_SYSTEM_PROMPT + "\n\n" + context)
    if not parsed:
        # Fallback: synthesize a minimal reply with the two records
        return ChatResponse(
            reply=f"{a['name']} measures {(a.get('keys') or ['—'])[0]}; {b['name']} measures {(b.get('keys') or ['—'])[0]}.",
            recommendations=[
                Recommendation(name=a["name"], url=a["url"], test_type=a.get("test_type", "")),
                Recommendation(name=b["name"], url=b["url"], test_type=b.get("test_type", "")),
            ],
        )
    recs = validate_recommendations(parsed.get("recommendations") or [], catalog, refill_pool=[a, b], min_recs=2)
    # Always include the two compared items even if the LLM forgot them
    if len(recs) < 2:
        recs = [
            Recommendation(name=a["name"], url=a["url"], test_type=a.get("test_type", "")),
            Recommendation(name=b["name"], url=b["url"], test_type=b.get("test_type", "")),
        ]
    return ChatResponse(
        reply=(parsed.get("reply") or "").strip() or f"Comparison of {a['name']} and {b['name']}.",
        recommendations=recs,
        end_of_conversation=bool(parsed.get("end_of_conversation", False)),
    )


def _handle_concrete_or_refinement(
    messages: list[dict],
    retriever: HybridRetriever,
    catalog: Catalog,
) -> ChatResponse:
    retrieved = retriever.retrieve_from_history(messages, top_k=30)
    context = render_retrieval_context(retrieved, max_items=30)
    parsed = call_llm(messages, SYSTEM_PROMPT + "\n\n" + context)
    if not parsed:
        # LLM hard failure — fall back to retrieval-driven shortlist
        recs = validate_recommendations([], catalog, refill_pool=retrieved, min_recs=8)
        return ChatResponse(
            reply="Here is a shortlist for your role.",
            recommendations=recs,
        )
    recs = validate_recommendations(
        parsed.get("recommendations") or [],
        catalog,
        refill_pool=retrieved,
        min_recs=8,
    )
    return ChatResponse(
        reply=(parsed.get("reply") or "").strip() or "Here is the shortlist.",
        recommendations=recs,
        end_of_conversation=bool(parsed.get("end_of_conversation", False)),
    )


def run_agent(messages: list[dict]) -> ChatResponse:
    """Top-level entry point. Always returns a schema-valid ChatResponse."""
    try:
        catalog = get_catalog()
        retriever = get_retriever()
        c = classify(messages)
        log.info("Intent=%s user_turns=%d extracted=%s", c.intent.value, c.user_turn_count, c.extracted)

        if c.intent == Intent.INJECTION:
            return _empty(REFUSAL_INJECTION)
        if c.intent == Intent.OFF_TOPIC:
            return _empty(REFUSAL_OFFTOPIC)
        if c.intent == Intent.END:
            return _empty(END_REPLY, end=True)
        if c.intent == Intent.VAGUE and c.user_turn_count <= 1:
            return _empty(_pick_clarifier(messages))
        if c.intent == Intent.COMPARISON:
            return _handle_comparison(c.extracted.get("subjects", []), messages, retriever, catalog)

        # CONCRETE, REFINEMENT, or VAGUE-after-turn-1
        return _handle_concrete_or_refinement(messages, retriever, catalog)

    except Exception as e:
        log.exception("Unhandled agent error: %s", e)
        return _empty(GENERIC_ERROR)
