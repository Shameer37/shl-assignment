"""Programmatic intent classifier for the conversational SHL recommender.

The agent uses these classifications to gate the LLM:
  - INJECTION / OFF_TOPIC → canned refusal, recs=[]
  - END                   → polite closer, end_of_conversation=True
  - VAGUE (turn 1)        → force a clarifier, recs=[]
  - COMPARISON            → bypass general retrieval, use the 2 named items
  - REFINEMENT / CONCRETE → standard retrieve + LLM
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Intent(Enum):
    VAGUE = "vague"
    CONCRETE = "concrete"
    COMPARISON = "comparison"
    REFINEMENT = "refinement"
    OFF_TOPIC = "off_topic"
    INJECTION = "injection"
    END = "end"


@dataclass
class Classification:
    intent: Intent
    user_turn_count: int
    extracted: dict = field(default_factory=dict)


_INJECTION_RE = re.compile(
    r"\b(ignore (the )?(previous|prior|above|all) (instructions?|prompt|rules?)"
    r"|act as (a|an) [a-z]+"
    r"|you are now"
    r"|jailbreak"
    r"|system prompt"
    r"|reveal (your|the) prompt"
    r"|disregard (the )?(previous|prior|above) (instructions?|prompt))",
    re.IGNORECASE,
)

_END_RE = re.compile(
    r"^\s*(thanks?( you)?\b|thx\b|ty\b|goodbye\b|bye\b|that('?s| is) all|perfect[.!]?$|great, thanks)",
    re.IGNORECASE,
)

_COMPARE_RE = re.compile(
    r"\b(compare|comparison|difference|differences|vs\.?|versus|how (does|do) .* (differ|compare)|"
    r"what'?s the difference|whats the difference)\b",
    re.IGNORECASE,
)

_REFINE_RE = re.compile(
    r"\b(add|drop|remove|exclude|swap|replace|update|change|switch|"
    r"instead|actually|also|prefer|but|swap out|in addition|on top of|"
    r"narrow|broaden|just keep|only keep)\b",
    re.IGNORECASE,
)

# Off-topic markers (when also no hiring/assessment lexicon)
_OFFTOPIC_HINTS = re.compile(
    r"\b(poem|sonnet|haiku|joke|recipe|cooking|weather|forecast|"
    r"capital of|translate|translation|write me a song|write a story|"
    r"meaning of life|who is the president|stock price|crypto price|"
    r"fix my code|debug my code|solve this equation)\b",
    re.IGNORECASE,
)

# Hiring-domain lexicon — presence keeps us on-topic
_HIRING_TOKENS = re.compile(
    r"\b(assess(ment)?s?|test|tests|candidate|candidates|hiring|hire|recruit|recruitment|"
    r"role|roles|seniority|junior|senior|mid-?level|graduate|grad|intern|"
    r"developer|engineer|manager|analyst|architect|consultant|sales|marketing|hr|leader|leadership|"
    r"shl|opq|mq|java|python|sql|excel|aws|docker|spring|react|angular|node|"
    r"personality|cognitive|numerical|verbal|abstract|inductive|deductive|"
    r"ability|aptitude|behavior|behaviour|skills?|knowledge|competenc(y|ies)|"
    r"team|stakeholder|jd|job description|battery|shortlist|recommend|recommendation)\b",
    re.IGNORECASE,
)

# Concrete signal: at least one role/skill/seniority/etc word
_CONCRETE_SIGNALS = _HIRING_TOKENS


def _last_user(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


def _user_turn_count(messages: list[dict]) -> int:
    return sum(1 for m in messages if m.get("role") == "user")


def _has_prior_assistant(messages: list[dict]) -> bool:
    return any(m.get("role") == "assistant" for m in messages)


def _extract_comparison_subjects(text: str) -> list[str]:
    """Pull out the things being compared. Naive: split on vs/and/&/,/or."""
    # Strip prefixes
    cleaned = re.sub(
        r"^.*?(compare|comparison|difference between|differences? between|whats? the difference between|how does)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\?$", "", cleaned).strip()
    # Split on vs/versus/and/&/or
    parts = re.split(r"\s+(?:vs\.?|versus|and|or|&)\s+", cleaned, flags=re.IGNORECASE)
    # Clean each
    out = []
    for p in parts:
        p = re.sub(r"[?,.;:]", "", p).strip()
        if 2 <= len(p) <= 80:
            out.append(p)
    return out[:2]


def classify(messages: list[dict]) -> Classification:
    last = _last_user(messages)
    turn_n = _user_turn_count(messages)

    if not last:
        return Classification(Intent.VAGUE, turn_n)

    # 1. Injection: highest priority
    if _INJECTION_RE.search(last):
        return Classification(Intent.INJECTION, turn_n)

    # 2. End-of-conversation
    if _END_RE.search(last) and _has_prior_assistant(messages):
        return Classification(Intent.END, turn_n)

    # 3. Comparison (must have at least 2 things to compare)
    if _COMPARE_RE.search(last):
        subjects = _extract_comparison_subjects(last)
        if len(subjects) >= 2:
            return Classification(Intent.COMPARISON, turn_n, {"subjects": subjects})

    # 4. Off-topic: explicit non-hiring hint AND no hiring tokens
    if _OFFTOPIC_HINTS.search(last) and not _HIRING_TOKENS.search(last):
        return Classification(Intent.OFF_TOPIC, turn_n)

    # 5. Refinement: prior assistant exists AND user is editing
    if _has_prior_assistant(messages) and _REFINE_RE.search(last):
        return Classification(Intent.REFINEMENT, turn_n)

    # 6. Concrete: has hiring/skill signals AND enough length
    has_signal = bool(_CONCRETE_SIGNALS.search(last))
    word_count = len(last.split())
    if has_signal and word_count >= 5:
        return Classification(Intent.CONCRETE, turn_n)
    # Or: prior context exists and short response (e.g., "mid-level around 4 years")
    if _has_prior_assistant(messages) and has_signal:
        return Classification(Intent.CONCRETE, turn_n)

    # 7. Default: vague
    return Classification(Intent.VAGUE, turn_n)
