"""Groq LLM client with JSON mode and single repair-retry."""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import json_repair
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()

_client = None
_health_cache: tuple[float, bool] | None = None
_HEALTH_TTL = 30.0


def _get_client():
    global _client
    if _client is None:
        from groq import Groq

        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set")
        _client = Groq(api_key=GROQ_API_KEY, timeout=20.0)
    return _client


def _parse_json(text: str) -> dict | None:
    """Strict json → strip fences → json_repair. None if all fail."""
    if not text:
        return None
    t = text.strip()
    # Strip markdown fences
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0] if t.endswith("```") else t
        t = t.strip()
    if t.startswith("json"):
        t = t[4:].lstrip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    try:
        repaired = json_repair.loads(t)
        if isinstance(repaired, dict):
            return repaired
    except Exception:
        pass
    return None


def call_llm(messages: list[dict], system: str, *, json_mode: bool = True) -> dict | None:
    """Call Groq with JSON mode; one repair retry on malformed output.

    Returns parsed dict or None on hard failure.
    """
    client = _get_client()
    payload = [{"role": "system", "content": system}] + messages
    kwargs: dict[str, Any] = {
        "model": GROQ_MODEL,
        "messages": payload,
        "temperature": 0.2,
        "max_tokens": 900,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        log.error("Groq call failed: %s", e)
        return None

    parsed = _parse_json(raw)
    if parsed is not None:
        return parsed

    log.warning("First-pass JSON parse failed; raw=%r", raw[:200])
    # Repair-retry with stricter reminder
    repair_payload = payload + [
        {"role": "assistant", "content": raw},
        {
            "role": "user",
            "content": "Your previous reply was not valid JSON. Respond again with the strict JSON object only, no prose, no markdown.",
        },
    ]
    try:
        resp2 = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=repair_payload,
            temperature=0.0,
            max_tokens=900,
            response_format={"type": "json_object"},
        )
        raw2 = resp2.choices[0].message.content or ""
        parsed2 = _parse_json(raw2)
        if parsed2 is None:
            log.error("Repair-retry still produced invalid JSON: %r", raw2[:200])
        return parsed2
    except Exception as e:
        log.error("Repair-retry Groq call failed: %s", e)
        return None


def is_healthy() -> bool:
    """1-token ping with 30 s caching."""
    global _health_cache
    now = time.time()
    if _health_cache and now - _health_cache[0] < _HEALTH_TTL:
        return _health_cache[1]
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "ok"}],
            max_tokens=1,
            temperature=0.0,
        )
        ok = bool(resp.choices)
    except Exception as e:
        log.warning("Groq health check failed: %s", e)
        ok = False
    _health_cache = (now, ok)
    return ok
