"""Parse the 10 sample conversation markdown traces into structured JSON.

Output: data/traces/parsed.json with shape:
[
  {
    "trace_id": "C1",
    "user_turns": ["...", "..."],          # all user utterances in order
    "agent_turns": [                       # parallel agent replies
        {"reply_prose": "...", "recommendations": [<rec rows>]}
    ],
    "gold_urls": ["https://www.shl.com/...", ...],   # URLs from the LAST committed recs
    "gold_recs": [<full rows>]
  }, ...
]
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACES_DIR = ROOT / "data" / "traces"
OUT_PATH = TRACES_DIR / "parsed.json"

TURN_RE = re.compile(r"^###\s+Turn\s+(\d+)\s*$", re.MULTILINE)


def _split_turns(md: str) -> list[str]:
    """Split the markdown into per-turn blocks. Returns list of bodies (skipping pre-turn header)."""
    parts = re.split(r"\n(?=###\s+Turn\s+\d+\s*\n)", md)
    return [p for p in parts if p.strip().startswith("### Turn")]


def _extract_user(block: str) -> str | None:
    """User content is inside a blockquote after **User**."""
    m = re.search(r"\*\*User\*\*\s*\n+((?:>\s.*\n?)+)", block)
    if not m:
        return None
    quoted = m.group(1)
    lines = [re.sub(r"^>\s?", "", ln).strip() for ln in quoted.splitlines() if ln.strip()]
    return "\n".join(lines).strip() or None


def _extract_agent_reply(block: str) -> str:
    """Agent reply prose: everything after **Agent** up to the first table or end-of-conversation marker."""
    m = re.search(r"\*\*Agent\*\*\s*\n+(.*?)(?=\n\| #|\n_No recommendations|\n_`end_of_conversation`|\Z)", block, re.DOTALL)
    return (m.group(1).strip() if m else "").strip()


def _extract_recommendations(block: str) -> list[dict]:
    """Pull markdown table rows. Columns: # | Name | Test Type | Keys | Duration | Languages | URL."""
    # Find the table (it starts with "| # | Name | ...")
    table_match = re.search(
        r"\|\s*#\s*\|\s*Name\s*\|.*?(\n\|[-:\s|]+\|\n)((?:\|.*\|\n?)+)",
        block,
        re.DOTALL | re.IGNORECASE,
    )
    if not table_match:
        return []
    body = table_match.group(2)
    rows = []
    for line in body.strip().split("\n"):
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 7:
            continue
        num, name, test_type, keys, duration, languages, url = cells[:7]
        # URL is wrapped: <https://...>
        url = re.sub(r"^<|>$", "", url).strip()
        rows.append({
            "name": name,
            "test_type": test_type,
            "keys": keys,
            "duration": duration,
            "languages": languages,
            "url": url,
        })
    return rows


def _extract_eoc(block: str) -> bool:
    m = re.search(r"`end_of_conversation`:\s*\*\*(true|false)\*\*", block, re.IGNORECASE)
    return bool(m and m.group(1).lower() == "true")


def parse_trace(path: Path) -> dict:
    md = path.read_text(encoding="utf-8", errors="replace")
    blocks = _split_turns(md)
    user_turns = []
    agent_turns = []
    for block in blocks:
        u = _extract_user(block)
        if u is not None:
            user_turns.append(u)
        agent_turns.append({
            "reply_prose": _extract_agent_reply(block),
            "recommendations": _extract_recommendations(block),
            "end_of_conversation": _extract_eoc(block),
        })

    # Gold = the LAST turn whose recommendations list is non-empty
    gold_recs: list[dict] = []
    for t in reversed(agent_turns):
        if t["recommendations"]:
            gold_recs = t["recommendations"]
            break
    gold_urls = [r["url"] for r in gold_recs if r.get("url")]

    return {
        "trace_id": path.stem,
        "user_turns": user_turns,
        "agent_turns": agent_turns,
        "gold_urls": gold_urls,
        "gold_recs": gold_recs,
    }


def main() -> None:
    files = sorted(TRACES_DIR.glob("C*.md"), key=lambda p: int(re.search(r"\d+", p.stem).group()))
    parsed = [parse_trace(p) for p in files]
    OUT_PATH.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    print(f"Parsed {len(parsed)} traces -> {OUT_PATH}")
    for t in parsed:
        print(f"  {t['trace_id']}: user_turns={len(t['user_turns'])} gold={len(t['gold_urls'])} urls")


if __name__ == "__main__":
    main()
