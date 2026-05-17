"""Replay the 10 SHL conversation traces against our /chat and report Recall@10.

Two modes:
  - simulated user (--mode simulated): the script feeds the agent ONLY the trace's
    user turns in order; the agent must arrive at a final shortlist within 8 total
    turns. This mirrors the structure the SHL evaluator describes.
  - oracle-replay (--mode oracle): same as simulated. (Kept for parity; the SHL
    PDF says the harness uses a Groq-driven simulated user, but for grading our
    own agent the deterministic replay is more reproducible.)

Computes per-trace Recall@10 = |returned ∩ gold| / |gold|, then mean.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
PARSED_PATH = ROOT / "data" / "traces" / "parsed.json"


def post_chat(api: str, messages: list[dict], timeout: int = 60, retries: int = 2) -> dict:
    import time
    last_exc = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(f"{api.rstrip('/')}/chat", json={"messages": messages}, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    raise last_exc


def recall_at_k(retrieved_urls: list[str], gold_urls: list[str], k: int = 10) -> float:
    if not gold_urls:
        return 1.0
    top = set(retrieved_urls[:k])
    hits = sum(1 for g in gold_urls if g in top)
    return hits / len(gold_urls)


def replay_trace(api: str, trace: dict, verbose: bool = True) -> dict:
    """Walk the trace's user turns through /chat, return the agent's final shortlist."""
    history: list[dict] = []
    final_urls: list[str] = []
    turn_count = 0
    final_body = None
    for user_msg in trace["user_turns"]:
        history.append({"role": "user", "content": user_msg})
        if len(history) > 8:
            # Trace itself exceeds turn cap; stop and use what we have
            break
        body = post_chat(api, history)
        final_body = body
        final_urls = [r["url"] for r in body.get("recommendations", [])]
        # Add the agent's reply to history for next turn
        history.append({"role": "assistant", "content": body.get("reply", "")})
        turn_count += 1
        if body.get("end_of_conversation"):
            break
    return {
        "trace_id": trace["trace_id"],
        "turns_used": turn_count,
        "final_recommendations": final_urls,
        "final_reply": final_body.get("reply", "") if final_body else "",
        "gold_urls": trace["gold_urls"],
        "recall_at_10": recall_at_k(final_urls, trace["gold_urls"], k=10),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--trace", default=None, help="run only this trace_id (e.g. C5)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    parsed = json.loads(PARSED_PATH.read_text(encoding="utf-8"))
    if args.trace:
        parsed = [t for t in parsed if t["trace_id"] == args.trace]

    results = []
    for trace in parsed:
        try:
            r = replay_trace(args.api, trace, verbose=not args.quiet)
        except Exception as e:
            r = {"trace_id": trace["trace_id"], "error": str(e), "recall_at_10": 0.0}
        results.append(r)
        if not args.quiet and not args.json:
            mark = "OK " if r.get("recall_at_10", 0) >= 0.5 else "LOW"
            print(f"[{mark}] {r['trace_id']}: R@10={r.get('recall_at_10', 0):.2f}  turns={r.get('turns_used', '?')}  returned={len(r.get('final_recommendations', []))}")

    recalls = [r["recall_at_10"] for r in results]
    mean = sum(recalls) / len(recalls) if recalls else 0.0

    if args.json:
        print(json.dumps({"results": results, "mean_recall_at_10": mean}, indent=2))
    else:
        print(f"\n=== MEAN Recall@10: {mean:.3f}  ({len(results)} traces)")

    sys.exit(0 if mean >= 0.5 else 1)


if __name__ == "__main__":
    main()
