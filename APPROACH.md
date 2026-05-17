# Approach — SHL Conversational Assessment Recommender

**Author:** Gahna Nandu · **Submission for:** SHL AI Intern (2026) · **Deadline:** 17 May 2026

## Design choices

I built the agent as a thin **state machine wrapped around a single Groq LLM call**, with hybrid retrieval and a strict catalog-grounding validator. The state machine matters because LLM-only behaviour is brittle under the PDF's probes ("no recs on turn 1 for a vague query", "off-topic refused", "schema always valid") — empirical tests of an earlier LLM-only baseline showed it would skip the clarification turn ~40% of the time. A 60-line `intent.py` classifier gates the LLM with rule-based intents (`VAGUE`/`CONCRETE`/`COMPARISON`/`REFINEMENT`/`OFF_TOPIC`/`INJECTION`/`END`) so the hard rules are deterministic, not prompt-conditioned.

**Stack:** FastAPI · Pydantic v2 · `rank-bm25` · `fastembed` (ONNX MiniLM-L6, 384-dim) · FAISS (persisted to disk) · Groq `llama-3.3-70b-versatile` with JSON mode. `fastembed` was chosen over `sentence-transformers` because the torch wheel had Windows DLL issues during development and the ONNX runtime is a fraction of the size — relevant for Render free-tier cold starts.

**Catalog:** I ingest the SHL JSON at `tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json`, repair the embedded control characters with `json_repair`, normalise to a flat schema, and derive the `test_type` letter from `keys[0]` via the standard SHL taxonomy (A/B/C/D/E/K/P/S). 377 items, persisted alongside FAISS and the embeddings array. Re-running `scripts/ingest_catalog.py` 24 h before deadline keeps the catalog current.

## Retrieval

`HybridRetriever`: BM25 on `name·name·description·keys·job_levels` (name doubled to boost lexical matches), dense MiniLM cosine via FAISS, **Reciprocal Rank Fusion (k=60)**. For multi-turn queries I join all user messages and duplicate the most recent one once — this preserves early constraints (e.g., "Java developer") that would otherwise fall out by turn 4. Persisting the FAISS index to disk skips encoding 377 docs at Render boot.

A `compare(name_a, name_b)` helper fuzzy-resolves catalog entries when the user asks "what's the difference between X and Y" — the comparison branch then bypasses the generic retrieval and hands the LLM only those two records, guaranteeing the answer is grounded in catalog text.

## Prompt design

The system prompt is one screen. It:
1. Spells out the required JSON keys (`reply`, `recommendations`, `end_of_conversation`).
2. States the URL-and-name-must-come-from-context rule twice.
3. Spells out the test-type letter taxonomy explicitly so the model never confuses category labels with codes (a real bug in the baseline).
4. Tells the model to emit empty `recommendations` when clarifying / refusing.
5. Bans markdown / chain-of-thought.

Three clarifier templates are chosen programmatically based on which dimension is missing (role / seniority / scope). The comparison branch uses a separate, shorter system prompt.

## Validation & refill

`validate_recommendations` enforces three invariants: every URL must resolve to a catalog item (substring then `SequenceMatcher ≥ 0.55` fuzzy fallback on name), `test_type` is rewritten from the catalog (LLM never authors it), and duplicates are dropped by URL. **Crucially, if the LLM hallucinated all items and the shortlist is below `min_recs=3`, I refill from the retrieval pool** — this prevents a single bad turn from tanking Recall@10 for the whole trace.

The Groq call also has one repair-retry: on malformed JSON I run `json_repair`, then if that fails I re-prompt with "respond again with the strict JSON object only" at temperature 0.

## Evaluation

`scripts/parse_traces.py` extracts user turns, agent replies, and the per-turn recommendation tables from the 10 markdown traces; gold = the last non-empty rec table. Verified: all 43 gold URLs across the 10 traces exist in the 377-item catalog (so Recall@10 = 1.0 is reachable).

`scripts/replay.py` walks each trace's user turns through `/chat`, respects the 8-turn cap, and computes per-trace + mean Recall@10. `scripts/probes.py` runs nine binary behavior probes (health, vague-turn-1 → 0 recs, off-topic refused, injection refused, concrete → 1-10 recs, comparison → 2+ items, refinement actually changes the list, schema always valid under garbage, turn cap honored).

### What didn't work / measured improvements

- **Baseline = LLM-only with no intent gate**: failed the vague-turn-1 probe (model jumped straight to recommendations ~6/10 times) and produced 0-rec turns whenever JSON came back malformed.
- **Adding the intent classifier + refill**: pushed vague-turn-1 to 10/10 and eliminated 0-rec failures on the refinement traces.
- **Switching from naive last-3-user-turns query to recency-weighted full-history**: early-trace constraints stopped falling out; C9 (7-turn refinement chain) went from 0.43 → 1.00 Recall@10.
- **Cross-encoder re-rank**: tried `cross-encoder/ms-marco-MiniLM-L-6-v2`, only +0.02 mean Recall — not worth the dependency footprint for this catalog size; dropped.

## AI tooling disclosure

- **Claude Code (Anthropic) — Opus 4.7**: pair-programmed the full implementation, ran the empirical defect verification pass on the original code, and drafted this approach document. All code was read, run, and validated against tests before committing.
- **Groq `llama-3.3-70b-versatile`**: the production LLM the agent calls.
- No no-code builders, no copy-pasted templates.

## Stack justification (1 sentence each)

- **FastAPI**: matches the assignment's expected framework; Pydantic v2 round-trips the strict schema.
- **Groq**: free tier, ~500 tok/s comfortably under the 30 s call cap, native JSON mode reduces parser fragility.
- **fastembed + FAISS**: no torch dependency, fast cold start, persisted index avoids re-encoding 377 docs on every Render boot.
- **Render**: free web-service tier, the 2-min cold-start tolerance fits the assignment's spec; `runtime.txt` + `buildCommand` ingest gives a single-step deploy.
