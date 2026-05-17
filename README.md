# SHL Conversational Assessment Recommender

Take-home assignment for the **SHL AI Intern (2026)** role. FastAPI service that recommends SHL Individual Test Solutions from a conversational query, grounded in the official 377-item catalog.

```
shl-recommender/
├── app/
│   ├── main.py        # FastAPI: /health, /chat, turn cap, schema-safe error handling
│   ├── agent.py       # Intent gate → retrieval → LLM → validate/refill
│   ├── intent.py      # Rule-based classifier (vague/concrete/comparison/refinement/off-topic/injection/end)
│   ├── retrieval.py   # BM25 + fastembed/FAISS + Reciprocal Rank Fusion
│   ├── catalog.py     # Catalog loader, fuzzy name lookup
│   ├── llm.py         # Groq client with JSON mode + repair-retry
│   ├── prompts.py     # System prompt, clarifier templates, context renderers
│   ├── validators.py  # URL grounding + refill-from-retrieval
│   └── models.py      # Pydantic schemas
├── data/
│   ├── catalog.json   # 377 SHL Individual Test Solutions (generated)
│   ├── faiss.index    # Persisted dense index (generated)
│   ├── embeddings.npy
│   └── traces/        # 10 sample conversation markdown files + parsed.json
├── scripts/
│   ├── ingest_catalog.py  # Download + normalise + embed + persist
│   ├── parse_traces.py    # Markdown → structured JSON
│   ├── probes.py          # 9 binary behavior probes
│   └── replay.py          # Trace replay + Recall@10 report
├── tests/                 # pytest unit tests
├── requirements.txt
├── runtime.txt
└── render.yaml
```

## Quickstart (local)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1. Build the catalog + FAISS index (one-time, ~10 s after model download)
python scripts/ingest_catalog.py

# 2. Set your Groq key
Copy-Item .env.example .env
# Edit .env and put your GROQ_API_KEY

# 3. Run the server
uvicorn app.main:app --reload --port 8000

# 4. Smoke test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" `
  -d '{"messages":[{"role":"user","content":"Hiring a senior Java backend engineer with Spring and SQL"}]}'
```

## Evaluation harness

```powershell
# Parse the 10 sample traces (one-time)
python scripts/parse_traces.py

# Replay all 10 against the local API; reports per-trace + mean Recall@10
python scripts/replay.py --api http://localhost:8000

# Run the 9 behavior probes
python scripts/probes.py --api http://localhost:8000

# All unit tests
pytest tests/ -v
```

## API

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /chat`

Request:
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What is the seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

Response:
```json
{
  "reply": "Got it. Here are 5 assessments that fit a mid-level Java dev with stakeholder needs.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/products/product-catalog/view/java-8-new/", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

**Schema invariants** (enforced):
- `reply` is always a string.
- `recommendations` is always an array; `[]` when clarifying or refusing.
- Each rec has exactly `name`, `url`, `test_type`; all URLs come from the catalog.
- Max 8 total messages per conversation; over-cap requests get `end_of_conversation: true`.
- `/chat` never raises — every error path returns a schema-valid body.

## Deploy on Render

1. Push to GitHub (the `.gitignore` blocks `.env`, `.venv/`, `*.zip`, and `assignment.pdf`).
2. New → Web Service → connect repo.
3. Render picks up `render.yaml` (`buildCommand` runs `pip install` + `scripts/ingest_catalog.py`; `startCommand` is `uvicorn app.main:app`).
4. Add a `GROQ_API_KEY` environment variable in the Render dashboard (it's declared `sync: false`).
5. After the deploy is green:
   ```powershell
   python scripts/replay.py --api https://<service>.onrender.com
   python scripts/probes.py --api https://<service>.onrender.com
   ```

See `APPROACH.md` for the full design write-up.
