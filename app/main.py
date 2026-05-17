"""FastAPI surface for the SHL Conversational Assessment Recommender."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .agent import run_agent
from .catalog import get_catalog
from .models import ChatRequest, ChatResponse, HealthResponse
from .prompts import TURN_CAP_MSG
from .retrieval import get_retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

# PDF: "The evaluator caps each conversation at 8 turns including user & assistant"
MAX_TOTAL_TURNS = 8


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Lifespan startup: warming catalog + retriever")
    try:
        catalog = get_catalog()
        retriever = get_retriever()
        log.info("Catalog=%d items  faiss=%s", len(catalog), retriever.faiss_index is not None)
    except Exception as e:
        log.exception("Startup warmup failed: %s", e)
    yield
    log.info("Lifespan shutdown")


app = FastAPI(
    title="SHL Conversational Assessment Recommender",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Per spec: returns {"status":"ok"} with HTTP 200. Independent of LLM."""
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Stateless chat endpoint. Always returns a schema-valid ChatResponse."""
    try:
        msgs = [{"role": m.role, "content": m.content} for m in request.messages]

        if not msgs:
            return ChatResponse(
                reply="Please send at least one user message describing the role you're hiring for.",
                recommendations=[],
                end_of_conversation=False,
            )

        # Hard turn cap: total user + assistant messages
        if len(msgs) > MAX_TOTAL_TURNS:
            return ChatResponse(
                reply=TURN_CAP_MSG,
                recommendations=[],
                end_of_conversation=True,
            )

        if msgs[-1]["role"] != "user":
            return ChatResponse(
                reply="The most recent message must be from the user.",
                recommendations=[],
                end_of_conversation=False,
            )

        return run_agent(msgs)

    except Exception as e:
        log.exception("Top-level /chat error: %s", e)
        return ChatResponse(
            reply="Something went wrong. Please rephrase and try again.",
            recommendations=[],
            end_of_conversation=False,
        )
