"""Hybrid BM25 + dense retrieval over the SHL catalog with persisted FAISS index."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .catalog import Catalog, get_catalog

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FAISS_PATH = ROOT / "data" / "faiss.index"
EMB_PATH = ROOT / "data" / "embeddings.npy"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _doc_text(rec: dict) -> str:
    parts = [
        rec["name"],
        rec["name"],  # double the name to boost lexical match
        rec.get("description") or "",
        " ".join(rec.get("keys") or []),
        " ".join(rec.get("job_levels") or []),
    ]
    return " ".join(p for p in parts if p)


class HybridRetriever:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.docs = [_doc_text(rec) for rec in catalog.items]
        tokenized = [_tokenize(d) for d in self.docs]
        self.bm25 = BM25Okapi(tokenized)
        self.faiss_index = None
        self.embeddings: np.ndarray | None = None
        self._emb_model = None
        self._init_dense()

    def _init_dense(self):
        if not FAISS_PATH.exists() or not EMB_PATH.exists():
            log.warning("FAISS / embeddings missing; BM25-only mode")
            return
        try:
            import faiss

            self.faiss_index = faiss.read_index(str(FAISS_PATH))
            self.embeddings = np.load(EMB_PATH)
            log.info("Loaded FAISS index ntotal=%d dim=%d", self.faiss_index.ntotal, self.embeddings.shape[1])
        except Exception as e:
            log.warning("Failed to load FAISS index (%s); BM25-only mode", e)

    def _ensure_model(self):
        if self._emb_model is None:
            from fastembed import TextEmbedding

            log.info("Lazy-loading fastembed model")
            self._emb_model = TextEmbedding(EMB_MODEL_NAME)

    def _embed(self, text: str) -> np.ndarray:
        self._ensure_model()
        vec = next(iter(self._emb_model.embed([text])))
        v = np.array(vec, dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n else v

    def retrieve(self, query: str, top_k: int = 15) -> list[dict]:
        if not query.strip():
            return []
        bm25_scores = self.bm25.get_scores(_tokenize(query))
        bm25_ranks = list(np.argsort(-bm25_scores)[: top_k * 2])

        dense_ranks: list[int] = []
        if self.faiss_index is not None:
            try:
                q = self._embed(query).reshape(1, -1)
                _, idx = self.faiss_index.search(q, top_k * 2)
                dense_ranks = list(int(i) for i in idx[0] if i >= 0)
            except Exception as e:
                log.warning("Dense search failed (%s); BM25 only", e)

        # Reciprocal-Rank Fusion
        k_rrf = 60
        scores: dict[int, float] = {}
        for r, idx in enumerate(bm25_ranks):
            scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (k_rrf + r + 1)
        for r, idx in enumerate(dense_ranks):
            scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (k_rrf + r + 1)

        ordered = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        results = []
        for idx, score in ordered:
            rec = dict(self.catalog.items[idx])
            rec["_score"] = score
            results.append(rec)
        return results

    # Common anchors that appear in most SHL hiring batteries — surface them in every retrieval
    ANCHOR_URLS = (
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
    )

    def retrieve_from_history(self, messages: list[dict], top_k: int = 30) -> list[dict]:
        """Use all user messages with recency boost on the latest one. Inject standard anchors."""
        user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
        if not user_msgs:
            return []
        query = " ".join(user_msgs) + (" " + user_msgs[-1]) * 3
        results = self.retrieve(query, top_k=top_k)
        # Inject anchors at the END so they are always in the LLM's context, never displacing top hits
        present_urls = {r["url"] for r in results}
        for anchor_url in self.ANCHOR_URLS:
            if anchor_url not in present_urls:
                anchor_item = self.catalog.by_url(anchor_url)
                if anchor_item is not None:
                    item = dict(anchor_item)
                    item["_score"] = 0.0
                    item["_anchor"] = True
                    results.append(item)
        return results

    def compare(self, name_a: str, name_b: str) -> tuple[dict | None, dict | None]:
        return self.catalog.by_name_fuzzy(name_a), self.catalog.by_name_fuzzy(name_b)


_singleton: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _singleton
    if _singleton is None:
        _singleton = HybridRetriever(get_catalog())
    return _singleton
