"""Download the canonical SHL Individual Test Solutions catalog and normalise it.

Outputs:
    data/catalog.json      — normalised records the app consumes
    data/embeddings.npy    — MiniLM embeddings parallel to catalog order
    data/faiss.index       — FAISS IndexFlatIP over normalised embeddings

Idempotent; safe to re-run before each deploy.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest")

CANONICAL_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CATALOG_PATH = DATA / "catalog.json"
EMB_PATH = DATA / "embeddings.npy"
FAISS_PATH = DATA / "faiss.index"

# SHL canonical taxonomy: full category name → single-letter code shown in trace tables
KEYS_TO_LETTER = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _parse_duration(raw: str) -> int | None:
    if not raw:
        return None
    m = re.search(r"(\d+)", raw)
    return int(m.group(1)) if m else None


def _yesno(raw: str) -> bool:
    return (raw or "").strip().lower() == "yes"


def _derive_test_type(keys: list[str]) -> tuple[list[str], str]:
    """Return (letter list, display string e.g. 'C, K')."""
    letters = []
    for k in keys or []:
        letter = KEYS_TO_LETTER.get(k.strip())
        if letter and letter not in letters:
            letters.append(letter)
    return letters, ", ".join(letters)


def _bm25_blob(rec: dict) -> str:
    parts = [
        rec["name"],
        rec.get("description") or "",
        " ".join(rec.get("keys") or []),
        " ".join(rec.get("job_levels") or []),
    ]
    return " ".join(p for p in parts if p)


def normalise(raw_items: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in raw_items:
        keys = r.get("keys") or []
        letters, display = _derive_test_type(keys)
        rec = {
            "id": r.get("entity_id") or r.get("name", "").lower().replace(" ", "_"),
            "name": r.get("name", "").strip(),
            "url": r.get("link", "").strip(),
            "description": (r.get("description") or "").strip(),
            "duration_min": _parse_duration(r.get("duration", "")),
            "remote": _yesno(r.get("remote", "")),
            "adaptive": _yesno(r.get("adaptive", "")),
            "keys": keys,
            "test_type_letters": letters,
            "test_type": display,
            "job_levels": r.get("job_levels") or [],
            "languages": r.get("languages") or [],
        }
        if not rec["name"] or not rec["url"]:
            log.warning("Skipping item with missing name or url: %s", r.get("entity_id"))
            continue
        out.append(rec)
    return out


def build_embeddings(catalog: list[dict]) -> np.ndarray:
    from fastembed import TextEmbedding

    log.info("Loading embedding model %s (ONNX)", EMB_MODEL_NAME)
    model = TextEmbedding(EMB_MODEL_NAME)
    blobs = [_bm25_blob(rec) for rec in catalog]
    log.info("Encoding %d items…", len(blobs))
    emb = np.array(list(model.embed(blobs)), dtype=np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb


def build_faiss(emb: np.ndarray):
    import faiss

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def _load_raw_json(url: str) -> list[dict]:
    import json_repair

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    text = resp.text
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        log.warning("Raw JSON had decode issue (%s); repairing with json_repair", exc)
        return json_repair.loads(text)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    log.info("Downloading catalog from %s", CANONICAL_URL)
    raw = _load_raw_json(CANONICAL_URL)
    log.info("Got %d raw items", len(raw))

    catalog = normalise(raw)
    log.info("Normalised %d items", len(catalog))

    letter_counter: Counter[str] = Counter()
    for rec in catalog:
        for L in rec["test_type_letters"]:
            letter_counter[L] += 1
        if not rec["test_type_letters"]:
            letter_counter["(none)"] += 1
    log.info("Test-type letter distribution: %s", dict(letter_counter))

    CATALOG_PATH.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    log.info("Wrote %s (%d items)", CATALOG_PATH, len(catalog))

    emb = build_embeddings(catalog)
    np.save(EMB_PATH, emb)
    log.info("Wrote %s shape=%s", EMB_PATH, emb.shape)

    import faiss

    index = build_faiss(emb)
    faiss.write_index(index, str(FAISS_PATH))
    log.info("Wrote %s ntotal=%d", FAISS_PATH, index.ntotal)


if __name__ == "__main__":
    sys.exit(main() or 0)
