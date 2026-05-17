"""SHL catalog loader and lookup helpers."""
from __future__ import annotations

import json
import logging
from difflib import SequenceMatcher
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG_PATH = ROOT / "data" / "catalog.json"


class Catalog:
    def __init__(self, items: list[dict]):
        self.items = items
        self._by_url: dict[str, dict] = {it["url"]: it for it in items}
        self._by_name_lower: dict[str, dict] = {it["name"].lower(): it for it in items}

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CATALOG_PATH) -> "Catalog":
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        log.info("Loaded %d catalog items from %s", len(items), path)
        return cls(items)

    def by_url(self, url: str) -> dict | None:
        return self._by_url.get((url or "").strip())

    def by_name_exact(self, name: str) -> dict | None:
        return self._by_name_lower.get((name or "").lower().strip())

    def by_name_fuzzy(self, name: str, min_ratio: float = 0.55) -> dict | None:
        """Substring-first, then SequenceMatcher fallback."""
        q = (name or "").lower().strip()
        if not q:
            return None
        if q in self._by_name_lower:
            return self._by_name_lower[q]
        for nm, item in self._by_name_lower.items():
            if q in nm or nm in q:
                return item
        best = None
        best_r = 0.0
        for nm, item in self._by_name_lower.items():
            r = SequenceMatcher(None, q, nm).ratio()
            if r > best_r:
                best_r = r
                best = item
        return best if best_r >= min_ratio else None

    def __len__(self) -> int:
        return len(self.items)


_singleton: Catalog | None = None


def get_catalog() -> Catalog:
    global _singleton
    if _singleton is None:
        _singleton = Catalog.load()
    return _singleton
