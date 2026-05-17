"""Validate LLM output against the catalog and refill from retrieval when needed."""
from __future__ import annotations

import logging

from .catalog import Catalog
from .models import Recommendation

log = logging.getLogger(__name__)


def _coerce_to_catalog(name: str, url: str, catalog: Catalog) -> dict | None:
    """Find a catalog item matching either the URL or name. None if neither hits."""
    by_url = catalog.by_url(url)
    if by_url is not None:
        return by_url
    by_name = catalog.by_name_fuzzy(name)
    if by_name is not None:
        return by_name
    return None


def validate_recommendations(
    raw: list[dict],
    catalog: Catalog,
    refill_pool: list[dict] | None = None,
    min_recs: int = 3,
    max_recs: int = 10,
) -> list[Recommendation]:
    """Drop hallucinated items, enforce taxonomy, de-dupe, refill from retriever if short."""
    chosen_urls: set[str] = set()
    out: list[Recommendation] = []

    for r in raw or []:
        name = (r.get("name") or "").strip()
        url = (r.get("url") or "").strip()
        if not name and not url:
            continue
        item = _coerce_to_catalog(name, url, catalog)
        if item is None:
            log.warning("Dropping hallucinated rec: name=%r url=%r", name, url)
            continue
        if item["url"] in chosen_urls:
            continue
        chosen_urls.add(item["url"])
        out.append(
            Recommendation(
                name=item["name"],
                url=item["url"],
                test_type=item.get("test_type") or (item.get("test_type_letters") or [""])[0] or "",
            )
        )
        if len(out) >= max_recs:
            break

    # Refill from retrieval pool if shortlist is too short
    if refill_pool and len(out) < min_recs:
        for item in refill_pool:
            if len(out) >= min_recs:
                break
            if item["url"] in chosen_urls:
                continue
            chosen_urls.add(item["url"])
            out.append(
                Recommendation(
                    name=item["name"],
                    url=item["url"],
                    test_type=item.get("test_type") or (item.get("test_type_letters") or [""])[0] or "",
                )
            )
            log.info("Refilled rec from retrieval: %s", item["name"])

    return out[:max_recs]
