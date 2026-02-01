"""
Policy retriever with metadata filtering (city, doc_type).
Works with any VectorStore implementation (InMemory, OpenSearch).
"""

from __future__ import annotations

from typing import Any

from .schemas import RetrievedExcerpt


def _norm_city(city: str) -> str:
    return " ".join((city or "").strip().split()).lower()


class PolicyRetriever:
    """Retrieves relevant policy excerpts by semantic similarity and metadata filters."""

    def __init__(self, vector_store):
        self._store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        city: str | None = None,
        doc_type: str | None = None,
    ) -> list[RetrievedExcerpt]:
        """
        Retrieve policy excerpts matching the query.
        Metadata filters: city (e.g. "Providence"), doc_type (e.g. "incident_response").
        """
        filters: dict[str, Any] = {}
        if city:
            filters["city"] = city
        if doc_type:
            filters["doc_type"] = doc_type

        # 1) Strict pass (with filters if present)
        strict = self._store.search(
            query=query,
            top_k=top_k,
            filters=filters if filters else None,
        ) or []

        # If we got something decent, return.
        # (Assumes store returns sorted by score desc; if not, your store should.)
        if strict:
            return strict[:top_k]

        # 2) Fallback pass (no city filter, keep doc_type if provided)
        relaxed_filters: dict[str, Any] = {}
        if doc_type:
            relaxed_filters["doc_type"] = doc_type

        relaxed = self._store.search(
            query=query,
            top_k=max(top_k, 10),
            filters=relaxed_filters if relaxed_filters else None,
        ) or []

        if not city or not relaxed:
            return relaxed[:top_k]

        # 3) Soft city preference: if metadata city matches loosely, prefer those
        target = _norm_city(city)

        def city_match_score(ex: RetrievedExcerpt) -> int:
            meta = getattr(ex, "metadata", {}) or {}
            ex_city = _norm_city(str(meta.get("city", "") or ""))
            if not ex_city:
                return 0
            if ex_city == target:
                return 2
            if target in ex_city or ex_city in target:
                return 1
            return 0

        relaxed_sorted = sorted(
            relaxed,
            key=lambda ex: (city_match_score(ex), float(getattr(ex, "score", 0) or 0)),
            reverse=True,
        )
        return relaxed_sorted[:top_k]
