from __future__ import annotations

from typing import Any

from app.utils import cosine_similarity, simple_embed
from .schemas import PolicyDocument, RetrievedExcerpt


class InMemoryVectorStore:
    """Simple in-memory vector store for dev/testing."""

    def __init__(self, *, embed_fn=None):
        self._embed_fn = embed_fn or simple_embed
        self._docs: list[PolicyDocument] = []

    def ingest(self, documents: list[PolicyDocument]) -> None:
        if not documents:
            return
        for doc in documents:
            if doc.embedding is None:
                doc.embedding = self._embed_fn(doc.text)
            self._docs.append(doc)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedExcerpt]:
        if not self._docs:
            return []

        query_vec = self._embed_fn(query)
        filtered: list[PolicyDocument] = []
        if filters:
            for doc in self._docs:
                meta = doc.metadata or {}
                if "city" in filters and filters["city"] and meta.get("city") != filters["city"]:
                    continue
                if "doc_type" in filters and filters["doc_type"] and meta.get("doc_type") != filters["doc_type"]:
                    continue
                filtered.append(doc)
        else:
            filtered = self._docs

        scored: list[tuple[float, PolicyDocument]] = []
        for doc in filtered:
            score = cosine_similarity(query_vec, doc.embedding or [])
            scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)

        excerpts: list[RetrievedExcerpt] = []
        for score, doc in scored[:top_k]:
            excerpts.append(
                RetrievedExcerpt(
                    text=doc.text,
                    score=score,
                    metadata=doc.metadata or {},
                    document_id=doc.id,
                )
            )
        return excerpts
