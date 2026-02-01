"""
Policy retriever with metadata filtering (city, doc_type).
Works with any VectorStore implementation (InMemory, OpenSearch).
"""

from typing import Any

from .schemas import RetrievedExcerpt
from .opensearch_store import OpenSearchVectorStore


class PolicyRetriever:
    """Retrieves relevant policy excerpts by semantic similarity and metadata filters."""

    def __init__(self, vector_store: OpenSearchVectorStore):
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

        return self._store.search(
            query=query,
            top_k=top_k,
            filters=filters if filters else None,
        )
