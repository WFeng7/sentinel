"""
Vector store with OpenSearch-compatible document structure.
In-memory implementation for development; swap to OpenSearch client for production.
"""

from typing import Any

from .embeddings import cosine_similarity, simple_embed
from .schemas import PolicyDocument


class InMemoryVectorStore:
    """
    In-memory vector store. Documents follow OpenSearch k-NN index structure:
    { id, text, embedding, metadata: { city, doc_type, ... } }
    """

    def __init__(self, embed_fn=None):
        self._embed_fn = embed_fn or simple_embed
        self._docs: list[PolicyDocument] = []
        self._indexed = False

    def ingest(self, documents: list[PolicyDocument]) -> None:
        """Ingest documents. Assigns embeddings if missing."""
        for doc in documents:
            if doc.embedding is None:
                doc.embedding = self._embed_fn(doc.text)
        self._docs = list(documents)
        self._indexed = True

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[PolicyDocument, float]]:
        """
        Search by vector similarity with optional metadata filters.
        filters: {"city": "Providence", "doc_type": "incident_response"}
        Returns [(doc, score), ...] sorted by score descending.
        """
        candidates = self._docs
        if filters:
            candidates = [
                d for d in candidates
                if all(d.metadata.get(k) == v for k, v in filters.items())
            ]
        scored = [(d, cosine_similarity(query_embedding, d.embedding or [])) for d in candidates]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def get_all(self, filters: dict[str, Any] | None = None) -> list[PolicyDocument]:
        """Return all documents, optionally filtered by metadata."""
        docs = self._docs
        if filters:
            docs = [d for d in docs if all(d.metadata.get(k) == v for k, v in filters.items())]
        return docs
