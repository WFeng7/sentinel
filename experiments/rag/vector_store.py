"""
Vector store abstraction. OpenSearch-compatible document structure.
- InMemoryVectorStore: dev fallback (simple_embed)
- LlamaIndexVectorStore: local MVP (OpenAI embedding, in-memory)
- OpenSearchVectorStore: future AWS production
"""

from typing import Any, Protocol

from .schemas import PolicyDocument, RetrievedExcerpt


class VectorStoreProtocol(Protocol):
    """Protocol for vector stores. Swap implementations for local vs AWS."""

    def ingest(self, documents: list[PolicyDocument]) -> None:
        """Ingest documents into the store."""
        ...

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedExcerpt]:
        """Search by semantic similarity. Returns excerpts sorted by score descending."""
        ...


class InMemoryVectorStore:
    """
    In-memory vector store. Uses simple_embed for dev/fallback when no OpenAI key.
    Documents follow OpenSearch k-NN structure: { id, text, embedding, metadata }
    """

    def __init__(self, embed_fn=None):
        from utils import simple_embed
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
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedExcerpt]:
        """Search by vector similarity. Query is embedded internally."""
        query_embedding = self._embed_fn(query)
        candidates = self._docs
        if filters:
            candidates = [
                d for d in candidates
                if all(d.metadata.get(k) == v for k, v in filters.items())
            ]
        from utils import cosine_similarity
        scored = [
            (d, cosine_similarity(query_embedding, d.embedding or []))
            for d in candidates
        ]
        scored.sort(key=lambda x: -x[1])
        return [
            RetrievedExcerpt(
                document_id=doc.id,
                text=doc.text,
                score=float(score),
                metadata=dict(doc.metadata),
            )
            for doc, score in scored[:top_k]
        ]

    def get_all(self, filters: dict[str, Any] | None = None) -> list[PolicyDocument]:
        """Return all documents, optionally filtered by metadata."""
        docs = self._docs
        if filters:
            docs = [d for d in docs if all(d.metadata.get(k) == v for k, v in filters.items())]
        return docs
