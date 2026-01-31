"""
Policy retriever with metadata filtering (city, doc_type).
Operates with mocked or real documents.
"""

from .embeddings import simple_embed
from .schemas import RetrievedExcerpt
from .vector_store import InMemoryVectorStore


class PolicyRetriever:
    """Retrieves relevant policy excerpts by semantic similarity and metadata filters."""

    def __init__(self, vector_store: InMemoryVectorStore, embed_fn=None):
        self._store = vector_store
        self._embed_fn = embed_fn or simple_embed

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
        filters: dict[str, str] = {}
        if city:
            filters["city"] = city
        if doc_type:
            filters["doc_type"] = doc_type

        query_embedding = self._embed_fn(query)
        results = self._store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters if filters else None,
        )
        return [
            RetrievedExcerpt(
                document_id=doc.id,
                text=doc.text,
                score=float(score),
                metadata=dict(doc.metadata),
            )
            for doc, score in results
        ]
