"""
RAG schemas: policy documents, retrieved excerpts, decision output.
OpenSearch-compatible metadata structure.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PolicyDocument:
    """A policy document chunk. Compatible with OpenSearch document structure."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def to_opensearch_doc(self) -> dict[str, Any]:
        """OpenSearch-compatible document structure."""
        doc: dict[str, Any] = {
            "id": self.id,
            "text": self.text,
            "metadata": {**self.metadata},
        }
        if self.embedding is not None:
            doc["embedding"] = self.embedding
        return doc

    @classmethod
    def from_opensearch_doc(cls, doc: dict[str, Any]) -> "PolicyDocument":
        return cls(
            id=doc["id"],
            text=doc["text"],
            metadata=doc.get("metadata", {}),
            embedding=doc.get("embedding"),
        )


@dataclass
class RetrievedExcerpt:
    """A policy excerpt retrieved for a query, with metadata."""

    document_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
