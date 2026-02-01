"""
RAG schemas: policy documents, retrieved excerpts.
OpenSearch-compatible metadata structure.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PolicyDocument(BaseModel):
    """A policy document chunk. Compatible with OpenSearch document structure."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
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


class RetrievedExcerpt(BaseModel):
    """A policy excerpt retrieved for a query, with metadata."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
