"""
RAG schemas: policy documents, retrieved excerpts.
OpenSearch-compatible metadata structure.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RAGBase(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

class DecisionInput(RAGBase):
    """Input to the decision engine."""

    model_config = ConfigDict(extra="forbid")

    event_type_candidates: list[str] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)
    city: str = "Providence"


class SupportingExcerpt(RAGBase):
    """A policy excerpt supporting the decision."""

    model_config = ConfigDict(extra="forbid")

    text: str
    document_id: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionOutput(RAGBase):
    """Output from the decision engine."""

    model_config = ConfigDict(extra="forbid")

    decision: dict[str, Any] = Field(default_factory=dict)
    explanation: str = ""
    supporting_excerpts: list[SupportingExcerpt] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

class PolicyDocument(RAGBase):
    """A policy document chunk. Compatible with OpenSearch document structure."""

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


class RetrievedExcerpt(RAGBase):
    """A policy excerpt retrieved for a query, with metadata."""
    
    document_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
