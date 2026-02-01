"""
Modular document ingestion pipeline.
Fetches from provider (S3PolicyProvider, MockPolicyProvider, etc.), ingests into vector store.
"""

from typing import Protocol

from .schemas import PolicyDocument


class _ProviderProtocol(Protocol):
    def fetch_documents(self) -> list[PolicyDocument]: ...


class DocumentIngestionPipeline:
    """Ingests documents from a provider into a vector store."""

    def __init__(self, provider: _ProviderProtocol, vector_store):
        self._provider = provider
        self._store = vector_store

    def run(self) -> int:
        """Fetch documents from provider and ingest. Returns count ingested."""
        documents: list[PolicyDocument] = self._provider.fetch_documents()
        self._store.ingest(documents)
        return len(documents)
