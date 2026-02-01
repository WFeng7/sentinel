"""
Modular document ingestion pipeline.
Fetches from PolicyProvider, ingests into vector store.
"""

from .providers import PolicyProvider
from .schemas import PolicyDocument


class DocumentIngestionPipeline:
    """Ingests documents from a PolicyProvider into a vector store."""

    def __init__(self, provider: PolicyProvider, vector_store):
        self._provider = provider
        self._store = vector_store

    def run(self) -> int:
        """Fetch documents from provider and ingest. Returns count ingested."""
        documents: list[PolicyDocument] = self._provider.fetch_documents()
        self._store.ingest(documents)
        return len(documents)
