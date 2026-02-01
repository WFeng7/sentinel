"""
Modular document ingestion pipeline.
Fetches from PolicyProvider, ingests into vector store.
"""

import os

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
        documents = self._chunk_documents(documents)
        if documents:
            print("[rag] Ingesting policy documents:")
            for doc in documents:
                preview = (doc.text or "").strip().replace("\n", " ")
                if len(preview) > 200:
                    preview = f"{preview[:200]}…"
                print(f"[rag] - id={doc.id} metadata={doc.metadata} text='{preview}'")
        self._store.ingest(documents)
        return len(documents)

    def _chunk_documents(self, documents: list[PolicyDocument]) -> list[PolicyDocument]:
        chunk_size = int(os.environ.get("RAG_CHUNK_SIZE", "1200"))
        overlap = int(os.environ.get("RAG_CHUNK_OVERLAP", "200"))
        if chunk_size <= 0:
            return documents

        chunked: list[PolicyDocument] = []
        for doc in documents:
            text = (doc.text or "").strip()
            if len(text) <= chunk_size:
                chunked.append(doc)
                continue

            start = 0
            idx = 0
            while start < len(text):
                end = min(len(text), start + chunk_size)
                chunk_text = text[start:end].strip()
                if not chunk_text:
                    break
                chunk_id = f"{doc.id}__chunk_{idx:04d}"
                metadata = {**(doc.metadata or {})}
                metadata["parent_id"] = doc.id
                metadata["chunk_index"] = idx
                metadata["chunk_start"] = start
                chunked.append(
                    PolicyDocument(
                        id=chunk_id,
                        text=chunk_text,
                        metadata=metadata,
                    )
                )
                if end == len(text):
                    break
                start = max(0, end - overlap)
                idx += 1
        return chunked
