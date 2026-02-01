"""
Stage 3: RAG decision layer for traffic incident pipeline.
Modular document ingestion, retrieval, and decision engine.

Local dev: LlamaIndex in-memory + OpenAI embedding + local ./data/ folder.
Swap to OpenSearch + S3 for AWS. Config via env: RAG_STORE, RAG_PROVIDER.
"""

import os
from pathlib import Path

from .decision_engine import DecisionEngine, DecisionInput, DecisionOutput
from .ingestion import DocumentIngestionPipeline
from .providers import LocalDataProvider, MockPolicyProvider, PolicyProvider
from .retriever import PolicyRetriever
from .schemas import PolicyDocument, RetrievedExcerpt
from .vector_store import InMemoryVectorStore

__all__ = [
    "PolicyProvider",
    "MockPolicyProvider",
    "LocalDataProvider",
    "PolicyDocument",
    "RetrievedExcerpt",
    "InMemoryVectorStore",
    "DocumentIngestionPipeline",
    "PolicyRetriever",
    "DecisionEngine",
    "DecisionInput",
    "DecisionOutput",
    "create_rag_pipeline",
]


def create_rag_pipeline(
    provider: PolicyProvider | None = None,
    *,
    store_type: str | None = None,
    provider_type: str | None = None,
    data_dir: str | Path | None = None,
) -> tuple[DocumentIngestionPipeline, PolicyRetriever, DecisionEngine]:
    """
    Factory: create ingestion pipeline, retriever, and decision engine.
    Ingestion runs once (lazy init). Default: LlamaIndex + local data when available.

    Config (env overrides):
        RAG_STORE: llamaindex | memory (llamaindex if OPENAI_API_KEY set)
        RAG_PROVIDER: local | mock (local if ./data/ exists)
        RAG_DATA_DIR: path to policy docs (default: rag/data/)
    """
    store_type = store_type or os.environ.get("RAG_STORE", "")
    provider_type = provider_type or os.environ.get("RAG_PROVIDER", "")
    data_dir = data_dir or os.environ.get("RAG_DATA_DIR")
    if data_dir:
        data_dir = Path(data_dir)

    # Resolve provider
    if provider is not None:
        pass
    elif provider_type == "local" or (not provider_type and _default_use_local(data_dir)):
        provider = LocalDataProvider(data_dir=data_dir)
    else:
        provider = MockPolicyProvider()

    # Resolve store: LlamaIndex when OpenAI key + LlamaIndex available, else memory
    if store_type == "memory":
        store = InMemoryVectorStore()
    elif store_type == "llamaindex" or (not store_type and _default_use_llamaindex()):
        try:
            from .llama_store import LlamaIndexVectorStore
            store = LlamaIndexVectorStore()
        except ImportError:
            store = InMemoryVectorStore()
    else:
        store = InMemoryVectorStore()

    pipeline = DocumentIngestionPipeline(provider=provider, vector_store=store)
    pipeline.run()
    retriever = PolicyRetriever(vector_store=store)
    engine = DecisionEngine(retriever=retriever, top_k=5)
    return pipeline, retriever, engine


def _default_use_local(data_dir: Path | None) -> bool:
    default_dir = Path(__file__).parent / "data"
    path = data_dir or default_dir
    return path.exists() and any(path.iterdir())


def _default_use_llamaindex() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))
