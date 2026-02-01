"""
Stage 3: RAG decision layer for traffic incident pipeline.
Modular document ingestion, retrieval, and decision engine.
"""

from .decision_engine import DecisionEngine, DecisionInput, DecisionOutput
from .ingestion import DocumentIngestionPipeline
from .providers import MockPolicyProvider, PolicyProvider
from .retriever import PolicyRetriever
from .schemas import PolicyDocument, RetrievedExcerpt
from .vector_store import InMemoryVectorStore

__all__ = [
    "PolicyProvider",
    "MockPolicyProvider",
    "PolicyDocument",
    "RetrievedExcerpt",
    "InMemoryVectorStore",
    "DocumentIngestionPipeline",
    "PolicyRetriever",
    "DecisionEngine",
    "DecisionInput",
    "DecisionOutput",
]


def create_rag_pipeline(
    provider: PolicyProvider | None = None,
) -> tuple[DocumentIngestionPipeline, PolicyRetriever, DecisionEngine]:
    """
    Factory: create ingestion pipeline, retriever, and decision engine.
    Swap provider (MockPolicyProvider -> S3PolicyProvider) here; no other refactor needed.
    """
    provider = provider or MockPolicyProvider()
    store = InMemoryVectorStore()
    pipeline = DocumentIngestionPipeline(provider=provider, vector_store=store)
    pipeline.run()
    retriever = PolicyRetriever(vector_store=store)
    engine = DecisionEngine(retriever=retriever, top_k=5)
    return pipeline, retriever, engine
