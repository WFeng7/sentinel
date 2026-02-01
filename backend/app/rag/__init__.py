"""
RAG decision layer for traffic incident pipeline.
S3 + OpenSearch (dummy placeholders). InMemory + Mock for dev.
"""

import os

from .decision_engine import DecisionEngine, DecisionInput, DecisionOutput
from .ingestion import DocumentIngestionPipeline
from .providers import MockPolicyProvider, S3PolicyProvider
from .retriever import PolicyRetriever
from .schemas import PolicyDocument, RetrievedExcerpt

__all__ = [
    "S3PolicyProvider",
    "MockPolicyProvider",
    "PolicyDocument",
    "RetrievedExcerpt",
    "DocumentIngestionPipeline",
    "PolicyRetriever",
    "DecisionEngine",
    "DecisionInput",
    "DecisionOutput",
    "create_rag_pipeline",
]


def create_rag_pipeline(
    provider: S3PolicyProvider | MockPolicyProvider | None = None,
    *,
    store_type: str | None = None,
    provider_type: str | None = None,
) -> tuple[DocumentIngestionPipeline, PolicyRetriever, DecisionEngine]:
    """
    Factory: create ingestion pipeline, retriever, and decision engine.
    Config (env): RAG_STORE=opensearch|memory, RAG_PROVIDER=s3|mock
    """
    store_type = store_type or os.environ.get("RAG_STORE", "memory")
    provider_type = provider_type or os.environ.get("RAG_PROVIDER", "mock")

    if provider is not None:
        pass
    elif provider_type == "s3":
        provider = S3PolicyProvider(
            bucket=os.environ.get("RAG_S3_BUCKET", "sentinel-policy-docs"),
            prefix=os.environ.get("RAG_S3_PREFIX", "policy/"),
            region=os.environ.get("AWS_REGION", "us-east-1"),
        )
    else:
        provider = MockPolicyProvider()
    from .opensearch_store import OpenSearchVectorStore
    store = OpenSearchVectorStore(
        endpoint=os.environ.get("OPENSEARCH_ENDPOINT", "https://placeholder.opensearch.region.es.amazonaws.com"),
        index_name=os.environ.get("OPENSEARCH_INDEX", "policy-docs"),
        region=os.environ.get("AWS_REGION", "us-east-1"),
    )

    pipeline = DocumentIngestionPipeline(provider=provider, vector_store=store)
    pipeline.run()
    retriever = PolicyRetriever(vector_store=store)
    engine = DecisionEngine(retriever=retriever, top_k=5)
    return pipeline, retriever, engine
