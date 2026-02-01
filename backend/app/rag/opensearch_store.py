#TODO: fill in correct OpenSearch endpoint

"""
OpenSearch vector store for AWS.
Dummy implementation: placeholder config, no-op ingest, empty search until wired.
"""

from typing import Any

from .schemas import PolicyDocument, RetrievedExcerpt


class OpenSearchVectorStore:
    """
    Vector store using AWS OpenSearch.
    Dummy implementation: placeholder endpoint, returns empty until wired to opensearch-py.
    """

    def __init__(
        self,
        *,
        endpoint: str = "https://placeholder.opensearch.region.es.amazonaws.com",
        index_name: str = "policy-docs",
        region: str = "us-east-1",
    ):
        self._endpoint = endpoint
        self._index_name = index_name
        self._region = region

    def ingest(self, documents: list[PolicyDocument]) -> None:
        """TODO: opensearch-py bulk index. No-op for now."""
        # Dummy: would use opensearch-py bulk() with embeddings
        pass

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedExcerpt]:
        """TODO: opensearch-py k-NN search. Returns empty for now."""
        # Dummy: would embed query, call k-NN search, map hits to RetrievedExcerpt
        return []
