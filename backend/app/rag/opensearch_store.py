#TODO: fill in correct OpenSearch endpoint and index name

from typing import Any

from app.utils import DEFAULT_EMBEDDING_DIM, simple_embed

from .schemas import PolicyDocument, RetrievedExcerpt


class OpenSearchVectorStore:
    """Vector store using OpenSearch k-NN. Supports AWS (SigV4) or standard auth."""

    VECTOR_FIELD = "embedding"
    TEXT_FIELD = "text"
    METADATA_FIELD = "metadata"

    def __init__(
        self,
        *,
        endpoint: str = "https://opensearch.us-east-1.amazonaws.com",
        index_name: str = "policy-docs",
        region: str = "us-east-1",
        embed_fn=None,
    ):
        self._endpoint = endpoint.rstrip("/")
        self._index_name = index_name
        self._region = region
        self._embed_fn = embed_fn or simple_embed
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        if "amazonaws.com" in self._endpoint:
            try:
                from opensearchpy import OpenSearch
                from requests_aws4auth import AWS4Auth
                import boto3

                credentials = boto3.Session().get_credentials()
                awsauth = AWS4Auth(
                    credentials.access_key,
                    credentials.secret_key,
                    self._region,
                    "es",
                    session_token=credentials.token,
                )
                host = self._endpoint.replace("https://", "").replace("http://", "")
                self._client = OpenSearch(
                    hosts=[{"host": host.split("/")[0], "port": 443, "use_ssl": True}],
                    http_auth=awsauth,
                    use_ssl=True,
                    verify_certs=True,
                )
            except ImportError as e:
                raise ImportError(
                    "AWS OpenSearch requires: pip install opensearch-py requests-aws4auth boto3"
                ) from e
        else:
            from opensearchpy import OpenSearch
            import os

            from urllib.parse import urlparse

            parsed = urlparse(self._endpoint)
            port = parsed.port or (443 if parsed.scheme == "https" else 9200)
            self._client = OpenSearch(
                hosts=[{"host": parsed.hostname or "localhost", "port": port, "use_ssl": parsed.scheme == "https"}],
                http_auth=(
                    os.environ.get("OPENSEARCH_USER", "admin"),
                    os.environ.get("OPENSEARCH_PASSWORD", "admin"),
                ),
                use_ssl=parsed.scheme == "https",
                verify_certs=False,
            )
        return self._client

    def _ensure_index(self) -> None:
        client = self._get_client()
        if client.indices.exists(index=self._index_name):
            return
        client.indices.create(
            index=self._index_name,
            body={
                "settings": {"index": {"knn": True}},
                "mappings": {
                    "properties": {
                        self.TEXT_FIELD: {"type": "text"},
                        self.METADATA_FIELD: {"type": "object", "enabled": False},
                        self.VECTOR_FIELD: {
                            "type": "knn_vector",
                            "dimension": DEFAULT_EMBEDDING_DIM,
                        },
                    }
                },
            },
        )

    def ingest(self, documents: list[PolicyDocument]) -> None:
        """Embed documents and bulk index into OpenSearch."""
        if not documents:
            return

        self._ensure_index()
        client = self._get_client()

        actions = []
        for doc in documents:
            embedding = doc.embedding
            if embedding is None:
                embedding = self._embed_fn(doc.text)
            actions.append(
                {
                    "_index": self._index_name,
                    "_id": doc.id,
                    "_source": {
                        self.TEXT_FIELD: doc.text,
                        self.METADATA_FIELD: doc.metadata,
                        self.VECTOR_FIELD: embedding,
                    },
                }
            )

        from opensearchpy import helpers

        helpers.bulk(client, actions, raise_on_error=False)
        client.indices.refresh(index=self._index_name)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedExcerpt]:
        """Embed query, run k-NN search, return RetrievedExcerpts."""
        if not self._get_client().indices.exists(index=self._index_name):
            return []

        query_vector = self._embed_fn(query)
        knn_query: dict[str, Any] = {"vector": query_vector, "k": top_k}
        # Metadata filter omitted: stored as object, not indexed. Add keyword subfields if needed.

        body = {"query": {"knn": {self.VECTOR_FIELD: knn_query}}}
        resp = self._get_client().search(index=self._index_name, body=body, size=top_k)

        excerpts = []
        for hit in resp.get("hits", {}).get("hits", []):
            src = hit.get("_source", {})
            score = hit.get("_score", 0.0)
            excerpts.append(
                RetrievedExcerpt(
                    document_id=hit.get("_id", ""),
                    text=src.get(self.TEXT_FIELD, ""),
                    score=float(score),
                    metadata=src.get(self.METADATA_FIELD, {}),
                )
            )
        return excerpts
