# backend/app/rag/opensearch_store.py

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

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
        endpoint: str | None = None,
        index_name: str | None = None,
        region: str | None = None,
        embed_fn=None,
    ):
        self._endpoint = (endpoint or os.environ.get("OPENSEARCH_ENDPOINT") or "http://localhost:9200").rstrip("/")
        self._index_name = index_name or os.environ.get("OPENSEARCH_INDEX") or "policy-docs"
        self._region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
        self._embed_fn = embed_fn or simple_embed
        self._client = None

    def _want_aws_sigv4(self) -> bool:
        # Allow forcing either way.
        # - OPENSEARCH_USE_AWS_SIGV4=1 forces SigV4 even if endpoint not amazonaws.
        # - OPENSEARCH_USE_AWS_SIGV4=0 disables SigV4 even if endpoint is amazonaws.
        flag = os.environ.get("OPENSEARCH_USE_AWS_SIGV4")
        if flag is not None:
            return flag.strip() == "1"
        return "amazonaws.com" in self._endpoint

    def _get_client(self):
        if self._client is not None:
            return self._client

        from opensearchpy import OpenSearch

        parsed = urlparse(self._endpoint)
        scheme = parsed.scheme or "http"
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if scheme == "https" else 9200)
        use_ssl = scheme == "https"

        if self._want_aws_sigv4():
            try:
                from requests_aws4auth import AWS4Auth
                import boto3
            except ImportError as e:
                raise ImportError(
                    "AWS OpenSearch requires: pip install opensearch-py requests-aws4auth boto3"
                ) from e

            session = boto3.Session()
            credentials = session.get_credentials()
            if credentials is None:
                # This is the bug you hit. Make it explicit and actionable.
                raise RuntimeError(
                    "OPENSEARCH endpoint looks like AWS, but no AWS credentials were found.\n"
                    "Fix one of:\n"
                    "  - Export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (and AWS_SESSION_TOKEN if needed)\n"
                    "  - Run `aws configure` (or use an IAM role on EC2/ECS)\n"
                    "  - For local dev against a non-AWS cluster, set OPENSEARCH_USE_AWS_SIGV4=0\n"
                    "  - Or set OPENSEARCH_ENDPOINT=http://localhost:9200\n"
                )

            frozen = credentials.get_frozen_credentials()
            awsauth = AWS4Auth(
                frozen.access_key,
                frozen.secret_key,
                self._region,
                "es",
                session_token=frozen.token,
            )

            self._client = OpenSearch(
                hosts=[{"host": host, "port": port}],
                http_auth=awsauth,
                use_ssl=use_ssl,
                verify_certs=True,
                ssl_assert_hostname=True,
                ssl_show_warn=False,
            )
            return self._client

        # Non-AWS basic auth
        user = os.environ.get("OPENSEARCH_USER", "admin")
        pwd = os.environ.get("OPENSEARCH_PASSWORD", "admin")

        self._client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=(user, pwd),
            use_ssl=use_ssl,
            verify_certs=False,  # local/dev convenience; turn on for prod
            ssl_show_warn=False,
        )
        return self._client

    def _ensure_index(self) -> None:
        client = self._get_client()
        if client.indices.exists(index=self._index_name):
            return

        hnsw_m = int(os.environ.get("RAG_HNSW_M", "16"))
        hnsw_ef = int(os.environ.get("RAG_HNSW_EF_CONSTRUCTION", "128"))

        # IMPORTANT: your previous mapping had metadata enabled=False, so filters could never work.
        # This mapping indexes metadata.city and metadata.doc_type as keywords, but keeps the rest non-indexed.
        body = {
            "settings": {
                "index": {
                    "knn": True,
                }
            },
            "mappings": {
                "dynamic": True,
                "properties": {
                    self.TEXT_FIELD: {"type": "text"},
                    self.METADATA_FIELD: {
                        "type": "object",
                        "dynamic": True,
                        "properties": {
                            "city": {"type": "keyword"},
                            "doc_type": {"type": "keyword"},
                        },
                    },
                    self.VECTOR_FIELD: {
                        "type": "knn_vector",
                        "dimension": DEFAULT_EMBEDDING_DIM,
                        "method": {
                            "name": "hnsw",
                            "engine": "nmslib",
                            "space_type": "cosinesimil",
                            "parameters": {"m": hnsw_m, "ef_construction": hnsw_ef},
                        },
                    },
                },
            },
        }

        client.indices.create(index=self._index_name, body=body)

    def ingest(self, documents: list[PolicyDocument]) -> None:
        """Embed documents and bulk index into OpenSearch."""
        if not documents:
            return

        self._ensure_index()
        client = self._get_client()

        actions: list[dict[str, Any]] = []
        for doc in documents:
            embedding = doc.embedding if doc.embedding is not None else self._embed_fn(doc.text)

            actions.append(
                {
                    "_index": self._index_name,
                    "_id": doc.id,
                    "_source": {
                        self.TEXT_FIELD: doc.text,
                        self.METADATA_FIELD: doc.metadata or {},
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
        client = self._get_client()
        if not client.indices.exists(index=self._index_name):
            return []

        query_vector = self._embed_fn(query)

        filter_clauses: list[dict[str, Any]] = []
        if filters:
            # Only reliably filter on indexed fields:
            # metadata.city, metadata.doc_type (per mapping above)
            if "city" in filters and filters["city"]:
                filter_clauses.append({"term": {f"{self.METADATA_FIELD}.city": filters["city"]}})
            if "doc_type" in filters and filters["doc_type"]:
                filter_clauses.append({"term": {f"{self.METADATA_FIELD}.doc_type": filters["doc_type"]}})

        # OpenSearch supports knn inside a bool query in common modern versions.
        num_candidates = int(os.environ.get("RAG_NUM_CANDIDATES", str(max(top_k * 4, 20))))
        body: dict[str, Any] = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                self.VECTOR_FIELD: {
                                    "vector": query_vector,
                                    "k": top_k,
                                    "num_candidates": num_candidates,
                                }
                            }
                        }
                    ],
                    "filter": filter_clauses,
                }
            },
        }

        resp = client.search(index=self._index_name, body=body)
        hits = resp.get("hits", {}).get("hits", []) or []

        excerpts: list[RetrievedExcerpt] = []
        for hit in hits:
            src = hit.get("_source", {}) or {}
            score = hit.get("_score", 0.0) or 0.0
            excerpts.append(
                RetrievedExcerpt(
                    document_id=hit.get("_id", "") or "",
                    text=src.get(self.TEXT_FIELD, "") or "",
                    score=float(score),
                    metadata=src.get(self.METADATA_FIELD, {}) or {},
                )
            )
        return excerpts
