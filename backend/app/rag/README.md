# RAG Decision Layer (Stage 3)

Modular document ingestion, retrieval, and decision engine.

## Config (env)

| Env | Values | Default |
|-----|--------|---------|
| RAG_STORE | opensearch \| memory | memory |
| RAG_PROVIDER | s3 \| mock | mock |
| RAG_MODE | llm \| retrieve | llm |
| RAG_CHUNK_SIZE | integer | 1200 |
| RAG_CHUNK_OVERLAP | integer | 200 |
| RAG_MAX_EXCERPTS | integer | 5 |
| RAG_MAX_EXCERPT_CHARS | integer | 1200 |
| RAG_HNSW_M | integer | 16 |
| RAG_HNSW_EF_CONSTRUCTION | integer | 128 |
| RAG_NUM_CANDIDATES | integer | max(top_k*4, 20) |
| RAG_S3_BUCKET | S3 bucket name | sentinel-policy-docs |
| RAG_S3_PREFIX | S3 prefix | policy/ |
| OPENSEARCH_ENDPOINT | OpenSearch URL | (placeholder) |
| OPENSEARCH_INDEX | Index name | policy-docs |
| AWS_REGION | Region | us-east-1 |

## Components

- **S3PolicyProvider:** Fetch docs from S3 (dummy: returns [] until wired)
- **OpenSearchVectorStore:** Vector search via OpenSearch (dummy: ingest no-op, search returns [])
- **InMemoryVectorStore:** Dev fallback with simple_embed
- **MockPolicyProvider:** Minimal hardcoded docs for testing

## API

```
POST /rag/decide
{ "event_type_candidates": ["lane_blockage"], "signals": ["lane_blocked"], "city": "Providence" }
```
