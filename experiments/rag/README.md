# RAG Decision Layer (Stage 3)

Modular document ingestion, retrieval, and decision engine for the traffic incident pipeline.

## Local Dev (MVP)

- **Store:** LlamaIndex in-memory + OpenAI embedding
- **Provider:** Local `./rag/data/` folder (PDFs, txt, md)
- **Config:** `RAG_STORE=llamaindex|memory`, `RAG_PROVIDER=local|mock`, `RAG_DATA_DIR=<path>`
- **Fallback:** No OpenAI key or LlamaIndex → InMemoryVectorStore + MockPolicyProvider
- **Ingestion:** Once per process (lazy init on first `/rag/decide`)

## AWS Swap Points (future)

| Component | Local | AWS |
|-----------|-------|-----|
| Provider | LocalDataProvider (./data/) | S3PolicyProvider |
| Vector store | LlamaIndexVectorStore (in-memory) | OpenSearchVectorStore |
| Embedding | OpenAI | OpenAI or Bedrock Titan |
| Queue | — | SQS / EventBridge (ingestion triggers) |

## Design

### PolicyProvider (document source)

- **Interface:** `PolicyProvider.fetch_documents() -> list[PolicyDocument]`
- **MockPolicyProvider:** Hardcoded sample Providence policy (no deps)
- **LocalDataProvider:** Loads from local `./data/` (PDF, txt, md) via LlamaIndex
- **S3PolicyProvider:** (future) Fetch from S3; swap in `create_rag_pipeline(provider=S3PolicyProvider(...))`

### Vector store (OpenSearch-compatible)

- Document structure: `{ id, text, embedding, metadata: { city, doc_type, ... } }`
- **LlamaIndexVectorStore:** Local MVP; OpenAI embedding, in-memory index
- **InMemoryVectorStore:** Fallback; simple_embed, no external deps
- **OpenSearchVectorStore:** (future) AWS OpenSearch for production

### Retriever

- `PolicyRetriever.retrieve(query, top_k, city=None, doc_type=None)`
- Metadata filtering by `city` and `doc_type` for RAG queries

### Decision engine

- **LLM-based** when `OPENAI_API_KEY` is set: GPT-4o-mini grounds decisions in retrieved policy excerpts
- **Rule-based fallback** when no API key or LLM fails

**Input:**

- `event_type_candidates`: e.g. `["lane_blockage", "multi_vehicle"]`
- `signals`: e.g. `["lane_blocked", "multi_vehicle"]`
- `city`: e.g. `"Providence"`

**Output:**

- `decision`: structured JSON (event_type, recommended_actions, severity, etc.)
- `explanation`: human-readable summary
- `supporting_excerpts`: list of policy excerpts with metadata

## API

```
POST /rag/decide
{
  "event_type_candidates": ["lane_blockage"],
  "signals": ["lane_blocked", "multi_vehicle"],
  "city": "Providence"
}
```

## Usage

```python
from app.rag import create_rag_pipeline, DecisionInput

# Default: LlamaIndex + local ./rag/data/ when OPENAI_API_KEY set and data exists
_, _, engine = create_rag_pipeline()

# Explicit config
_, _, engine = create_rag_pipeline(store_type="llamaindex", provider_type="local")
inp = DecisionInput(
    event_type_candidates=["lane_blockage"],
    signals=["lane_blocked"],
    city="Providence",
)
out = engine.decide(inp)
print(out.decision, out.explanation, out.supporting_excerpts)
```

To switch to S3 (when implemented):

```python
from app.rag import create_rag_pipeline
from app.rag.providers import S3PolicyProvider

_, _, engine = create_rag_pipeline(provider=S3PolicyProvider(bucket="..."))
```
