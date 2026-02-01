# RAG Decision Layer (Stage 3)

Modular document ingestion, retrieval, and decision engine for the traffic incident pipeline.

## Design

### PolicyProvider (document source)

- **Interface:** `PolicyProvider.fetch_documents() -> list[PolicyDocument]`
- **MockPolicyProvider:** Returns sample Providence policy text (no external dependencies)
- **S3PolicyProvider:** (future) Fetch from S3; swap in `create_rag_pipeline(provider=S3PolicyProvider(...))` with no other refactor

### Vector store (OpenSearch-compatible)

- Document structure: `{ id, text, embedding, metadata: { city, doc_type, ... } }`
- **InMemoryVectorStore:** For development; swap to OpenSearch client for production
- Supports metadata filtering (city, doc_type) in search

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

_, _, engine = create_rag_pipeline()  # Uses MockPolicyProvider by default
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
