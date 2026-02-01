"""
LlamaIndex-backed vector store. Local MVP with OpenAI embeddings, in-memory index.

AWS: Add app/rag/opensearch_store.py with OpenSearchVectorStore implementing
VectorStoreProtocol (ingest, search). Wire via create_rag_pipeline(store_type="opensearch").
"""

from typing import Any

from .schemas import PolicyDocument, RetrievedExcerpt


class LlamaIndexVectorStore:
    """
    Vector store using LlamaIndex VectorStoreIndex + OpenAI embeddings.
    In-memory by default. Use for local dev / demo.
    """

    def __init__(self, embed_model=None):
        self._embed_model = embed_model
        self._index = None
        self._doc_count = 0

    def ingest(self, documents: list[PolicyDocument]) -> None:
        """Ingest documents. Builds LlamaIndex in-memory index."""
        try:
            from llama_index.core import Document, VectorStoreIndex
            from llama_index.embeddings.openai import OpenAIEmbedding
        except ImportError as e:
            raise ImportError(
                "LlamaIndex required for LlamaIndexVectorStore. "
                "pip install llama-index llama-index-embeddings-openai"
            ) from e

        embed_model = self._embed_model or OpenAIEmbedding()
        llama_docs = [
            Document(
                text=d.text,
                metadata={**d.metadata, "id": d.id},
                id_=d.id,
            )
            for d in documents
        ]
        self._index = VectorStoreIndex.from_documents(
            llama_docs,
            embed_model=embed_model,
            show_progress=False,
        )
        self._doc_count = len(documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedExcerpt]:
        """Search by semantic similarity. Metadata filters ignored for MVP."""
        if self._index is None:
            return []

        retriever = self._index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)

        excerpts = []
        for i, item in enumerate(nodes):
            node = getattr(item, "node", item)
            text = getattr(node, "text", str(node))
            metadata = getattr(node, "metadata", None) or {}
            score = getattr(item, "score", 0.0) or 0.0
            doc_id = metadata.get("id", getattr(node, "node_id", str(i)))
            excerpts.append(
                RetrievedExcerpt(
                    document_id=str(doc_id),
                    text=text,
                    score=float(score),
                    metadata=dict(metadata),
                )
            )
        return excerpts
