"""
AWS S3-based RAG implementation.
Minimal modification to existing RAG to use AWS S3 for document storage.
"""

import json
import numpy as np
import faiss
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

from .s3_manager import s3_manager

class S3PolicyRetriever:
    """Policy retriever that uses S3 for document storage and FAISS for local search."""
    
    def __init__(self, bucket_name: str = "sentinel-bucket-hackbrown"):
        self.bucket_name = bucket_name
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self._embeddings_data = None
        self._faiss_index = None
        self._chunks = None
        self._sources = None
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load all embeddings from S3 and build FAISS index."""
        print("[S3-RAG] Loading embeddings from S3...")
        
        # Get embeddings index
        index_data = s3_manager.get_embeddings_index()
        
        if not index_data.get('documents'):
            print("[S3-RAG] No documents found in embeddings index")
            return
        
        # Load all embeddings
        all_embeddings = []
        all_chunks = []
        all_sources = []
        
        for doc_info in index_data['documents']:
            embeddings_data = s3_manager.load_embeddings_data(doc_info['embeddings'])
            
            if embeddings_data:
                for i, chunk in enumerate(embeddings_data['chunks']):
                    all_embeddings.append(embeddings_data['embeddings'][i])
                    all_chunks.append(chunk)
                    all_sources.append(embeddings_data['document'])
        
        if not all_embeddings:
            print("[S3-RAG] No embeddings loaded")
            return
        
        # Build FAISS index
        embeddings_matrix = np.array(all_embeddings).astype('float32')
        faiss.normalize_L2(embeddings_matrix)
        
        self._faiss_index = faiss.IndexFlatIP(embeddings_matrix.shape[1])
        self._faiss_index.add(embeddings_matrix)
        
        self._chunks = all_chunks
        self._sources = all_sources
        
        print(f"[S3-RAG] Loaded {len(all_chunks)} chunks from {len(index_data['documents'])} documents")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        city: str | None = None,
        doc_type: str | None = None,
    ) -> List[Dict]:
        """
        Retrieve policy excerpts matching the query.
        Returns list of dicts compatible with existing RAG pipeline.
        """
        if not self._faiss_index:
            print("[S3-RAG] No embeddings available for search")
            return []
        
        # Embed query
        query_embedding = self.model.encode([query])[0].astype('float32')
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self._faiss_index.search(query_embedding, min(top_k, len(self._chunks)))
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self._chunks):
                # Create result compatible with existing RetrievedExcerpt schema
                result = {
                    "text": self._chunks[idx],
                    "source": self._sources[idx],
                    "score": float(score),
                    "metadata": {
                        "city": city or "Providence",
                        "doc_type": doc_type or "policy",
                        "rank": i + 1
                    }
                }
                results.append(result)
        
        return results
    
    def get_stats(self) -> Dict:
        """Get statistics about the loaded embeddings."""
        if not self._faiss_index:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "total_chunks": len(self._chunks) if self._chunks else 0,
            "total_documents": len(set(self._sources)) if self._sources else 0,
            "embedding_dimension": self._faiss_index.d if self._faiss_index else 0,
            "model_name": self.model._modules['0'].auto_model.name if hasattr(self.model, '_modules') else 'unknown'
        }

class S3RAGPipeline:
    """Complete RAG pipeline using S3 for storage."""
    
    def __init__(self, bucket_name: str = "sentinel-bucket-hackbrown"):
        self.retriever = S3PolicyRetriever(bucket_name)
    
    def retrieve(self, query: str, top_k: int = 5, **filters) -> List[Dict]:
        """Retrieve relevant documents."""
        return self.retriever.retrieve(query, top_k, **filters)
    
    def get_context(self, query: str, max_chars: int = 1000, **filters) -> str:
        """Get formatted context for RAG."""
        results = self.retrieve(query, top_k=3, **filters)
        
        context_parts = []
        current_chars = 0
        
        for result in results:
            chunk_text = f"[{result['source']}]: {result['text']}"
            
            if current_chars + len(chunk_text) > max_chars:
                # Truncate the last chunk if needed
                remaining_chars = max_chars - current_chars
                if remaining_chars > 50:  # Only add if we have meaningful space
                    chunk_text = chunk_text[:remaining_chars-3] + "..."
                    context_parts.append(chunk_text)
                break
            
            context_parts.append(chunk_text)
            current_chars += len(chunk_text)
        
        return '\n\n'.join(context_parts)
    
    def decide(self, decision_input):
        """Make a decision using the RAG pipeline - compatible with existing DecisionEngine interface."""
        from app.rag.decision_engine import _record_rag_stats, get_rag_stats
        from app.rag.schemas import DecisionOutput, SupportingExcerpt
        
        # Get context for decision
        query = ' '.join(decision_input.event_type_candidates + decision_input.signals)
        context = self.get_context(query, max_chars=1200)
        
        # For now, create a simple rule-based decision
        # In production, this would use an LLM with the context
        decision = {"action": "monitor", "priority": "medium"}
        explanation = "Based on the traffic policies and incident data, this situation requires monitoring."
        
        # Create supporting excerpts from retrieved documents
        supporting_excerpts = []
        results = self.retrieve(query, top_k=3)
        for i, result in enumerate(results):
            excerpt = SupportingExcerpt(
                document_id=result['source'],
                text=result['text'],
                score=result['score']  # Fixed: use 'score' instead of 'relevance_score'
            )
            supporting_excerpts.append(excerpt)
        
        # Record stats
        _record_rag_stats(supporting_excerpts)
        
        return DecisionOutput(
            decision=decision,
            explanation=explanation,
            supporting_excerpts=supporting_excerpts
        )

def create_aws_rag_pipeline(bucket_name: str = "sentinel-bucket-hackbrown"):
    """
    Create RAG pipeline that uses AWS S3.
    Compatible with existing create_rag_pipeline interface.
    """
    pipeline = S3RAGPipeline(bucket_name)
    
    # Return interface compatible with existing code
    return None, pipeline.retriever, pipeline
