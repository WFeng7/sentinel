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
        import os
        
        # Get context for decision
        query = ' '.join(decision_input.event_type_candidates + decision_input.signals)
        
        # Retrieve relevant documents
        results = self.retrieve(query, top_k=5)
        
        # Create supporting excerpts from retrieved documents
        supporting_excerpts = []
        context_text = ""
        
        for i, result in enumerate(results):
            excerpt = SupportingExcerpt(
                document_id=result['source'],
                text=result['text'],
                score=result['score']
            )
            supporting_excerpts.append(excerpt)
            context_text += f"\nDocument {i+1} ({result['source']}):\n{result['text']}\n"
        
        # Generate LLM summary using OpenAI
        decision, explanation = self._generate_llm_decision(query, context_text, decision_input)
        
        # Record stats
        _record_rag_stats(supporting_excerpts)
        
        return DecisionOutput(
            decision=decision,
            explanation=explanation,
            supporting_excerpts=supporting_excerpts
        )
    
    def _generate_llm_decision(self, query, context_text, decision_input):
        """Generate decision and explanation using OpenAI LLM with retrieved context."""
        import os
        import json
        import re
        
        try:
            from openai import OpenAI
            
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                # Fallback to default response
                return {"action": "monitor", "priority": "medium"}, "OpenAI API key not configured. Based on the traffic policies and incident data, this situation requires monitoring."
            
            client = OpenAI(api_key=api_key)
            
            # Check if this is an unknown/no-event situation
            event_types = decision_input.event_type_candidates or []
            signals = decision_input.signals or []
            
            is_unknown = (not event_types or 'unknown' in event_types[0].lower()) and \
                        (not signals or all('no significant' in s.lower() or 'unknown' in s.lower() for s in signals))
            
            if is_unknown:
                return {
                    "action": "monitor", 
                    "priority": "low"
                }, "No significant events detected in the camera feed. The traffic monitoring system shows normal conditions with no incidents requiring immediate attention. Continue routine monitoring."
            
            # Create the prompt for LLM
            prompt = f"""You are a traffic incident analysis expert. Based on the retrieved policy documents and the incident information, provide a clear decision and explanation.

INCIDENT INFORMATION:
- Event Types: {', '.join(decision_input.event_type_candidates)}
- Signals: {', '.join(decision_input.signals)}
- Location: {decision_input.city}

RETRIEVED POLICY DOCUMENTS:
{context_text}

TASK:
1. Analyze the incident against the retrieved policies
2. Provide a specific decision (action and priority level)
3. Write a clear explanation that references the relevant policies

RESPONSE FORMAT:
Decision: {{"action": "specific_action", "priority": "high/medium/low"}}
Explanation: [Clear explanation referencing specific policies and procedures]

Example response:
Decision: {{"action": "dispatch_emergency_services", "priority": "high"}}
Explanation: Based on the traffic safety policies, this incident requires immediate emergency response due to [specific reason]. The NPFD Operations Manual (SAF 1.2) mandates apparatus response for such incidents, and the Strategic Highway Safety Plan requires high-priority intervention for similar scenarios.

Provide your analysis:"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a traffic incident analysis expert specializing in emergency response protocols and traffic safety policies."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content.strip()
            print(f"[AWS-RAG] LLM Response: {response_text[:200]}...")
            
            # Parse the response
            decision = {"action": "monitor", "priority": "medium"}
            explanation = response_text
            
            # Try to extract decision from response
            if "Decision:" in response_text:
                try:
                    # Extract the JSON part after "Decision:"
                    decision_start = response_text.find("Decision:") + len("Decision:")
                    decision_end = response_text.find("Explanation:", decision_start)
                    if decision_end == -1:
                        decision_end = len(response_text)
                    
                    decision_part = response_text[decision_start:decision_end].strip()
                    print(f"[AWS-RAG] Decision part: {decision_part}")
                    
                    # Try to parse as JSON
                    import json
                    decision = json.loads(decision_part)
                    print(f"[AWS-RAG] Parsed decision: {decision}")
                except Exception as e:
                    print(f"[AWS-RAG] JSON parsing failed: {e}")
                    # Try to extract action and priority manually
                    if "action" in decision_part.lower() and "priority" in decision_part.lower():
                        # Simple regex extraction
                        import re
                        action_match = re.search(r'action["\']?\s*[:=]\s*["\']?(\w+)["\']?', decision_part.lower())
                        priority_match = re.search(r'priority["\']?\s*[:=]\s*["\']?(\w+)["\']?', decision_part.lower())
                        
                        if action_match:
                            decision["action"] = action_match.group(1)
                        if priority_match:
                            decision["priority"] = priority_match.group(1)
            
            # Extract explanation
            if "Explanation:" in response_text:
                explanation_start = response_text.find("Explanation:") + len("Explanation:")
                explanation = response_text[explanation_start:].strip()
            else:
                # If no "Explanation:" tag, use everything after the decision
                if "Decision:" in response_text:
                    decision_end = response_text.find("Explanation:", response_text.find("Decision:"))
                    if decision_end != -1:
                        explanation = response_text[decision_end + len("Explanation:"):].strip()
                    else:
                        explanation = response_text.split("Decision:")[1].strip()
            
            return decision, explanation
            
        except Exception as e:
            print(f"[AWS-RAG] LLM generation failed: {e}")
            # Fallback to default response
            return {"action": "monitor", "priority": "medium"}, f"Based on the traffic policies and incident data, this situation requires monitoring. Error: {str(e)}"

def create_aws_rag_pipeline(bucket_name: str = "sentinel-bucket-hackbrown"):
    """
    Create RAG pipeline that uses AWS S3.
    Compatible with existing create_rag_pipeline interface.
    """
    pipeline = S3RAGPipeline(bucket_name)
    
    # Return interface compatible with existing code
    return None, pipeline.retriever, pipeline
