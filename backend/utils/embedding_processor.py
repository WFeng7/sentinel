"""
Embedding processor for RAG documents.
Handles text extraction, chunking, and embedding generation.
"""

import json
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import pypdf
from datetime import datetime, timezone

from .s3_manager import s3_manager

class EmbeddingProcessor:
    """Processes documents and creates embeddings for RAG."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.chunk_size = 500  # words per chunk
        self.chunk_overlap = 50  # words overlap between chunks
    
    def process_document(self, s3_key: str) -> bool:
        """Process a single document from S3 and save embeddings."""
        print(f"[Embedding] Processing document: {s3_key}")
        
        # Download document from S3
        document_bytes = s3_manager.download_document(s3_key)
        if not document_bytes:
            print(f"[Embedding] Failed to download: {s3_key}")
            return False
        
        # Extract text
        text = self._extract_text(document_bytes, s3_key)
        if not text.strip():
            print(f"[Embedding] No text extracted from: {s3_key}")
            return False
        
        # Create chunks
        chunks = self._chunk_text(text)
        if not chunks:
            print(f"[Embedding] No chunks created from: {s3_key}")
            return False
        
        # Generate embeddings
        embeddings = self.model.encode(chunks)
        embeddings_list = embeddings.tolist()
        
        # Save to S3
        embeddings_key = s3_manager.save_embeddings(s3_key, chunks, embeddings_list)
        
        success = embeddings_key is not None
        print(f"[Embedding] {'✅ Success' if success else '❌ Failed'}: {s3_key}")
        return success
    
    def process_all_documents(self) -> Tuple[int, int]:
        """Process all documents in S3 and return (success_count, total_count)."""
        documents = s3_manager.list_documents()
        
        if not documents:
            print("[Embedding] No documents found in S3")
            return 0, 0
        
        print(f"[Embedding] Found {len(documents)} documents to process")
        
        success_count = 0
        for doc in documents:
            if self.process_document(doc['key']):
                success_count += 1
        
        print(f"[Embedding] Processed {success_count}/{len(documents)} documents successfully")
        return success_count, len(documents)
    
    def _extract_text(self, document_bytes: bytes, filename: str) -> str:
        """Extract text from document bytes."""
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_file:
            tmp_file.write(document_bytes)
            tmp_file_path = tmp_file.name
        
        try:
            # Extract based on file type
            if filename.lower().endswith('.pdf'):
                return self._extract_pdf_text(tmp_file_path)
            elif filename.lower().endswith(('.txt', '.md')):
                return self._extract_text_file(tmp_file_path)
            else:
                print(f"[Embedding] Unsupported file type: {filename}")
                return ""
        finally:
            # Clean up temp file
            os.unlink(tmp_file_path)
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file."""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
        except Exception as e:
            print(f"[Embedding] Error extracting PDF text: {e}")
        
        return text.strip()
    
    def _extract_text_file(self, file_path: str) -> str:
        """Extract text from text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"[Embedding] Error reading text file: {e}")
            return ""
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        if len(words) <= self.chunk_size:
            return [text]
        
        # Create overlapping chunks
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            
            # Move start position with overlap
            if end >= len(words):
                break
            start = end - self.chunk_overlap
        
        return chunks
    
    def get_embedding_stats(self) -> Dict:
        """Get statistics about processed embeddings."""
        index = s3_manager.get_embeddings_index()
        
        total_documents = len(index['documents'])
        total_chunks = 0
        embedding_dim = None
        
        for doc_info in index['documents']:
            embeddings_data = s3_manager.load_embeddings_data(doc_info['embeddings'])
            if embeddings_data:
                total_chunks += embeddings_data.get('chunk_count', 0)
                if embedding_dim is None:
                    embedding_dim = embeddings_data.get('embedding_dim')
        
        # Get model name safely
        model_name = 'all-MiniLM-L6-v2'  # Default model name
        try:
            if hasattr(self.model, '_modules') and '0' in self.model._modules:
                model = self.model._modules['0']
                if hasattr(model, 'auto_model') and hasattr(model.auto_model, 'name'):
                    model_name = model.auto_model.name
        except:
            pass  # Use default name if any error
        
        return {
            'total_documents': total_documents,
            'total_chunks': total_chunks,
            'embedding_dimension': embedding_dim,
            'last_updated': index.get('last_updated'),
            'model_name': model_name
        }

# Global embedding processor instance
embedding_processor = EmbeddingProcessor()
