"""
S3 utilities for Sentinel AWS integration.
Handles uploading incidents, documents, and managing embeddings.
"""

import os
import json
import boto3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# S3 Configuration
BUCKET_NAME = "sentinel-bucket-hackbrown"
INCIDENTS_PREFIX = "sentinel/incidents"
DOCUMENTS_PREFIX = "documents/rag-data"
EMBEDDINGS_PREFIX = "documents/embeddings"
VIDEOS_PREFIX = "videos"

class S3Manager:
    """Manages S3 operations for Sentinel."""
    
    def __init__(self, bucket_name: str = BUCKET_NAME):
        self.bucket_name = bucket_name
        
        # Check if running on EC2 with IAM role (preferred for production)
        try:
            import urllib.request
            urllib.request.urlopen('http://169.254.169.254/latest/meta-data/instance-id', timeout=2)
            # We're on EC2, use instance profile
            self.s3_client = boto3.client('s3')
            print("[S3] Using EC2 instance profile (IAM role)")
        except:
            # Not on EC2 or no metadata access, use credentials
            pem_file = os.environ.get('AWS_PEM_FILE')
            if pem_file and os.path.exists(pem_file):
                # For local development with PEM key reference
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                    region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
                )
                print(f"[S3] Using AWS credentials (PEM file reference: {pem_file})")
            else:
                # Standard credentials
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                    region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
                )
                print("[S3] Using standard AWS credentials")
    
    def upload_incident_keyframes(
        self, 
        keyframes: List[Tuple[float, bytes]], 
        camera_id: str, 
        event_id: str
    ) -> List[str]:
        """Upload incident keyframes to S3."""
        uris = []
        
        for idx, (timestamp, jpeg_bytes) in enumerate(keyframes):
            key = f"{INCIDENTS_PREFIX}/{camera_id}/{event_id}/kf_{idx}.jpg"
            
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=jpeg_bytes,
                    ContentType='image/jpeg'
                )
                uris.append(f"s3://{self.bucket_name}/{key}")
                print(f"[S3] Uploaded keyframe: {key}")
            except Exception as e:
                print(f"[S3] Failed to upload keyframe {key}: {e}")
                
        return uris
    
    def upload_incident_metadata(
        self,
        camera_id: str,
        event_id: str,
        metadata: Dict
    ) -> Optional[str]:
        """Upload incident metadata to S3."""
        key = f"{INCIDENTS_PREFIX}/{camera_id}/{event_id}/metadata.json"
        
        try:
            metadata['uploaded_at'] = datetime.now(timezone.utc).isoformat()
            metadata['bucket'] = self.bucket_name
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json.dumps(metadata, indent=2),
                ContentType='application/json'
            )
            
            uri = f"s3://{self.bucket_name}/{key}"
            print(f"[S3] Uploaded metadata: {key}")
            return uri
            
        except Exception as e:
            print(f"[S3] Failed to upload metadata {key}: {e}")
            return None
    
    def upload_video_file(self, video_path: Path, camera_id: str) -> Optional[str]:
        """Upload a video file to S3."""
        if not video_path.exists():
            print(f"[S3] Video file not found: {video_path}")
            return None
            
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        key = f"{VIDEOS_PREFIX}/{camera_id}/{timestamp}_{video_path.name}"
        
        try:
            self.s3_client.upload_file(str(video_path), self.bucket_name, key)
            uri = f"s3://{self.bucket_name}/{key}"
            print(f"[S3] Uploaded video: {key}")
            return uri
            
        except Exception as e:
            print(f"[S3] Failed to upload video {key}: {e}")
            return None
    
    def upload_documents_from_folder(self, documents_folder: Path) -> int:
        """Upload all documents from a local folder to S3."""
        if not documents_folder.exists():
            print(f"[S3] Documents folder not found: {documents_folder}")
            return 0
            
        uploaded_count = 0
        
        for file_path in documents_folder.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.txt', '.md', '.docx']:
                # Create S3 key preserving folder structure
                relative_path = file_path.relative_to(documents_folder)
                s3_key = f"{DOCUMENTS_PREFIX}/{relative_path}"
                
                try:
                    self.s3_client.upload_file(str(file_path), self.bucket_name, s3_key)
                    print(f"[S3] Uploaded document: {s3_key}")
                    uploaded_count += 1
                except Exception as e:
                    print(f"[S3] Failed to upload document {s3_key}: {e}")
                    
        return uploaded_count
    
    def list_documents(self) -> List[Dict]:
        """List all documents in the rag-data folder."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=DOCUMENTS_PREFIX + '/'
            )
            
            documents = []
            for obj in response.get('Contents', []):
                if not obj['Key'].endswith('/'):
                    documents.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'],
                        'name': obj['Key'].replace(f'{DOCUMENTS_PREFIX}/', '')
                    })
            
            return documents
            
        except Exception as e:
            print(f"[S3] Failed to list documents: {e}")
            return []
    
    def download_document(self, s3_key: str) -> Optional[bytes]:
        """Download a document from S3."""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return response['Body'].read()
            
        except Exception as e:
            print(f"[S3] Failed to download document {s3_key}: {e}")
            return None
    
    def save_embeddings(
        self, 
        document_key: str, 
        chunks: List[str], 
        embeddings: List[List[float]]
    ) -> Optional[str]:
        """Save embeddings for a document."""
        embeddings_key = document_key.replace(DOCUMENTS_PREFIX, EMBEDDINGS_PREFIX) + '.json'
        
        embeddings_data = {
            'document': document_key,
            'chunks': chunks,
            'embeddings': embeddings,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'chunk_count': len(chunks),
            'embedding_dim': len(embeddings[0]) if embeddings else 0
        }
        
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=embeddings_key,
                Body=json.dumps(embeddings_data),
                ContentType='application/json'
            )
            
            # Update index
            self._update_embeddings_index(document_key, embeddings_key)
            
            print(f"[S3] Saved embeddings: {embeddings_key}")
            return embeddings_key
            
        except Exception as e:
            print(f"[S3] Failed to save embeddings {embeddings_key}: {e}")
            return None
    
    def _update_embeddings_index(self, document_key: str, embeddings_key: str):
        """Update the embeddings index file."""
        index_key = f"{EMBEDDINGS_PREFIX}/index.json"
        
        try:
            # Try to load existing index
            response = self.s3_client.get_object(self.bucket_name, index_key)
            index = json.loads(response['Body'].read())
        except:
            # Create new index if none exists
            index = {'documents': [], 'last_updated': None}
        
        # Add new document
        index['documents'].append({
            'document': document_key,
            'embeddings': embeddings_key,
            'processed_at': datetime.now(timezone.utc).isoformat()
        })
        index['last_updated'] = datetime.now(timezone.utc).isoformat()
        
        # Save updated index
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=index_key,
            Body=json.dumps(index, indent=2),
            ContentType='application/json'
        )
    
    def get_embeddings_index(self) -> Dict:
        """Get the embeddings index."""
        index_key = f"{EMBEDDINGS_PREFIX}/index.json"
        
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=index_key)
            return json.loads(response['Body'].read())
        except Exception as e:
            print(f"[S3] Failed to get embeddings index: {e}")
            return {'documents': [], 'last_updated': None}
    
    def load_embeddings_data(self, embeddings_key: str) -> Optional[Dict]:
        """Load embeddings data for a specific document."""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=embeddings_key)
            return json.loads(response['Body'].read())
        except Exception as e:
            print(f"[S3] Failed to load embeddings {embeddings_key}: {e}")
            return None

# Global S3 manager instance
s3_manager = S3Manager()
