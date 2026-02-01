"""
Automated S3 upload and embedding pipeline.
Uploads documents and processes embeddings for RAG.
"""

import os
import sys
from pathlib import Path
from typing import Tuple

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.s3_manager import s3_manager
from utils.embedding_processor import embedding_processor

def upload_documents_from_local_folder(documents_path: str = "documents") -> Tuple[int, int]:
    """Upload documents from local folder to S3 and process embeddings.
    
    Returns:
        Tuple of (uploaded_count, processed_count)
    """
    documents_folder = Path(documents_path)
    
    if not documents_folder.exists():
        print(f"❌ Documents folder not found: {documents_folder}")
        print("Please create a 'documents' folder with your PDF/text files")
        return 0, 0
    
    print(f"📁 Uploading documents from: {documents_folder}")
    
    # Upload documents to S3
    uploaded_count = s3_manager.upload_documents_from_folder(documents_folder)
    print(f"✅ Uploaded {uploaded_count} documents to S3")
    
    if uploaded_count == 0:
        print("⚠️  No documents were uploaded")
        return 0, 0
    
    # Process embeddings
    print("\n🔄 Processing embeddings...")
    processed_count, total_count = embedding_processor.process_all_documents()
    
    print(f"\n📊 Summary:")
    print(f"   Documents uploaded: {uploaded_count}")
    print(f"   Documents processed: {processed_count}/{total_count}")
    
    # Get stats
    stats = embedding_processor.get_embedding_stats()
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Embedding dimension: {stats['embedding_dimension']}")
    
    return uploaded_count, processed_count

def upload_videos_to_s3() -> int:
    """Upload all videos from backend/app/videos to S3."""
    from utils.video_manager import video_manager
    
    print("🎥 Uploading videos to S3...")
    uploaded_count = video_manager.upload_all_videos_to_s3()
    print(f"✅ Uploaded {uploaded_count} videos to S3")
    
    return uploaded_count

def setup_sample_documents() -> bool:
    """Create sample documents for testing if none exist."""
    documents_folder = Path("documents")
    documents_folder.mkdir(exist_ok=True)
    
    # Create sample files
    sample_files = {
        "policies/traffic-policy.txt": """
        TRAFFIC INCIDENT POLICY
        
        1. ACCIDENT RESPONSE
        - All accidents must be reported within 5 minutes
        - Emergency services contacted immediately for severe incidents
        - Traffic control measures implemented within 10 minutes
        
        2. INCIDENT CLASSIFICATION
        - Minor: Property damage only, no injuries
        - Major: Injuries requiring medical attention
        - Critical: Life-threatening injuries or fatalities
        
        3. PROTOCOL VIOLATIONS
        - Running red lights: Immediate notification required
        - Wrong way driving: High priority alert
        - Stopped vehicle in lane: Monitor for 2 minutes, then alert
        """,
        
        "regulations/city-ordinances.md": """
        # City Traffic Ordinances
        
        ## Speed Limits
        - Downtown: 25 mph
        - Residential: 20 mph
        - Highways: 55 mph
        
        ## Parking Regulations
        - No parking in bike lanes
        - 2-hour limit in commercial zones
        - Permit required in residential zones
        
        ## Traffic Control
        - Obey all traffic signals
        - Yield to pedestrians
        - No U-turns in business districts
        """,
        
        "procedures/incident-response.txt": """
        INCIDENT RESPONSE PROCEDURES
        
        STEP 1: DETECTION
        - Automated system detects potential incident
        - Verify incident severity through visual analysis
        
        STEP 2: CLASSIFICATION
        - Determine incident type (accident, breakdown, obstruction)
        - Assess severity level (minor, major, critical)
        
        STEP 3: NOTIFICATION
        - Contact emergency services if needed
        - Notify traffic management center
        - Update digital signage and traffic systems
        
        STEP 4: RESOLUTION
        - Monitor incident resolution
        - Clear incident when resolved
        - Document lessons learned
        """
    }
    
    created_count = 0
    for file_path, content in sample_files.items():
        full_path = documents_folder / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not full_path.exists():
            full_path.write_text(content.strip())
            print(f"✅ Created sample file: {file_path}")
            created_count += 1
        else:
            print(f"⚠️  File already exists: {file_path}")
    
    if created_count > 0:
        print(f"\n📝 Created {created_count} sample documents")
        print("You can now run the upload process:")
        print("  python -m utils.s3_uploader")
    
    return created_count > 0

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload documents and videos to S3")
    parser.add_argument("--documents", default="documents", help="Local documents folder")
    parser.add_argument("--videos-only", action="store_true", help="Only upload videos")
    parser.add_argument("--setup-samples", action="store_true", help="Create sample documents")
    parser.add_argument("--stats", action="store_true", help="Show RAG statistics")
    
    args = parser.parse_args()
    
    if args.stats:
        stats = embedding_processor.get_embedding_stats()
        print("📊 RAG Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        return
    
    if args.setup_samples:
        setup_sample_documents()
        return
    
    if args.videos_only:
        upload_videos_to_s3()
        return
    
    # Default: upload documents and process embeddings
    upload_documents_from_local_folder(args.documents)
    
    # Also upload videos
    upload_videos_to_s3()

if __name__ == "__main__":
    main()
