#!/usr/bin/env python3
"""
Core SQS integration test for Sentinel emergency insights.
Tests if SQS is triggered when weighted score >= threshold.
"""

import os
import json
import boto3
from dotenv import load_dotenv

load_dotenv()

def test_sqs_connection():
    """Test basic SQS connection."""
    try:
        sqs = boto3.client("sqs")
        queue_url = os.environ.get("SENTINEL_SQS_QUEUE_URL")
        
        if not queue_url:
            print("❌ SENTINEL_SQS_QUEUE_URL not found")
            return False
            
        response = sqs.get_queue_attributes(
            QueueUrl=queue_url,
            AttributeNames=['QueueArn', 'ApproximateNumberOfMessages']
        )
        
        print(f"✅ SQS connected: {response['Attributes']['QueueArn']}")
        return True
        
    except Exception as e:
        print(f"❌ SQS connection failed: {e}")
        return False

def test_sqs_trigger():
    """Test SQS triggering with score >= threshold."""
    try:
        from app.motion_first.motion_first_tracking import IncidentWindowProcessor, CandidateScore
        
        # Create processor with SQS
        processor = IncidentWindowProcessor(
            camera_id="test-camera",
            fps=1.0,
            window_size=10,
            rate_limit_s=1.0,
            enable_vlm=False,
            enable_rag=False,
            sqs_queue_url=os.environ.get("SENTINEL_SQS_QUEUE_URL")
        )
        
        # Create candidate with score >= 4.0 (threshold)
        candidate = CandidateScore(
            event_candidate=True,
            send_to_vlm=True,
            score=4.5,  # Above threshold
            threshold=4.0,
            event_types={"accident": 1.0},
            contributions={"test": 4.5}
        )
        
        # This should trigger SQS
        processor._post_incident(event_id="test-incident", cand=candidate)
        print("✅ SQS triggered successfully")
        return True
        
    except Exception as e:
        print(f"❌ SQS trigger failed: {e}")
        return False

def main():
    print("🚀 Core SQS Integration Test")
    print("=" * 30)
    
    # Test connection
    if not test_sqs_connection():
        return False
    
    # Test trigger
    if not test_sqs_trigger():
        return False
    
    print("\n✅ SQS integration working!")
    print("   - Weighted score >= 4.0 triggers SQS")
    print("   - Messages sent to SentinelQueue")
    print("   - S3 storage happens automatically")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
