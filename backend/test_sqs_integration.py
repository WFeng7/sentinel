#!/usr/bin/env python3
"""
Test script to verify AWS SQS is working for emergency insights.
"""

import os
import json
import time
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

def test_sqs_connection():
    """Test basic SQS connection and queue access."""
    print("🔗 Testing SQS Connection...")
    try:
        import boto3
        sqs = boto3.client("sqs")
        
        # Get queue URL from environment
        queue_url = os.environ.get("SENTINEL_SQS_QUEUE_URL")
        if not queue_url:
            print("❌ SENTINEL_SQS_QUEUE_URL not found in environment")
            return False
            
        # Test queue access
        response = sqs.get_queue_attributes(
            QueueUrl=queue_url,
            AttributeNames=['QueueArn', 'ApproximateNumberOfMessages']
        )
        
        print(f"✅ SQS connection successful")
        print(f"   Queue ARN: {response['Attributes']['QueueArn']}")
        print(f"   Approximate messages: {response['Attributes']['ApproximateNumberOfMessages']}")
        return True, queue_url
        
    except Exception as e:
        print(f"❌ SQS connection failed: {e}")
        return False, None

def test_sqs_send_message(queue_url):
    """Test sending a test emergency insight to SQS."""
    print("\n📤 Testing SQS Send Message...")
    try:
        import boto3
        sqs = boto3.client("sqs")
        
        # Create a test emergency insight payload
        test_payload = {
            "camera_id": "test-camera-001",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "emergency_insight",
            "insight": {
                "type": "accident",
                "severity": "high",
                "description": "Test emergency insight - vehicle collision detected",
                "location": "Providence, RI",
                "confidence": 0.95
            },
            "metadata": {
                "test": True,
                "source": "sqs_test_script"
            }
        }
        
        # Send message
        response = sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(test_payload),
            MessageAttributes={
                "camera_id": {"DataType": "String", "StringValue": "test-camera-001"},
                "event_type": {"DataType": "String", "StringValue": "emergency_insight"},
                "severity": {"DataType": "String", "StringValue": "high"}
            }
        )
        
        print(f"✅ SQS message sent successfully")
        print(f"   Message ID: {response['MessageId']}")
        print(f"   MD5: {response['MD5OfMessageBody']}")
        return response['MessageId']
        
    except Exception as e:
        print(f"❌ SQS send message failed: {e}")
        return None

def test_sqs_receive_message(queue_url, message_id=None):
    """Test receiving messages from SQS."""
    print("\n📥 Testing SQS Receive Message...")
    try:
        import boto3
        sqs = boto3.client("sqs")
        
        # Receive messages
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=5,
            AttributeNames=['All'],
            MessageAttributeNames=['All']
        )
        
        messages = response.get('Messages', [])
        
        if not messages:
            print("ℹ️  No messages found in queue")
            return True
            
        print(f"✅ Found {len(messages)} messages in queue")
        
        for i, msg in enumerate(messages):
            print(f"\n   Message {i+1}:")
            print(f"     ID: {msg['MessageId']}")
            print(f"     Body: {msg['Body'][:200]}...")
            
            # Check if this is our test message
            if message_id and msg['MessageId'] == message_id:
                print("     ✅ Found our test message!")
                
                # Clean up test message
                sqs.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=msg['ReceiptHandle']
                )
                print("     🧹 Test message deleted")
        
        return True
        
    except Exception as e:
        print(f"❌ SQS receive message failed: {e}")
        return False

def test_motion_first_sqs():
    """Test motion first pipeline SQS integration."""
    print("\n🎬 Testing Motion First SQS Integration...")
    try:
        from app.motion_first.motion_first_tracking import IncidentWindowProcessor, CandidateScore
        
        # Create a mock processor with SQS enabled
        processor = IncidentWindowProcessor(
            camera_id="test-camera-sqs",
            fps=1.0,
            window_size=10,
            rate_limit_s=1.0,
            enable_vlm=False,
            enable_rag=False,
            sqs_queue_url=os.environ.get("SENTINEL_SQS_QUEUE_URL")
        )
        
        # Create a test candidate score
        test_candidate = CandidateScore(
            event_candidate=True,
            send_to_vlm=True,
            score=5.0,
            threshold=4.0,
            event_types={"accident": 1.0},
            contributions={"test": 1.0}
        )
        
        # Test the _post_incident method which includes SQS
        processor._post_incident(event_id="test-event-001", cand=test_candidate)
        
        print("✅ Motion First SQS integration test completed")
        return True
        
    except Exception as e:
        print(f"❌ Motion First SQS integration failed: {e}")
        return False

def main():
    print("🚀 Sentinel SQS Test")
    print("=" * 40)
    
    # Test SQS connection
    connection_result = test_sqs_connection()
    if not connection_result[0]:
        print("\n❌ SQS connection failed. Cannot continue.")
        return False
    
    queue_url = connection_result[1]
    
    # Test sending message
    message_id = test_sqs_send_message(queue_url)
    
    # Test receiving messages
    receive_result = test_sqs_receive_message(queue_url, message_id)
    
    # Test motion first integration
    motion_result = test_motion_first_sqs()
    
    print("\n" + "=" * 40)
    print("📊 SQS Test Summary:")
    print(f"   SQS Connection: {'✅ PASS' if connection_result[0] else '❌ FAIL'}")
    print(f"   SQS Send Message: {'✅ PASS' if message_id else '❌ FAIL'}")
    print(f"   SQS Receive Message: {'✅ PASS' if receive_result else '❌ FAIL'}")
    print(f"   Motion First Integration: {'✅ PASS' if motion_result else '❌ FAIL'}")
    
    all_passed = all([connection_result[0], bool(message_id), receive_result, motion_result])
    
    if all_passed:
        print("\n🎉 All SQS tests passed! Emergency insights will be sent to SQS.")
    else:
        print("\n⚠️  Some SQS tests failed. Check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
