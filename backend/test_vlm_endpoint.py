#!/usr/bin/env python3
"""
Test the VLM endpoint with Gemini.
"""

import base64
import json
import requests
from dotenv import load_dotenv
load_dotenv()

def test_vlm_endpoint():
    """Test the VLM endpoint with a sample request."""
    print("🔍 Testing VLM endpoint with Gemini...")
    
    # Create a simple test image (1x1 pixel)
    test_image_data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    
    # Prepare the request
    payload = {
        "context": {
            "event_id": "test-event-123",
            "camera_id": "test-camera",
            "fps": 30.0,
            "window_seconds": 10.0,
            "cv_notes": "Test incident detection"
        },
        "keyframes": [
            {
                "ts": 0.0,
                "base64": base64.b64encode(test_image_data).decode()
            }
        ]
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/vlm/analyze",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ VLM endpoint working!")
            print(f"   Event type: {result.get('result', {}).get('event', {}).get('type', 'unknown')}")
            print(f"   Severity: {result.get('result', {}).get('event', {}).get('severity', 'unknown')}")
            print(f"   Confidence: {result.get('result', {}).get('event', {}).get('confidence', 0):.2f}")
            return True
        else:
            print(f"❌ VLM endpoint failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Start with: uvicorn app.main:app --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        print(f"❌ VLM test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_vlm_endpoint()
    exit(0 if success else 1)
