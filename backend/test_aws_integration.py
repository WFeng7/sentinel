#!/usr/bin/env python3
"""
Test script to verify AWS integration is working properly.
"""

from dotenv import load_dotenv
load_dotenv()
import os
import sys

def test_openai_key():
    """Test OpenAI API key is valid."""
    print("🔑 Testing OpenAI API Key...")
    try:
        from app.vlm.analyzer import EventAnalyzer
        analyzer = EventAnalyzer(api_key=os.environ.get('OPENAI_API_KEY'))
        print("✅ OpenAI API key loaded successfully")
        print(f"   Full key: {os.environ.get('OPENAI_API_KEY')}")
        return True
    except Exception as e:
        print(f"❌ OpenAI API key failed: {e}")
        return False

def test_gemini_key():
    """Test Gemini API key is valid."""
    print("🔑 Testing Gemini API Key...")
    try:
        from app.vlm import EventAnalyzer
        analyzer = EventAnalyzer(api_key=os.environ.get('GEMINI_API_KEY'))
        print("✅ Gemini API key loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Gemini API key failed: {e}")
        return False

def test_aws_credentials():
    """Test AWS credentials are working."""
    print("🔑 Testing AWS Credentials...")
    try:
        from utils.s3_manager import s3_manager
        # Test basic S3 access
        response = s3_manager.s3_client.list_buckets()
        buckets = [bucket['Name'] for bucket in response['Buckets']]
        
        if 'sentinel-bucket-hackbrown' in buckets:
            print("✅ AWS credentials working - found target bucket")
            return True
        else:
            print(f"❌ Target bucket not found. Available: {buckets[:3]}...")
            return False
    except Exception as e:
        print(f"❌ AWS credentials failed: {e}")
        return False

def test_rag_pipeline():
    """Test RAG pipeline is using AWS."""
    print("🧠 Testing AWS RAG Pipeline...")
    try:
        from app.main import get_decision_engine
        from app.rag.schemas import DecisionInput
        
        engine = get_decision_engine()
        decision_input = DecisionInput(
            event_type_candidates=['accident'],
            signals=['traffic collision detected'],
            city='Providence'
        )
        
        result = engine.decide(decision_input)
        print(f"✅ AWS RAG working - decision: {result.decision}")
        print(f"   Supporting excerpts: {len(result.supporting_excerpts)}")
        return True
    except Exception as e:
        print(f"❌ AWS RAG failed: {e}")
        return False

def main():
    print("🚀 Sentinel AWS Integration Test")
    print("=" * 40)
    
    tests = [
        ("Gemini API Key", test_gemini_key),
        ("OpenAI API Key", test_openai_key),
        ("AWS Credentials", test_aws_credentials), 
        ("AWS RAG Pipeline", test_rag_pipeline)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n📋 {name}:")
        results.append(test_func())
    
    print("\n" + "=" * 40)
    print("📊 Test Summary:")
    
    for i, (name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"   {name}: {status}")
    
    all_passed = all(results)
    if all_passed:
        print("\n🎉 All tests passed! AWS integration is ready.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
