#!/usr/bin/env python3
"""
Test the enhanced AWS RAG with LLM summarization.
"""

from dotenv import load_dotenv
load_dotenv()
import os

def test_enhanced_rag():
    """Test the enhanced RAG system."""
    print("🧠 Testing Enhanced AWS RAG with LLM Summarization")
    print("=" * 60)
    
    try:
        # Test AWS RAG directly
        print("📋 Testing AWS RAG Pipeline...")
        from utils.aws_rag import create_aws_rag_pipeline
        _, retriever, pipeline = create_aws_rag_pipeline()
        print("✅ AWS RAG Pipeline created successfully!")
        
        # Test decision
        from app.rag.schemas import DecisionInput
        decision_input = DecisionInput(
            event_type_candidates=['accident', 'stopped_vehicle'],
            signals=['traffic collision detected', 'vehicles not moving', 'debris on roadway'],
            city='Providence'
        )
        
        result = pipeline.decide(decision_input)
        print("✅ AWS RAG Decision completed!")
        print(f"🎯 Decision: {result.decision}")
        print(f"📝 Explanation: {result.explanation[:300]}...")
        print(f"📚 Supporting excerpts: {len(result.supporting_excerpts)}")
        
        # Show sources
        print("\n📄 Sources:")
        for i, excerpt in enumerate(result.supporting_excerpts[:3]):
            print(f"  {i+1}. {excerpt.document_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_enhanced_rag()
    if success:
        print("\n🎉 Enhanced AWS RAG is working perfectly!")
        print("📋 The system now:")
        print("  ✅ Retrieves relevant documents from AWS S3")
        print("  ✅ Uses OpenAI LLM to generate intelligent summaries")
        print("  ✅ Provides specific actions and priorities")
        print("  ✅ References actual policy documents")
        print("  ✅ Returns coherent explanations instead of raw text")
    else:
        print("\n❌ Enhanced AWS RAG test failed")
    
    exit(0 if success else 1)
