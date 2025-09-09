#!/usr/bin/env python3
"""
Test StructuredTool RAG implementation.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from agent.rag.retrieval_tool import RAGRetrievalTool, RAGManagementTool


def test_structured_tools():
    """Test structured RAG tools."""
    print("🧪 Testing Structured RAG Tools")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize tools
        rag_tool = RAGRetrievalTool(
            store_type="chroma",
            persist_directory=os.path.join(temp_dir, "structured_test"),
            collection_name="structured_test"
        )
        management_tool = RAGManagementTool(rag_tool)
        
        # Get LangChain tools
        retrieval_lc_tool = rag_tool.get_tool()
        management_lc_tool = management_tool.get_tool()
        
        print(f"✅ Retrieval tool type: {type(retrieval_lc_tool).__name__}")
        print(f"✅ Management tool type: {type(management_lc_tool).__name__}")
        
        # Test structured management
        print("\n📝 Testing structured management...")
        
        # Add text document
        result = management_lc_tool.func(
            action="add_text",
            content="This is a test document about vector databases and RAG systems.",
            title="Test Document"
        )
        print(f"Add text result: {result}")
        
        # Get info
        result = management_lc_tool.func(action="info")
        print(f"Info result: {result}")
        
        # Test structured retrieval
        print("\n🔍 Testing structured retrieval...")
        
        # Simple retrieval
        result = retrieval_lc_tool.func(query="vector database", k=3)
        print(f"Simple retrieval: {result[:100]}...")
        
        # Retrieval with scores
        result = retrieval_lc_tool.func(query="RAG system", k=2, with_scores=True)
        print(f"Scored retrieval: {result[:100]}...")
        
        print("\n✅ Structured tools test completed!")


if __name__ == "__main__":
    test_structured_tools()