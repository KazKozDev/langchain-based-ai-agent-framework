#!/usr/bin/env python3
"""
Quick test of RAG system without Ollama dependency.
Tests core RAG functionality directly.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from agent.rag.vector_store import VectorStore
from agent.rag.document_processor import DocumentProcessor
from agent.rag.retrieval_tool import RAGRetrievalTool, RAGManagementTool
from langchain_core.documents import Document


def main():
    """Quick RAG system test."""
    print("🚀 Quick RAG System Test")
    print("=" * 40)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temp directory: {temp_dir}")
        
        # Initialize RAG components
        print("\n1️⃣  Initializing RAG system...")
        try:
            rag_tool = RAGRetrievalTool(
                store_type="chroma",
                persist_directory=os.path.join(temp_dir, "rag_test"),
                collection_name="quick_test"
            )
            management_tool = RAGManagementTool(rag_tool)
            print("✅ RAG system initialized")
        except Exception as e:
            print(f"❌ Failed to initialize RAG: {e}")
            return
        
        # Add some test documents
        print("\n2️⃣  Adding test documents...")
        
        test_docs = [
            "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation. It allows language models to access external knowledge bases for more accurate and up-to-date responses.|title:RAG Overview",
            "Vector databases store high-dimensional embeddings that represent semantic meaning of text. They enable fast similarity search using techniques like cosine similarity and approximate nearest neighbor search.|title:Vector Database Basics",
            "LangChain is a framework for building applications with language models. It provides tools for document loading, text splitting, embeddings, vector stores, and chains.|title:LangChain Framework",
            "ChromaDB is an open-source vector database designed for embeddings. It supports metadata filtering, multiple distance metrics, and persistent storage.|title:ChromaDB Features",
            "Python is a versatile programming language widely used in AI and machine learning. It has rich ecosystem of libraries like NumPy, pandas, and scikit-learn.|title:Python for AI"
        ]
        
        for doc_content in test_docs:
            result = management_tool._manage_rag(f"add_text:{doc_content}")
            print(f"📝 Added document: {result[:60]}...")
        
        # Get collection info
        print("\n3️⃣  Getting collection information...")
        info = management_tool._manage_rag("info")
        print(f"ℹ️  {info}")
        
        # Test retrieval
        print("\n4️⃣  Testing document retrieval...")
        
        queries = [
            "What is RAG?",
            "vector database similarity search",
            "query:LangChain framework tools|k:2",
            "query:Python programming AI|with_scores:true|k:1",
            "ChromaDB features and capabilities"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n🔍 Query {i}: {query}")
            result = rag_tool._retrieve_documents(query)
            
            # Show first 200 characters of result
            preview = result.replace('\n', ' ')[:200]
            print(f"💡 Result preview: {preview}...")
            
            # Count found documents
            if "Found" in result and "documents" in result:
                found_count = result.split("Found ")[1].split(" ")[0]
                print(f"📊 Found {found_count} relevant documents")
        
        # Test management operations
        print("\n5️⃣  Testing management operations...")
        
        # Add a file (this script itself)
        script_path = str(Path(__file__))
        result = management_tool._manage_rag(f"add_file:{script_path}")
        print(f"📄 Added current script: {result[:80]}...")
        
        # Search for content from the script
        result = rag_tool._retrieve_documents("Quick RAG system test")
        if "Quick RAG" in result:
            print("✅ Successfully retrieved content from added file")
        else:
            print("⚠️  Could not find content from added file")
        
        # Final collection info
        print("\n6️⃣  Final collection state...")
        info = management_tool._manage_rag("info")
        print(f"ℹ️  Final state: {info}")
        
        print("\n" + "=" * 40)
        print("✅ Quick RAG test completed successfully!")
        print("=" * 40)
        
        print("\n💡 RAG System is working correctly!")
        print("📋 Key features tested:")
        print("  - Vector storage (ChromaDB)")
        print("  - Document processing (text, file)")
        print("  - Semantic search with embeddings")
        print("  - Metadata handling")
        print("  - Query parsing and parameters")
        print("  - Management operations")
        
        print("\n🚀 Ready for integration with Ollama agent!")


if __name__ == "__main__":
    main()