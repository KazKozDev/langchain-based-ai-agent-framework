#!/usr/bin/env python3
"""
Test script for RAG system functionality.
Tests vector database, document processing, and retrieval capabilities.
"""

import os
import sys
import tempfile
from pathlib import Path
import logging

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from agent.rag.vector_store import VectorStore
from agent.rag.document_processor import DocumentProcessor
from agent.rag.retrieval_tool import RAGRetrievalTool, RAGManagementTool
from langchain_core.documents import Document


def test_vector_store():
    """Test vector store functionality."""
    print("🧪 Testing Vector Store...")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test ChromaDB
        print("\n  📊 Testing ChromaDB...")
        try:
            chroma_store = VectorStore(
                store_type="chroma",
                persist_directory=os.path.join(temp_dir, "chroma_test"),
                collection_name="test_collection"
            )
            
            # Add test documents
            test_docs = [
                Document(
                    page_content="Python is a high-level programming language.",
                    metadata={"source": "test1", "category": "programming"}
                ),
                Document(
                    page_content="Machine learning is a subset of artificial intelligence.",
                    metadata={"source": "test2", "category": "ai"}
                ),
                Document(
                    page_content="Vector databases are used for similarity search.",
                    metadata={"source": "test3", "category": "database"}
                )
            ]
            
            # Add documents
            doc_ids = chroma_store.add_documents(test_docs)
            print(f"    ✅ Added {len(doc_ids)} documents to ChromaDB")
            
            # Search
            results = chroma_store.similarity_search("programming language", k=2)
            print(f"    ✅ Found {len(results)} results for 'programming language'")
            
            # Search with scores
            scored_results = chroma_store.similarity_search_with_score("machine learning", k=2)
            print(f"    ✅ Found {len(scored_results)} scored results")
            
            # Get collection info
            info = chroma_store.get_collection_info()
            print(f"    ✅ Collection info: {info['document_count']} documents")
            
        except Exception as e:
            print(f"    ❌ ChromaDB test failed: {e}")
        
        # Test FAISS
        print("\n  📊 Testing FAISS...")
        try:
            faiss_store = VectorStore(
                store_type="faiss",
                persist_directory=os.path.join(temp_dir, "faiss_test")
            )
            
            # Add documents
            doc_ids = faiss_store.add_documents(test_docs)
            print(f"    ✅ Added {len(doc_ids)} documents to FAISS")
            
            # Search
            results = faiss_store.similarity_search("artificial intelligence", k=2)
            print(f"    ✅ Found {len(results)} results for 'artificial intelligence'")
            
        except Exception as e:
            print(f"    ❌ FAISS test failed: {e}")


def test_document_processor():
    """Test document processor functionality."""
    print("\n🧪 Testing Document Processor...")
    
    processor = DocumentProcessor()
    
    # Test supported extensions
    extensions = processor.get_supported_extensions()
    print(f"  ✅ Supported extensions: {', '.join(extensions)}")
    
    # Create temporary test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Create test text file
        text_file = temp_dir / "test.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("This is a test document.\nIt contains multiple lines.\nFor testing purposes.")
        
        # Create test markdown file
        md_file = temp_dir / "test.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# Test Document\n\nThis is a **markdown** test file.\n\n## Section\n\nWith content.")
        
        # Create test Python file
        py_file = temp_dir / "test.py"
        with open(py_file, 'w', encoding='utf-8') as f:
            f.write("#!/usr/bin/env python3\n# Test Python file\nprint('Hello, World!')")
        
        # Test single file processing
        print("\n  📄 Testing single file processing...")
        
        # Process text file
        docs = processor.process_file(str(text_file))
        print(f"    ✅ Processed text file: {len(docs)} documents")
        if docs:
            print(f"    📝 Content preview: {docs[0].page_content[:50]}...")
        
        # Process markdown file
        docs = processor.process_file(str(md_file))
        print(f"    ✅ Processed markdown file: {len(docs)} documents")
        
        # Process Python file
        docs = processor.process_file(str(py_file))
        print(f"    ✅ Processed Python file: {len(docs)} documents")
        
        # Test directory processing
        print("\n  📁 Testing directory processing...")
        all_docs = processor.process_directory(str(temp_dir), recursive=True)
        print(f"    ✅ Processed directory: {len(all_docs)} total documents")
        
        # Test with patterns
        py_docs = processor.process_directory(
            str(temp_dir), 
            recursive=True, 
            file_patterns=['*.py']
        )
        print(f"    ✅ Processed Python files only: {len(py_docs)} documents")
        
        # Test text processing
        print("\n  📝 Testing text processing...")
        text_doc = processor.process_text("This is a direct text input for testing.")
        print(f"    ✅ Processed text: doc_id = {text_doc.metadata.get('doc_id', 'N/A')}")


def test_rag_tools():
    """Test RAG tools functionality."""
    print("\n🧪 Testing RAG Tools...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize RAG tool
        rag_tool = RAGRetrievalTool(
            store_type="chroma",
            persist_directory=os.path.join(temp_dir, "rag_test"),
            collection_name="test_rag"
        )
        
        # Initialize management tool
        management_tool = RAGManagementTool(rag_tool)
        
        print("\n  🔧 Testing RAG Management Tool...")
        
        # Add text document
        result = management_tool._manage_rag("add_text:RAG systems combine retrieval and generation for better AI responses.|title:RAG Explanation")
        print(f"    ✅ Added text document: {result[:100]}...")
        
        # Add more test content
        test_content = [
            "add_text:Vector databases store high-dimensional embeddings for similarity search.|title:Vector DB Info",
            "add_text:LangChain is a framework for building applications with language models.|title:LangChain Info",
            "add_text:ChromaDB is a vector database optimized for embeddings and metadata.|title:ChromaDB Info"
        ]
        
        for content in test_content:
            result = management_tool._manage_rag(content)
            print(f"    ✅ Added document")
        
        # Get collection info
        info_result = management_tool._manage_rag("info")
        print(f"    ℹ️  Collection info: {info_result.split(chr(10))[1] if chr(10) in info_result else info_result}")
        
        print("\n  🔍 Testing RAG Retrieval Tool...")
        
        # Test simple retrieval
        result = rag_tool._retrieve_documents("vector database")
        print(f"    ✅ Simple search result preview: {result[:100]}...")
        
        # Test retrieval with parameters
        result = rag_tool._retrieve_documents("query:LangChain framework|k:2")
        print(f"    ✅ Parameterized search completed")
        
        # Test retrieval with scores
        result = rag_tool._retrieve_documents("query:RAG system|with_scores:true|k:1")
        print(f"    ✅ Scored search completed")
        
        # Test LangChain Tool interfaces
        print("\n  🛠️  Testing LangChain Tool Interface...")
        
        retrieval_tool = rag_tool.get_tool()
        print(f"    ✅ Retrieval tool name: {retrieval_tool.name}")
        print(f"    📝 Tool description preview: {retrieval_tool.description[:100]}...")
        
        management_lc_tool = management_tool.get_tool()
        print(f"    ✅ Management tool name: {management_lc_tool.name}")
        print(f"    📝 Tool description preview: {management_lc_tool.description[:100]}...")


def test_integration():
    """Test full integration with agent."""
    print("\n🧪 Testing Full Integration...")
    
    try:
        from agent import OllamaAgent
        
        print("  🤖 Initializing agent with RAG...")
        # Note: This will only work if Ollama is running
        # For testing, we'll just check if the tools are loaded
        
        try:
            agent = OllamaAgent(verbose=False)  # Reduced verbosity for testing
            tools = agent.list_tools()
            
            rag_tools = [tool for tool in tools if 'rag' in tool.lower()]
            print(f"    ✅ Found RAG tools: {rag_tools}")
            
            if rag_tools:
                print("    ✅ RAG integration successful!")
            else:
                print("    ❌ RAG tools not found in agent")
                
        except Exception as e:
            print(f"    ⚠️  Agent initialization failed (Ollama may not be running): {e}")
            print("    ℹ️  This is expected if Ollama server is not available")
            
    except ImportError as e:
        print(f"    ❌ Import error: {e}")


def main():
    """Run all tests."""
    print("🚀 RAG System Test Suite")
    print("=" * 50)
    
    # Set up logging for tests
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    try:
        test_vector_store()
        test_document_processor()
        test_rag_tools()
        test_integration()
        
        print("\n" + "=" * 50)
        print("✅ All RAG system tests completed!")
        print("=" * 50)
        
        print("\n💡 To use RAG system:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run example: python examples/rag_example.py")
        print("3. Or use in your code: from agent.rag import RAGRetrievalTool")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()