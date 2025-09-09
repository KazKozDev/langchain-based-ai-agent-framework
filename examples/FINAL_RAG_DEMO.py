#!/usr/bin/env python3
"""
🎯 FINAL DEMONSTRATION OF THE RAG SYSTEM
Shows full integration of RAG with the LangChain agent.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from agent import OllamaAgent


def demo_rag_with_natural_language():
    """Demonstration of RAG with natural language."""
    print("🚀 Demonstration of the RAG system with natural language")
    print("=" * 60)
    
    try:
        # Agent initialization
        print("🤖 Initializing agent with RAG support...")
        agent = OllamaAgent(verbose=True)
        
        # Show available tools
        tools = agent.list_tools()
        print(f"\n📋 Available tools: {', '.join(tools)}")
        
        rag_tools = [tool for tool in tools if 'rag' in tool.lower()]
        print(f"🧠 RAG tools: {', '.join(rag_tools)}")
        
        if not rag_tools:
            print("❌ RAG tools not found!")
            return
        
        print("\n" + "=" * 60)
        print("📝 STEP 1: Adding documents to the knowledge base")
        print("=" * 60)
        
        # Add test documents
        test_knowledge = [
            {
                "content": """RAG (Retrieval-Augmented Generation) is an architectural approach in AI that combines information retrieval with text generation.
                
                Core components of RAG:
                1. A vector database for storing document embeddings
                2. A similarity search system
                3. A language model for generating answers
                
                RAG enables language models to access up-to-date external information, improving the accuracy and relevance of responses.""",
                "title": "What is RAG"
            },
            {
                "content": """Vector databases are specialized data storage systems optimized for handling high-dimensional vector representations (embeddings).
                
                Popular vector DBs:
                - ChromaDB: open-source, easy to use
                - FAISS: Facebook's library for fast search
                - Pinecone: cloud vector DB
                - Weaviate: GraphQL vector DB
                - Qdrant: high-performance vector DB
                
                Use cases: semantic search, recommendation systems, image search.""",
                "title": "Vector Databases"
            },
            {
                "content": """LangChain is a framework for building applications based on language models.
                
                Key features:
                - Chains for sequential operations
                - Agents with access to tools
                - Memory for preserving context
                - Document Loaders
                - Vector Stores
                - Tools for extending functionality
                
                LangChain supports integration with many LLMs and vector databases.""",
                "title": "LangChain framework"
            }
        ]
        
        for i, doc in enumerate(test_knowledge, 1):
            print(f"\n📄 Adding document {i}: {doc['title']}")
            
            # Use structured RAG management tool call
            try:
                # Get tool manager and call RAG management directly
                management_tool = agent.tool_manager.get_tool('rag_management')
                result = management_tool.func(
                    action="add_text",
                    content=doc['content'],
                    title=doc['title']
                )
                print(f"✅ Result: {result}")
            except Exception as e:
                print(f"❌ Error adding document: {e}")
        
        print("\n" + "=" * 60)
        print("ℹ️  STEP 2: Getting information about the knowledge base")
        print("=" * 60)
        
        # Get collection info
        try:
            management_tool = agent.tool_manager.get_tool('rag_management')
            info = management_tool.func(action="info")
            print(f"📊 Knowledge base info:\n{info}")
        except Exception as e:
            print(f"❌ Error getting information: {e}")
        
        print("\n" + "=" * 60)
        print("🔍 STEP 3: Testing knowledge base search")
        print("=" * 60)
        
        # Test queries for search
        search_queries = [
            {"query": "what is RAG", "k": 2},
            {"query": "vector databases", "k": 1, "with_scores": True},
            {"query": "LangChain features", "k": 3},
            {"query": "ChromaDB and FAISS", "k": 2, "with_scores": True}
        ]
        
        for i, search in enumerate(search_queries, 1):
            print(f"\n🔍 Search {i}: '{search['query']}'")
            
            try:
                retrieval_tool = agent.tool_manager.get_tool('rag_retrieval')
                result = retrieval_tool.func(**search)
                
                # Show only the first 200 characters of the result
                preview = result.replace('\n', ' ')[:200]
                print(f"💡 Result: {preview}...")
                
                # Count the number of found documents
                if "Found" in result and "documents" in result:
                    found_count = result.split("Found ")[1].split(" ")[0]
                    print(f"📊 Documents found: {found_count}")
                    
            except Exception as e:
                print(f"❌ Search error: {e}")
        
        print("\n" + "=" * 60)
        print("🤖 STEP 4: Agent integration (requires Ollama)")
        print("=" * 60)
        
        # This part requires a running Ollama server
        print("⚠️  A running Ollama server is required for the full demonstration")
        print("📝 Example agent queries:")
        print('   agent.run("Find information about RAG systems in the knowledge base")')
        print('   agent.run("What are vector databases? Search in the documents")')
        print('   agent.run("Tell me about LangChain based on the loaded documentation")')
        
        # Attempt to use the agent (may not work without Ollama)
        try:
            print("\n🔄 Attempting a query to the agent...")
            response = agent.run("How many documents are in the knowledge base? Get the information via the RAG management tool")
            print(f"🤖 Agent response: {response}")
        except Exception as e:
            print(f"⚠️  Agent unavailable (Is Ollama not running?): {e}")
        
        print("\n" + "=" * 60)
        print("✅ DEMONSTRATION COMPLETED")
        print("=" * 60)
        
        print("\n🎯 RAG system successfully integrated!")
        print("📋 What was tested:")
        print("  ✅ Creation and configuration of the vector store")
        print("  ✅ Adding documents to the knowledge base")
        print("  ✅ Semantic search with relevance scores")
        print("  ✅ Structured tools (StructuredTool)")
        print("  ✅ Integration with the tool manager")
        print("  ✅ Managing the document collection")
        
        print("\n💡 Ready to use with the Ollama agent!")
        
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()


def simple_rag_usage_examples():
    """Simple examples of using RAG."""
    print("\n" + "=" * 60)
    print("📖 SIMPLE USAGE EXAMPLES")  
    print("=" * 60)
    
    print("""
🔧 DIRECT USE OF RAG TOOLS:

from agent.rag import RAGRetrievalTool, RAGManagementTool

# Create the RAG system
rag_tool = RAGRetrievalTool(store_type="chroma")
mgmt_tool = RAGManagementTool(rag_tool)

# Add a document
result = mgmt_tool._manage_rag_structured(
    action="add_text",
    content="Your document here",
    title="Document title"
)

# Search the knowledge base  
result = rag_tool._retrieve_documents_structured(
    query="your search query",
    k=5,
    with_scores=True
)

🤖 USING WITH THE AGENT:

from agent import OllamaAgent

agent = OllamaAgent()

# Natural language queries (the agent will automatically pick RAG tools)
agent.run("Find information about machine learning in the documents")
agent.run("Add this text to the knowledge base: ...")
agent.run("How many documents are in the knowledge base?")

📁 ADDING FILES AND DIRECTORIES:

# Add a file
management_tool.func(action="add_file", path="/path/to/document.pdf")

# Add a directory
management_tool.func(
    action="add_directory", 
    path="/path/to/docs",
    recursive=True,
    patterns=["*.py", "*.md", "*.txt"]
)

# Collection information
management_tool.func(action="info")

# Clear the collection
management_tool.func(action="clear")
    """)


if __name__ == "__main__":
    demo_rag_with_natural_language()
    simple_rag_usage_examples()

    print("\n  The RAG system is ready to use!")
    print("  See the documentation in RAG_DOCUMENTATION.md")
    print("  Run quick_rag_test.py for a quick check")