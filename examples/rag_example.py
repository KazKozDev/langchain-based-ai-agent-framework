#!/usr/bin/env python3
"""
Example usage of RAG (Retrieval-Augmented Generation) system with the LangChain agent.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from agent import OllamaAgent


def main():
    """
    Demonstrate RAG functionality with the LangChain agent.
    """
    print("🤖 LangChain Agent with RAG System Example")
    print("=" * 50)
    
    # Initialize agent with RAG support
    try:
        agent = OllamaAgent(
            model_name="gpt-oss:20b",
            temperature=0.1,
            verbose=True
        )
        print("✅ Agent initialized successfully with RAG support")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    # Show available tools (should include RAG tools)
    print("\n📋 Available tools:")
    tools = agent.list_tools()
    for tool in tools:
        print(f"  - {tool}")
    
    print("\n" + "=" * 50)
    print("RAG System Demonstration")
    print("=" * 50)
    
    # Example 1: Add some documents to the knowledge base
    print("\n1️⃣  Adding documents to knowledge base...")
    
    # Add this script as a document
    script_path = str(Path(__file__))
    response = agent.run(f"rag_management:add_file:{script_path}")
    print(f"📄 Adding current script: {response}")
    
    # Add README file if it exists
    readme_path = str(Path(__file__).parent.parent / "README.md")
    if Path(readme_path).exists():
        response = agent.run(f"rag_management:add_file:{readme_path}")
        print(f"📄 Adding README: {response}")
    
    # Add some text directly
    text_content = """
    RAG (Retrieval-Augmented Generation) is a technique that combines:
    1. Information retrieval from a knowledge base
    2. Text generation using a language model
    
    Key benefits:
    - Access to external knowledge
    - Factual accuracy
    - Contextual responses
    - Domain-specific expertise
    """
    response = agent.run(f"rag_management:add_text:{text_content}|title:RAG Overview")
    print(f"📝 Adding text document: {response}")
    
    # Example 2: Get knowledge base info
    print("\n2️⃣  Getting knowledge base information...")
    response = agent.run("rag_management:info")
    print(f"ℹ️  Knowledge base info:\n{response}")
    
    # Example 3: Query the knowledge base
    print("\n3️⃣  Querying the knowledge base...")
    
    # Simple query
    response = agent.run("rag_retrieval:What is RAG?")
    print(f"🔍 Query 'What is RAG?':\n{response}")
    
    # Query with parameters
    response = agent.run("rag_retrieval:query:Python agent implementation|k:3")
    print(f"🔍 Query 'Python agent implementation' (top 3):\n{response}")
    
    # Query with scores
    response = agent.run("rag_retrieval:query:LangChain tools|with_scores:true|k:2")
    print(f"🔍 Query 'LangChain tools' (with scores):\n{response}")
    
    # Example 4: Advanced RAG queries through natural language
    print("\n4️⃣  Advanced RAG queries...")
    
    queries = [
        "Find information about the RAG system in the knowledge base",
        "What tools are available in this agent? Search the documentation",
        "Explain how the vector database works based on the uploaded documents"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n📝 Query {i}: {query}")
        response = agent.run(query)
        print(f"🤖 Response: {response}")
        print("-" * 30)
    
    print("\n" + "=" * 50)
    print("RAG System Features Demonstration Complete!")
    print("=" * 50)
    
    print("\n💡 Try these commands:")
    print("1. Add more documents: rag_management:add_directory:path/to/docs")
    print("2. Search knowledge base: rag_retrieval:your search query")
    print("3. Clear knowledge base: rag_management:clear")
    print("4. Get collection info: rag_management:info")
    
    # Interactive mode
    print("\n🔄 Interactive mode (type 'exit' to quit):")
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if user_input:
                response = agent.run(user_input)
                print(f"🤖 Agent: {response}")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()