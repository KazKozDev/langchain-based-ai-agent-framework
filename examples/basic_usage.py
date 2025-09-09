#!/usr/bin/env python3
"""
Basic example of using the LangChain agent with Ollama.
"""

import sys
import os

# Add parent directory to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import OllamaAgent


def main():
    """Main function with agent usage examples."""
    
    print("🚀 Initializing LangChain agent with Ollama...")
    
    # Create agent
    try:
        agent = OllamaAgent(
            model_name="gpt-oss:20b",
            temperature=0.1,
            verbose=True
        )
        print("✅ Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Agent initialization error: {e}")
        print("Make sure Ollama is running and the gpt-oss:20b model is available.")
        return
    
    # Show available tools
    print("\n📋 Available tools:")
    tools = agent.list_tools()
    for tool_name in tools:
        print(f"  - {tool_name}")
    
    print("\n" + "="*60)
    
    # Example 1: Simple math calculation
    print("\n🧮 Example 1: Math calculation")
    response1 = agent.run("Calculate (15 + 25) * 2 - 10")
    print(f"Answer: {response1}")
    
    print("\n" + "-"*40)
    
    # Example 2: Get current time
    print("\n🕒 Example 2: Get current time")
    response2 = agent.run("What is the current time and date?")
    print(f"Answer: {response2}")
    
    print("\n" + "-"*40)
    
    # Example 3: Search for information
    print("\n🔍 Example 3: Information search")
    response3 = agent.run("Find information about the Python programming language")
    print(f"Answer: {response3}")
    
    print("\n" + "-"*40)
    
    # Example 4: Working with files
    print("\n📁 Example 4: Create a file")
    response4 = agent.run("Create a file test_file.txt with content 'Hello from the agent!'")
    print(f"Answer: {response4}")
    
    # Verify that the file was created
    response5 = agent.run("Read the contents of the file test_file.txt")
    print(f"File check: {response5}")
    
    print("\n" + "-"*40)
    
    # Example 5: Complex task
    print("\n🎯 Example 5: Complex task")
    complex_query = """
    Help me create a report:
    1. Calculate how much 365 days * 24 hours is
    2. Get the current date
    3. Save the result to a file report.txt
    """
    response6 = agent.run(complex_query)
    print(f"Answer: {response6}")
    
    print("\n" + "="*60)
    print("✅ All examples completed!")
    
    # Show conversation history
    print("\n💭 Conversation history:")
    memory = agent.get_memory()
    for i, message in enumerate(memory[-4:], 1):  # Last 4 messages
        role = "User" if hasattr(message, 'content') else "Agent"
        content = message.content if hasattr(message, 'content') else str(message)
        print(f"  {i}. {role}: {content[:100]}...")


if __name__ == "__main__":
    main()