#!/usr/bin/env python3
"""
Test webscraper and observation stage in LangChain.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import OllamaAgent

def test_webscraper_and_observation():
    print("🕷️ Testing webscraper and observation stage...")
    
    try:
        agent = OllamaAgent(verbose=True)
        print(f"✅ Agent created with {len(agent.list_tools())} tools!")
        
        # Show all tools
        print("\n🛠️ Available tools:")
        for tool_name in agent.list_tools():
            descriptions = agent.get_tool_descriptions()
            print(f"  - {tool_name}: {descriptions.get(tool_name, 'No description')[:80]}...")
        
        print("\n" + "="*60)
        
        # Test 1: Math (simple test to check observation)
        print("\n🧮 Test 1: Math (check observation)")
        result = agent.process_query("Calculate 25 * 4")
        print(f"Result: {result}")
        
        print("\n" + "-"*40)
        
        # Test 2: Webscraper
        print("\n🕷️ Test 2: Webscraper")
        result = agent.process_query("Extract content from https://example.com")
        print(f"Result: {result}")
        
        print("\n" + "-"*40)
        
        # Test 3: Combined query
        print("\n🎯 Test 3: Combined query")
        complex_query = """
        Help me:
        1. Calculate what is 60 * 24
        2. Extract the title from https://example.com  
        3. Save the result to test_result.txt
        """
        result = agent.process_query(complex_query)
        print(f"Result: {result}")
        
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_webscraper_and_observation()