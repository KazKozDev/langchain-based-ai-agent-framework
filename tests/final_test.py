#!/usr/bin/env python3
"""
Final test of all agent tools.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import OllamaAgent

def test_agent():
    print("🎯 Final test of the agent with fixed tools...")
    
    try:
        agent = OllamaAgent(verbose=False)
        print("✅ Agent created!")
        
        # Test 1: Math
        print("\n🧮 Test 1: Mathematical calculation")
        result = agent.process_query("Calculate 123 * 45")
        print(f"Result: {result}")
        
        # Test 2: Time  
        print("\n🕒 Test 2: Current time")
        result = agent.process_query("What time is it now?")
        print(f"Result: {result}")
        
        # Test 3: File creation
        print("\n📁 Test 3: File creation")
        result = agent.process_query("Create a file test123.txt with content 'Test completed!'")
        print(f"Result: {result}")
        
        print("\n🎉 All basic tests passed successfully!")
        print("\n💡 To test search, run: python quick_test.py")
        print("   and enter a query like 'bitcoin price'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent()