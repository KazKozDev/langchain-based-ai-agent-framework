#!/usr/bin/env python3
"""
Example of creating and adding custom tools to the agent.
"""

import sys
import os
import requests
import random

# Add parent directory to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import OllamaAgent
from langchain_core.tools import Tool


def weather_tool_function(city: str) -> str:
    """
    Example tool for getting the weather (simulation).
    
    Args:
        city: City name
        
    Returns:
        Weather information
    """
    if not city:
        return "Error: City not specified"
    
    # Simulate weather request (in a real app, an API call would be here)
    weather_conditions = ["sunny", "cloudy", "rain", "snow", "fog"]
    temperature = random.randint(-10, 35)
    condition = random.choice(weather_conditions)
    
    return f"Weather in {city}: {condition}, temperature {temperature}°C"


def currency_converter_function(query: str) -> str:
    """
    Example tool for currency conversion (simulation).
    
    Args:
        query: Query in the format "100 USD to EUR"
        
    Returns:
        Conversion result
    """
    try:
        parts = query.split()
        if len(parts) != 4 or parts[2].lower() != "to":
            return "Format: '100 USD to EUR'"
        
        amount = float(parts[0])
        from_currency = parts[1].upper()
        to_currency = parts[3].upper()
        
        # Simulated exchange rates
        rates = {
            ("USD", "EUR"): 0.85,
            ("EUR", "USD"): 1.18,
            ("USD", "RUB"): 90.0,
            ("RUB", "USD"): 0.011,
            ("EUR", "RUB"): 106.0,
            ("RUB", "EUR"): 0.0094
        }
        
        rate = rates.get((from_currency, to_currency))
        if rate is None:
            return f"Rate {from_currency} -> {to_currency} not found"
        
        result = amount * rate
        return f"{amount} {from_currency} = {result:.2f} {to_currency}"
        
    except ValueError:
        return "Error: Invalid amount"
    except Exception as e:
        return f"Conversion error: {str(e)}"


def random_joke_function(category: str = "general") -> str:
    """
    Example tool for getting a random joke.
    
    Args:
        category: Joke category
        
    Returns:
        A random joke
    """
    jokes = {
        "programming": [
            "Why do programmers confuse Halloween and Christmas? Because 31 Oct = 25 Dec!",
            "Wife tells a programmer: 'Go to the store for bread. And if they have eggs, buy a dozen.' The programmer comes home with 10 loaves of bread.",
            "What do you call a programmer who doesn't drink coffee? Asleep."
        ],
        "general": [
            "What does a fish do when it's bored? Plays sea tic-tac-toe!",
            "Why doesn't the bear wear shoes? Because it has paws!",
            "What did one ocean say to the other? Nothing, they just waved."
        ],
        "science": [
            "Why did the atom lose an electron? It was positive!",
            "What did oxygen say to hydrogen? Without me, you are nothing!",
            "Why do mathematicians love parks? Because of natural logs."
        ]
    }
    
    category_jokes = jokes.get(category.lower(), jokes["general"])
    return random.choice(category_jokes)


def system_info_function(info_type: str) -> str:
    """
    Example tool for getting system information.
    
    Args:
        info_type: Type of information (os, python, memory, etc.)
        
    Returns:
        System information
    """
    import platform
    import psutil
    
    try:
        if info_type.lower() == "os":
            return f"Operating system: {platform.system()} {platform.release()}"
        elif info_type.lower() == "python":
            return f"Python version: {platform.python_version()}"
        elif info_type.lower() == "memory":
            memory = psutil.virtual_memory()
            return f"Memory: {memory.percent}% used ({memory.used // 1024 // 1024} MB / {memory.total // 1024 // 1024} MB)"
        elif info_type.lower() == "cpu":
            cpu_percent = psutil.cpu_percent(interval=1)
            return f"CPU load: {cpu_percent}%"
        else:
            return "Available types: os, python, memory, cpu"
    except Exception as e:
        return f"Error getting information: {str(e)}"


def main():
    """Main function with examples of custom tools."""
    
    print("🛠️ Creating agent with custom tools...")
    
    # Create agent
    try:
        agent = OllamaAgent(
            model_name="gpt-oss:20b", 
            temperature=0.1,
            verbose=True
        )
        print("✅ Agent created!")
    except Exception as e:
        print(f"❌ Agent creation error: {e}")
        return
    
    # Create custom tools
    print("\n🔧 Creating custom tools...")
    
    # 1. Weather tool
    weather_tool = Tool(
        name="weather",
        description="Get weather information for the specified city. Use the city name in English or Russian.",
        func=weather_tool_function
    )
    
    # 2. Currency converter
    currency_tool = Tool(
        name="currency_converter",
        description="Currency conversion. Format: '100 USD to EUR'. Supports USD, EUR, RUB.",
        func=currency_converter_function
    )
    
    # 3. Joke generator
    joke_tool = Tool(
        name="random_joke",
        description="Get a random joke. Categories: general, programming, science.",
        func=random_joke_function
    )
    
    # 4. System information
    system_tool = Tool(
        name="system_info", 
        description="Get system information. Types: os, python, memory, cpu.",
        func=system_info_function
    )
    
    # Add tools to the agent
    print("➕ Adding tools to the agent...")
    
    try:
        agent.add_tool(weather_tool)
        agent.add_tool(currency_tool)
        agent.add_tool(joke_tool)
        agent.add_tool(system_tool)
        print("✅ All tools added!")
    except Exception as e:
        print(f"❌ Error adding tools: {e}")
        return
    
    # Show all available tools
    print(f"\n📋 Total tools: {len(agent.list_tools())}")
    for tool_name in agent.list_tools():
        print(f"  - {tool_name}")
    
    print("\n" + "="*60)
    
    # Testing custom tools
    print("\n🧪 Testing custom tools:\n")
    
    # Test 1: Weather
    print("🌤️ Test 1: Weather in Moscow")
    response1 = agent.run("What is the weather in Moscow?")
    print(f"Answer: {response1}")
    print("\n" + "-"*40)
    
    # Test 2: Currency converter  
    print("\n💱 Test 2: Currency conversion")
    response2 = agent.run("Convert 100 dollars to euros")
    print(f"Answer: {response2}")
    print("\n" + "-"*40)
    
    # Test 3: Joke
    print("\n😄 Test 3: Programming joke")
    response3 = agent.run("Tell a programming joke")
    print(f"Answer: {response3}")
    print("\n" + "-"*40)
    
    # Test 4: System information
    print("\n💻 Test 4: System information")
    response4 = agent.run("Show system information - operating system and Python version")
    print(f"Answer: {response4}")
    print("\n" + "-"*40)
    
    # Test 5: Combined query
    print("\n🎯 Test 5: Combined query")
    complex_query = """
    Help me:
    1. Get the weather in London
    2. Convert 50 euros to rubles  
    3. Tell a science joke
    4. Save all results to a file results.txt
    """
    response5 = agent.run(complex_query)
    print(f"Answer: {response5}")
    
    print("\n" + "="*60)
    print("✅ Testing completed!")
    
    # Demonstration of tool removal
    print("\n🗑️ Demonstration of tool removal...")
    print(f"Tools before removal: {len(agent.list_tools())}")
    
    agent.remove_tool("random_joke")
    print(f"Tools after removing 'random_joke': {len(agent.list_tools())}")
    
    remaining_tools = agent.list_tools()
    print("Remaining tools:")
    for tool_name in remaining_tools:
        print(f"  - {tool_name}")


if __name__ == "__main__":
    main()