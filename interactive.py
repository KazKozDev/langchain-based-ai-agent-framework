#!/usr/bin/env python3
"""
langchain-based-ai-agent-framework - Interactive Shell

Entry point for interactive mode with comprehensive tool integration,
RAG capabilities, and unrestricted web search functionality.

Usage: python interactive.py [options]
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

from agent import OllamaAgent


class InteractiveAgent:
    """Interactive shell for the agent."""
    
    def __init__(self, model_name: str = "gpt-oss:20b", verbose: bool = True):
        """
        Initialize the interactive agent.
        
        Args:
            model_name: Ollama model name
            verbose: Verbose output
        """
        self.model_name = model_name
        self.verbose = verbose
        self.agent = None
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the agent."""
        print("Initializing langchain-based-ai-agent-framework...")
        print(f"Model: {self.model_name}")
        
        try:
            self.agent = OllamaAgent(
                model_name=self.model_name,
                temperature=0.1,
                verbose=self.verbose
            )
            
            tools = self.agent.list_tools()
            rag_tools = [tool for tool in tools if 'rag' in tool.lower()]
            
            print(f"Agent initialized successfully")
            print(f"Tools available: {len(tools)}")
            print(f"RAG tools: {len(rag_tools)}")
            
            if rag_tools:
                print(f"   • {', '.join(rag_tools)}")
            
            return True
            
        except Exception as e:
            print(f"Agent initialization error: {e}")
            print("\nMake sure that:")
            print("   • Ollama server is running (ollama serve)")
            print(f"   • Model {self.model_name} is pulled (ollama pull {self.model_name})")
            return False
    
    def show_help(self):
        """Show command help."""
        help_text = """
langchain-based-ai-agent-framework - Interactive Shell

BASIC COMMANDS:
  /help, /h          - Show this help
  /tools             - Show available tools  
  /clear             - Clear agent memory
  /quit, /exit, /q   - Exit the program

RAG COMMANDS:
  /rag info          - Knowledge base info
  /rag add <file>    - Add a file to the knowledge base
  /rag add_dir <dir> - Add a directory to the knowledge base
  /rag search <query> - Search the knowledge base
  /rag clear         - Clear the knowledge base

EXAMPLE QUERIES:
  • Calculate 15 * 25 + sqrt(144)
  • Find information about Python on the internet
  • What time is it in Moscow now?
  • Create a file test.txt with content "Hello World"
  • Find documents about machine learning in the knowledge base
  • Add README.md to the knowledge base

Just type your query or command!
"""
        print(help_text)
    
    def show_tools(self):
        """Show available tools."""
        if not self.agent:
            print("Agent is not initialized")
            return
        
        tools = self.agent.list_tools()
        descriptions = self.agent.get_tool_descriptions()
        
        print(f"\nAvailable tools ({len(tools)}):")
        print("=" * 60)
        
        for tool in tools:
            desc = descriptions.get(tool, "Description not available")
            # Shorten description to 80 characters
            short_desc = desc[:80] + "..." if len(desc) > 80 else desc
            print(f"{tool:<20} - {short_desc}")
        
        print("=" * 60)
    
    def handle_rag_command(self, command_parts: list):
        """Handle RAG commands."""
        if not self.agent:
            print("Agent is not initialized")
            return
        
        if len(command_parts) < 2:
            print("Incomplete RAG command. Use /help for guidance")
            return
        
        rag_action = command_parts[1].lower()
        
        try:
            if rag_action == "info":
                # Get knowledge base info
                result = self.agent.run("Get information about the knowledge base via the RAG management tool")
                print(f"Info: {result}")
                
            elif rag_action == "add" and len(command_parts) >= 3:
                # Add file
                file_path = " ".join(command_parts[2:])
                if not os.path.exists(file_path):
                    print(f"File not found: {file_path}")
                    return
                
                result = self.agent.run(f"Add file {file_path} to the knowledge base")
                print(f"File: {result}")
                
            elif rag_action == "add_dir" and len(command_parts) >= 3:
                # Add directory
                dir_path = " ".join(command_parts[2:])
                if not os.path.exists(dir_path):
                    print(f"Directory not found: {dir_path}")
                    return
                
                result = self.agent.run(f"Add all files from directory {dir_path} to the knowledge base")
                print(f"Directory: {result}")
                
            elif rag_action == "search" and len(command_parts) >= 3:
                # Search knowledge base
                query = " ".join(command_parts[2:])
                result = self.agent.run(f"Search in the knowledge base: {query}")
                print(f"Search: {result}")
                
            elif rag_action == "clear":
                # Clear knowledge base
                confirm = input("Are you sure you want to clear the entire knowledge base? (y/N): ")
                if confirm.lower() in ['y', 'yes']:
                    result = self.agent.run("Clear the entire knowledge base")
                    print(f"Cleared: {result}")
                else:
                    print("Operation cancelled")
            else:
                print(f"Unknown RAG command: {rag_action}")
                print("Available commands: info, add <file>, add_dir <dir>, search <query>, clear")
                
        except Exception as e:
            print(f"Error executing RAG command: {e}")
    
    def run(self):
        """Run the interactive shell."""
        if not self.agent:
            print("Failed to initialize the agent. Exiting.")
            return
        
        print("\nlangchain-based-ai-agent-framework interactive shell started!")
        print("Enter /help for help or /quit to exit\n")
        
        while True:
            try:
                # Input prompt
                user_input = input("Agent> ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['/quit', '/exit', '/q']:
                    print("Goodbye!")
                    break
                    
                elif user_input.lower() in ['/help', '/h']:
                    self.show_help()
                    
                elif user_input.lower() == '/tools':
                    self.show_tools()
                    
                elif user_input.lower() == '/clear':
                    if hasattr(self.agent, 'reset_memory'):
                        self.agent.reset_memory()
                        print("Agent memory cleared")
                    else:
                        print("Memory reset function is not available")
                    
                elif user_input.lower().startswith('/rag'):
                    command_parts = user_input.split()
                    self.handle_rag_command(command_parts)
                    
                else:
                    # Regular query to the agent
                    print("Processing your request...")
                    try:
                        response = self.agent.run(user_input)
                        print(f"Response: {response}\n")
                    except Exception as e:
                        print(f"Error processing request: {e}\n")
            
            except KeyboardInterrupt:
                print("\n\nExiting via Ctrl+C")
                break
            except EOFError:
                print("\n\nExiting via EOF")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                continue


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="langchain-based-ai-agent-framework - Interactive Shell")
    
    parser.add_argument(
        '--model', '-m', 
        default="gpt-oss:20b",
        help="Ollama model name (default: gpt-oss:20b)"
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help="Quiet mode (disable verbose output)"
    )
    
    parser.add_argument(
        '--test-connection',
        action='store_true',
        help="Only test connection to Ollama"
    )
    
    args = parser.parse_args()
    
    if args.test_connection:
        print("Testing connection to Ollama...")
        try:
            # Quick connection check
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json()
                print("Ollama server is available")
                print(f"Available models: {len(models.get('models', []))}")
                for model in models.get('models', [])[:5]:  # Show first 5
                    print(f"   • {model.get('name', 'Unknown')}")
                return
            else:
                print(f"Ollama server returned status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to connect to Ollama: {e}")
            print("Make sure Ollama is running: ollama serve")
        return
    
    # Start interactive shell
    verbose = not args.quiet
    interactive_agent = InteractiveAgent(model_name=args.model, verbose=verbose)
    interactive_agent.run()


if __name__ == "__main__":
    main()