#!/usr/bin/env python3
"""
🖥️  CLI module for LangChain Agent.
Provides commands to manage the agent and the RAG system.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional

from .core import OllamaAgent
from .rag.retrieval_tool import RAGRetrievalTool, RAGManagementTool


class AgentCLI:
    """CLI interface for LangChain Agent."""
    
    def __init__(self):
        """Initialize CLI."""
        self.agent = None
    
    def init_agent(self, model_name: str = "gpt-oss:20b", verbose: bool = False) -> bool:
        """
        Initialize the agent.
        
        Args:
            model_name: Ollama model name
            verbose: Verbose output
            
        Returns:
            True if successful, False if error
        """
        try:
            self.agent = OllamaAgent(
                model_name=model_name,
                temperature=0.1,
                verbose=verbose
            )
            return True
        except Exception as e:
            print(f"❌ Agent initialization error: {e}")
            return False
    
    def query(self, text: str) -> str:
        """
        Execute a query to the agent.
        
        Args:
            text: Query text
            
        Returns:
            Agent response
        """
        if not self.agent:
            return "❌ Agent is not initialized"
        
        try:
            return self.agent.run(text)
        except Exception as e:
            return f"❌ Error executing query: {e}"
    
    def list_tools(self) -> List[str]:
        """
        Get list of available tools.
        
        Returns:
            List of tool names
        """
        if not self.agent:
            return []
        
        return self.agent.list_tools()
    
    def rag_add_file(self, file_path: str) -> str:
        """
        Add a file to the RAG knowledge base.
        
        Args:
            file_path: Path to file
            
        Returns:
            Operation result
        """
        if not os.path.exists(file_path):
            return f"❌ File not found: {file_path}"
        
        return self.query(f"Add file {file_path} to the knowledge base")
    
    def rag_add_directory(self, dir_path: str, patterns: Optional[List[str]] = None) -> str:
        """
        Add a directory to the RAG knowledge base.
        
        Args:
            dir_path: Path to directory
            patterns: File patterns to include
            
        Returns:
            Operation result
        """
        if not os.path.exists(dir_path):
            return f"❌ Directory not found: {dir_path}"
        
        if patterns:
            pattern_str = ",".join(patterns)
            return self.query(f"Add files {pattern_str} from directory {dir_path} to the knowledge base")
        else:
            return self.query(f"Add all files from directory {dir_path} to the knowledge base")
    
    def rag_search(self, query: str, k: int = 5, with_scores: bool = False) -> str:
        """
        Search in the RAG knowledge base.
        
        Args:
            query: Search query
            k: Number of results
            with_scores: Include relevance scores
            
        Returns:
            Search results
        """
        search_query = f"Search the knowledge base (top {k}{' with scores' if with_scores else ''}): {query}"
        return self.query(search_query)
    
    def rag_info(self) -> str:
        """
        Get information about the RAG knowledge base.
        
        Returns:
            Collection information
        """
        return self.query("Get information about the knowledge base")
    
    def rag_clear(self) -> str:
        """
        Clear the RAG knowledge base.
        
        Returns:
            Operation result
        """
        return self.query("Clear the entire knowledge base")


def cmd_query(args):
    """Command to execute a query."""
    cli = AgentCLI()
    
    if not cli.init_agent(model_name=args.model, verbose=args.verbose):
        return 1
    
    if args.text:
        # Query from arguments
        query_text = " ".join(args.text)
    else:
        # Query from stdin
        query_text = sys.stdin.read().strip()
    
    if not query_text:
        print("❌ Empty query")
        return 1
    
    response = cli.query(query_text)
    print(response)
    return 0


def cmd_tools(args):
    """Command to list tools."""
    cli = AgentCLI()
    
    if not cli.init_agent(model_name=args.model, verbose=args.verbose):
        return 1
    
    tools = cli.list_tools()
    
    if args.json:
        print(json.dumps(tools, indent=2))
    else:
        print(f"📋 Available tools ({len(tools)}):")
        for tool in tools:
            print(f"  • {tool}")
    
    return 0


def cmd_rag_add(args):
    """Command to add files to RAG."""
    cli = AgentCLI()
    
    if not cli.init_agent(model_name=args.model, verbose=args.verbose):
        return 1
    
    if args.directory:
        # Add directory
        result = cli.rag_add_directory(args.path, patterns=args.patterns)
    else:
        # Add file
        result = cli.rag_add_file(args.path)
    
    print(result)
    return 0


def cmd_rag_search(args):
    """Command to search in RAG."""
    cli = AgentCLI()
    
    if not cli.init_agent(model_name=args.model, verbose=args.verbose):
        return 1
    
    query_text = " ".join(args.query)
    result = cli.rag_search(query_text, k=args.k, with_scores=args.scores)
    print(result)
    return 0


def cmd_rag_info(args):
    """Command to get RAG info."""
    cli = AgentCLI()
    
    if not cli.init_agent(model_name=args.model, verbose=args.verbose):
        return 1
    
    result = cli.rag_info()
    print(result)
    return 0


def cmd_rag_clear(args):
    """Command to clear RAG."""
    cli = AgentCLI()
    
    if not cli.init_agent(model_name=args.model, verbose=args.verbose):
        return 1
    
    if not args.force:
        confirm = input("⚠️  Are you sure you want to clear the entire knowledge base? (y/N): ")
        if confirm.lower() not in ['y', 'yes']:
            print("❌ Operation cancelled")
            return 0
    
    result = cli.rag_clear()
    print(result)
    return 0


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="🤖 LangChain Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  %(prog)s query "Calculate 2 + 2"
  %(prog)s tools --json
  %(prog)s rag add document.pdf
  %(prog)s rag add-dir ./docs --patterns "*.py" "*.md"
  %(prog)s rag search "machine learning" --k 3
  %(prog)s rag info
        """
    )
    
    # General arguments
    parser.add_argument(
        '--model', '-m',
        default='gpt-oss:20b',
        help='Ollama model name (default: gpt-oss:20b)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Command: query
    parser_query = subparsers.add_parser('query', help='Execute a query to the agent')
    parser_query.add_argument('text', nargs='*', help='Query text')
    parser_query.set_defaults(func=cmd_query)
    
    # Command: tools
    parser_tools = subparsers.add_parser('tools', help='Show available tools')
    parser_tools.add_argument('--json', action='store_true', help='Output in JSON format')
    parser_tools.set_defaults(func=cmd_tools)
    
    # RAG commands
    parser_rag = subparsers.add_parser('rag', help='Commands to manage the RAG system')
    rag_subparsers = parser_rag.add_subparsers(dest='rag_command', help='RAG commands')
    
    # RAG add
    parser_rag_add = rag_subparsers.add_parser('add', help='Add a file to the knowledge base')
    parser_rag_add.add_argument('path', help='Path to file or directory')
    parser_rag_add.add_argument('--directory', '-d', action='store_true', help='Treat path as a directory')
    parser_rag_add.add_argument('--patterns', nargs='*', help='File patterns for the directory')
    parser_rag_add.set_defaults(func=cmd_rag_add)
    
    # RAG search  
    parser_rag_search = rag_subparsers.add_parser('search', help='Search the knowledge base')
    parser_rag_search.add_argument('query', nargs='+', help='Search query')
    parser_rag_search.add_argument('--k', type=int, default=5, help='Number of results')
    parser_rag_search.add_argument('--scores', action='store_true', help='Include relevance scores')
    parser_rag_search.set_defaults(func=cmd_rag_search)
    
    # RAG info
    parser_rag_info = rag_subparsers.add_parser('info', help='Knowledge base info')
    parser_rag_info.set_defaults(func=cmd_rag_info)
    
    # RAG clear
    parser_rag_clear = rag_subparsers.add_parser('clear', help='Clear the knowledge base')
    parser_rag_clear.add_argument('--force', action='store_true', help='Do not ask for confirmation')
    parser_rag_clear.set_defaults(func=cmd_rag_clear)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        return 1
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())