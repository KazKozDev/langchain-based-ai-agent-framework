"""
Tool manager for handling agent tools.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain_core.tools import Tool

from .tools.calculator import CalculatorTool
from .tools.search import get_search_tool
from .tools.file_manager import FileManagerTool
from .tools.datetime_tool import DateTimeTool
from .tools.webscraper import get_langchain_scraper_tool
from .tools.http_download import get_http_download_tool
from .tools.arxiv import (
    get_arxiv_tool,
    get_arxiv_search_tool,
    get_arxiv_versions_tool,
    get_arxiv_bibtex_tool,
    get_arxiv_pdf_info_tool,
)
from .rag.retrieval_tool import RAGRetrievalTool, RAGManagementTool


class ToolManager:
    """
    Manager class for handling agent tools.
    """
    
    def __init__(self, enable_rag: bool = True):
        """Initialize the tool manager with default tools."""
        self.logger = logging.getLogger(__name__)
        self._tools: Dict[str, Tool] = {}
        self.enable_rag = enable_rag
        
        # Initialize RAG system if enabled
        if self.enable_rag:
            try:
                self.rag_tool = RAGRetrievalTool()
                self.rag_management = RAGManagementTool(self.rag_tool)
            except Exception as e:
                self.logger.warning(f"Failed to initialize RAG system: {e}")
                self.enable_rag = False
                self.rag_tool = None
                self.rag_management = None
        else:
            self.rag_tool = None
            self.rag_management = None
        
        self._load_default_tools()
    
    def _load_default_tools(self):
        """Load default tools."""
        try:
            # Initialize default tools
            default_tools = [
                CalculatorTool().get_tool(),
                get_search_tool(),
                DateTimeTool().get_tool(),
                get_langchain_scraper_tool(),
                get_http_download_tool(),
                get_arxiv_tool(),
                get_arxiv_search_tool(),
                get_arxiv_versions_tool(),
                get_arxiv_bibtex_tool(),
                get_arxiv_pdf_info_tool(),
            ]
            # Add both legacy and structured file tools
            default_tools.extend(FileManagerTool().get_tools())
            
            # Add RAG tools if enabled
            if self.enable_rag and self.rag_tool and self.rag_management:
                default_tools.extend([
                    self.rag_tool.get_tool(),
                    self.rag_management.get_tool()
                ])
            
            for tool in default_tools:
                self._tools[tool.name] = tool
                
            self.logger.info(f"Loaded {len(default_tools)} default tools")
            
        except Exception as e:
            self.logger.error(f"Error loading default tools: {e}")
    
    def add_tool(self, tool: Tool):
        """
        Add a tool to the manager.
        
        Args:
            tool: LangChain Tool instance
        """
        if not isinstance(tool, Tool):
            raise ValueError("Tool must be an instance of langchain.tools.Tool")
        
        self._tools[tool.name] = tool
        self.logger.info(f"Tool added: {tool.name}")
    
    def remove_tool(self, tool_name: str):
        """
        Remove a tool from the manager.
        
        Args:
            tool_name: Name of the tool to remove
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            self.logger.info(f"Tool removed: {tool_name}")
        else:
            self.logger.warning(f"Tool not found: {tool_name}")
    
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """
        Get a specific tool by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(tool_name)
    
    def get_tools(self) -> List[Tool]:
        """
        Get all available tools.
        
        Returns:
            List of Tool instances
        """
        return list(self._tools.values())
    
    def list_tools(self) -> List[str]:
        """
        Get list of tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all tools.
        
        Returns:
            Dictionary mapping tool names to descriptions
        """
        return {name: tool.description for name, tool in self._tools.items()}
    
    def create_custom_tool(
        self, 
        name: str, 
        description: str, 
        function: callable,
        return_direct: bool = False
    ) -> Tool:
        """
        Create a custom tool.
        
        Args:
            name: Tool name
            description: Tool description
            function: Function to execute
            return_direct: Whether to return function result directly
            
        Returns:
            Tool instance
        """
        try:
            tool = Tool(
                name=name,
                description=description,
                func=function,
                return_direct=return_direct
            )
            self.logger.info(f"Custom tool created: {name}")
            return tool
        except Exception as e:
            self.logger.error(f"Error creating custom tool {name}: {e}")
            raise
    
    def tool_exists(self, tool_name: str) -> bool:
        """
        Check if a tool exists.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if tool exists, False otherwise
        """
        return tool_name in self._tools