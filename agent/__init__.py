"""
LangChain Agent with Ollama Integration
A modular AI agent system with tool calling capabilities.
"""

from .core import OllamaAgent
from .tool_manager import ToolManager

__version__ = "1.0.0"
__all__ = ["OllamaAgent", "ToolManager"]