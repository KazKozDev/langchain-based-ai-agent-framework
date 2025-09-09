# LangChain Ollama Agent

A production-ready LangChain-based AI agent framework with comprehensive tool integration, RAG capabilities, and unrestricted web search functionality.

## Overview

This project demonstrates advanced LangChain implementation patterns through a modular agent architecture that combines local LLM inference via Ollama with powerful information retrieval and processing tools. The agent serves as a comprehensive research and automation platform with particular strengths in academic research, web search, and document processing.

## Target Audience

- **AI Engineers** seeking production-ready LangChain implementations
- **Researchers** requiring robust information retrieval from multiple sources
- **Developers** building AI-powered automation tools
- **Data Scientists** needing RAG-enabled document processing pipelines

## Key Features & Differentiators

### Advanced Search Capabilities
- **Unrestricted Web Search**: Custom Selenium-based DuckDuckGo implementation bypasses typical API limitations and rate limits
- **Academic Research Integration**: Direct arXiv.org API integration with metadata extraction, search, and PDF handling

### Production-Ready LangChain Architecture
- **Modular Tool System**: Extensible tool manager with automatic registration and dynamic loading
- **Memory Management**: Persistent conversation memory with configurable buffer strategies  
- **Error Handling**: Comprehensive exception handling with retry mechanisms and graceful degradation
- **Callback System**: Custom callback handlers for detailed execution monitoring

### RAG Implementation
- **Multi-Vector Store Support**: Chroma and FAISS backends with automatic optimization
- **Document Processing Pipeline**: Support for PDF, DOCX, TXT, Markdown with intelligent chunking
- **Semantic Search**: Sentence-transformers embeddings with configurable similarity thresholds
- **Knowledge Base Management**: Add/remove/query operations with persistence

### Tool Ecosystem
- **Web Scraping**: BeautifulSoup-based content extraction with robust parsing
- **File Operations**: Comprehensive file system tools with safety constraints
- **HTTP Downloads**: Streaming downloads with progress tracking for large files
- **System Integration**: Date/time handling, system information, and process management

## Technical Architecture

Built on modern LangChain patterns with emphasis on:
- **Tool Calling Agents**: Utilizes LangChain's tool-calling architecture
- **Structured Tools**: Pydantic-based tool definitions with type safety
- **Agent Executors**: Configurable execution with intermediate step tracking
- **Prompt Engineering**: Optimized system prompts with multilingual support

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama server
ollama serve

# Pull a model (e.g., gpt-oss:20b)
ollama pull gpt-oss:20b

# Run interactive shell
python interactive.py
```

## Configuration

The agent supports extensive configuration through YAML files:
- `config/agent_config.yaml` - Core agent settings
- `config/rag_config.yaml` - RAG system configuration  
- `config/ollama_config.yaml` - LLM model parameters

## Example Usage

```python
from agent import OllamaAgent

# Initialize agent
agent = OllamaAgent(model_name="gpt-oss:20b", verbose=True)

# Web search
response = agent.run("Find recent papers about transformer architectures")

# RAG operations
agent.run("Add the research papers directory to knowledge base")
agent.run("What are the key findings about attention mechanisms?")

# Academic research
agent.run("Search arXiv for 'large language models' and summarize top 3 papers")
```

## Project Structure

```
langchain_agent/
├── agent/                      # Core agent implementation
│   ├── core.py                 # Main OllamaAgent class
│   ├── tool_manager.py         # Tool management system
│   ├── callbacks.py            # Custom callback handlers
│   └── tools/                  # Tool implementations
│       ├── search.py           # Web search (DuckDuckGo)
│       ├── arxiv.py            # arXiv integration
│       ├── webscraper.py       # Web scraping
│       ├── file_manager.py     # File operations
│       └── calculator.py       # Mathematical tools
├── config/                     # Configuration files
│   ├── rag_config.yaml         # RAG system settings
│   ├── agent_config.yaml       # Agent configuration
│   └── ollama_config.yaml      # LLM parameters
├── tests/                      # Comprehensive test suite
├── examples/                   # Usage examples
└── interactive.py              # Interactive shell
```

## Available Tools

- **web_search** - DuckDuckGo search with anti-detection measures
- **arxiv_metadata** - arXiv paper metadata and abstract extraction
- **arxiv_search** - Academic paper search via arXiv API
- **advanced_web_scraper** - Content extraction with BeautifulSoup
- **http_download** - Streaming file downloads
- **rag_retrieval** - Semantic search in knowledge base
- **rag_management** - Document ingestion and management
- **file_manager** - File system operations
- **calculator** - Mathematical computations
- **datetime_tool** - Date and time operations

## RAG System Features

- **Multiple Vector Stores**: ChromaDB and FAISS support
- **Document Processing**: PDF, DOCX, TXT, Markdown, code files
- **Semantic Chunking**: Configurable chunk size and overlap
- **Metadata Extraction**: Automatic document metadata parsing
- **Query Expansion**: Advanced retrieval strategies
- **Persistence**: Automatic vector store persistence

## Development Status

This implementation showcases:
- Advanced LangChain framework utilization
- Custom tool development and integration
- Production-ready error handling and logging
- Modular architecture for easy extension
- Comprehensive test coverage

The codebase serves as a practical demonstration of LangChain's capabilities while solving real-world information retrieval and processing challenges.

## Installation

```bash
git clone <repository-url>
cd langchain_agent
pip install -r requirements.txt

# Setup Ollama
ollama serve
ollama pull gpt-oss:20b
```

## Testing

```bash
# Quick RAG test
python tests/quick_rag_test.py

# Full demonstration
python examples/FINAL_RAG_DEMO.py

# Integration tests
python tests/test_agent_rag_integration.py
```

## License

MIT License. See LICENSE for details.