# 📁 LangChain Agent Project Structure with RAG

## 🏗️ File Organization

```
langchain_agent/
├── 📋 README.md                    # Main project documentation
├── 📄 LICENSE                      # MIT license
├── 🖥️ simple_interactive.py        # Simple interactive shell ⭐
├── 🛠️ interactive.py               # Advanced interactive shell
├── 📖 USAGE_GUIDE.md               # Detailed usage guide
├── 📝 requirements.txt             # Python dependencies
├── ⚙️  setup.py                    # Setup file
├── 🚫 .gitignore                   # Git ignore rules
│
├── 🤖 agent/                       # MAIN AGENT MODULE
│   ├── __init__.py                 # Module initialization
│   ├── core.py                     # 🔧 Main OllamaAgent class
│   ├── cli.py                      # 📋 CLI interface
│   ├── tool_manager.py             # 🛠️  Tool manager
│   ├── callbacks.py                # 📞 Agent event handlers
│   │
│   ├── tools/                      # STANDARD TOOLS
│   │   ├── __init__.py
│   │   ├── calculator.py           # 🧮 Mathematical computations
│   │   ├── search.py               # 🔍 Web search (DuckDuckGo)
│   │   ├── file_manager.py         # 📁 File operations
│   │   ├── datetime_tool.py        # 📅 Date and time
│   │   └── webscraper.py           # 🌐 Web scraping
│   │
│   └── rag/                        # 🧠 RAG SYSTEM
│       ├── __init__.py             # RAG module exports
│       ├── vector_store.py         # 📊 Vector store (ChromaDB/FAISS)
│       ├── document_processor.py   # 📄 Document processing (PDF, DOCX, TXT...)
│       └── retrieval_tool.py       # 🔍 RAG tools (search + management)
│
├── 🧪 tests/                       # TEST FILES
│   ├── __init__.py
│   ├── test_rag_system.py          # 🏆 Full RAG testing
│   ├── quick_rag_test.py           # ⚡ Quick RAG test
│   ├── test_structured_rag.py      # 🔧 StructuredTool tests
│   ├── test_agent_rag_integration.py # 🔗 Integration tests
│   ├── test_all_tools.py           # 🛠️  All tools tests
│   ├── test_webscraper.py          # 🌐 Web scraper tests
│   ├── bitcoin_test.py             # ₿ Bitcoin function tests
│   └── final_test.py               # 🎯 Final tests
│
├── 📚 examples/                    # USAGE EXAMPLES
│   ├── basic_usage.py              # 🚀 Basic examples
│   ├── custom_tools.py             # 🔨 Custom tools
│   ├── rag_example.py              # 🧠 RAG examples
│   └── FINAL_RAG_DEMO.py           # 🎯 Full system demo
│
├── 📖 docs/                        # DOCUMENTATION
│   ├── RAG_DOCUMENTATION.md        # 📘 Detailed RAG documentation
│   └── original_README.md          # 📝 Original agent README
│
├── 🔧 scripts/                     # UTILITIES AND SCRIPTS
│   ├── clean_chromedriver.py       # 🧹 Clean ChromeDriver
│   ├── fix_chromedriver.py         # 🔧 Fix ChromeDriver
│   └── install_chromedriver_m1.sh  # 💻 Install ChromeDriver (M1 Mac)
│
├── ⚙️  config/                     # CONFIGURATION FILES
│   ├── agent_config.yaml           # 🤖 Agent settings
│   ├── ollama_config.yaml          # 🦙 Ollama settings
│   └── rag_config.yaml             # 🧠 RAG system settings
│
└── 💾 data/                       # PROJECT DATA (created automatically)
    ├── README.md                   # Data structure description
    ├── vector_store/               # Vector databases
    │   └── chroma/                 # ChromaDB data
    ├── logs/                       # Agent logs (optional)
    └── cache/                      # Embeddings cache (optional)
```

## 📋 File Categories

### 🔥 KEY FILES
- `agent/core.py` - Main agent class
- `agent/rag/retrieval_tool.py` - RAG tools
- `agent/rag/vector_store.py` - Vector DB
- `README.md` - Main documentation

### 🧪 TESTING
- `tests/quick_rag_test.py` - Quick RAG check
- `tests/test_rag_system.py` - Full RAG tests
- `examples/FINAL_RAG_DEMO.py` - Demonstration

### 📚 DOCUMENTATION
- `README.md` - Main documentation
- `docs/RAG_DOCUMENTATION.md` - RAG documentation  
- `PROJECT_STRUCTURE.md` - This file

### ⚙️  CONFIGURATION
- `requirements.txt` - Python dependencies
- `setup.py` - Package installation
- `config/*.yaml` - Component settings

## 🚀 Commands to Run

### Testing
```bash
# Quick RAG test (without Ollama)
python tests/quick_rag_test.py

# Full demo (requires Ollama)  
python examples/FINAL_RAG_DEMO.py

# All RAG tests
python tests/test_rag_system.py

# Integration tests
python tests/test_agent_rag_integration.py
```

### Usage
```bash
# Basic examples
python examples/basic_usage.py

# RAG examples
python examples/rag_example.py

# Custom tools
python examples/custom_tools.py
```

## 🧩 Component Architecture

### 🤖 Agent (agent/)
- **OllamaAgent**: Main agent class with Ollama integration
- **ToolManager**: Tool management with RAG support
- **Callbacks**: Event handlers and logging

### 🧠 RAG System (agent/rag/)
- **VectorStore**: Vector store (ChromaDB/FAISS)
- **DocumentProcessor**: Document processing (PDF, DOCX, TXT, MD, code)
- **RAGRetrievalTool**: Semantic search in the knowledge base
- **RAGManagementTool**: Document and collection management

### 🛠️ Tools (agent/tools/)
- **Calculator**: Mathematical computations
- **WebSearch**: Internet search  
- **FileManager**: File operations
- **DateTime**: Date/time operations
- **WebScraper**: Web page scraping

## 📊 Project Metrics

- **Total Python files**: ~25
- **Lines of code**: ~3000+
- **Test files**: 8
- **Examples**: 4
- **Tools**: 7 (including RAG)
- **Supported document formats**: 12+

## 🔄 Development Lifecycle

1. **Development** - Code in `agent/`
2. **Testing** - Tests in `tests/`
3. **Examples** - Demonstrations in `examples/`
4. **Documentation** - Descriptions in `docs/`
5. **Configuration** - Settings in `config/`

## 🎯 Entry Points

### For users
```python
from agent import OllamaAgent
agent = OllamaAgent()
```

### For RAG developers
```python  
from agent.rag import RAGRetrievalTool, VectorStore
rag = RAGRetrievalTool()
```

### For tool developers
```python
from agent.tool_manager import ToolManager
tools = ToolManager()
```

---

**The project is organized for maximum readability, testability, and extensibility! 🚀**