# 🧠 RAG System Documentation

## Overview

The RAG (Retrieval-Augmented Generation) system enables the agent to access an external knowledge base via semantic search. The system includes a vector store, document processing, and retrieval tools.

## 🏗️ Architecture

```
RAG System
├── VectorStore          # Vector store (ChromaDB/FAISS)
├── DocumentProcessor    # Document processing
├── RAGRetrievalTool     # Retrieval tool
└── RAGManagementTool    # Management tool
```

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Initialize the agent with RAG

```python
from agent import OllamaAgent

agent = OllamaAgent()
print(agent.list_tools())  # Includes rag_retrieval and rag_management
```

### 3. Add documents

```python
# Add a file
agent.run("rag_management:add_file:/path/to/document.pdf")

# Add a directory
agent.run("rag_management:add_directory:/path/to/docs")

# Add text directly
agent.run("rag_management:add_text:Important information|title:Note")
```

### 4. Search the knowledge base

```python
# Simple search
agent.run("rag_retrieval:machine learning")

# Search with parameters
agent.run("rag_retrieval:query:Python development|k:5|with_scores:true")
```

## RAG Tools

### RAG Retrieval Tool (`rag_retrieval`)

**Purpose:** Semantic search in the knowledge base

**Input format:**
- Simple query: "search query"
- With parameters: "query:your_query|k:5|with_scores:true"

**Parameters:**
- `query` - search query (required)
- `k` - number of results (default: 5)
- `with_scores` - include relevance scores (true/false)
- `filter` - metadata filters

**Examples:**
```python
# Basic search
agent.run("rag_retrieval:vector database")

# Limited number of results
agent.run("rag_retrieval:query:LangChain tools|k:3")

# With relevance scores  
agent.run("rag_retrieval:query:Python AI|with_scores:true|k:2")
```

### RAG Management Tool (`rag_management`)

**Purpose:** Knowledge base management

**Available actions:**

#### Adding documents
```python
# Single file
agent.run("rag_management:add_file:/path/to/file.pdf")

# Multiple files
agent.run("rag_management:add_files:file1.txt,file2.pdf,file3.md")

# Directory
agent.run("rag_management:add_directory:/path/to/docs")

# Text directly
agent.run("rag_management:add_text:Document text|title:Title")
```

#### Information and management
```python
# Collection info
agent.run("rag_management:info")

# Clear collection
agent.run("rag_management:clear")
```

## Supported formats

- **Text:** `.txt`, `.md`, `.py`, `.js`, `.json`, `.yaml`, `.xml`, `.html`, `.css`, `.sql`
- **Documents:** `.pdf`, `.docx` (if libraries are available)

## Configuration

### Vector store configuration

```python
from agent.rag import VectorStore

# ChromaDB (default)
store = VectorStore(
    store_type="chroma",
    persist_directory="./vector_store",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# FAISS
store = VectorStore(
    store_type="faiss",
    persist_directory="./faiss_store"
)
```

### Document processing configuration

```python
from agent.rag import DocumentProcessor

processor = DocumentProcessor()

# Process a file
documents = processor.process_file("/path/to/file.txt")

# Process a directory with filters
documents = processor.process_directory(
    "/path/to/docs",
    recursive=True,
    file_patterns=["*.py", "*.md"],
    exclude_patterns=["__pycache__/*"]
)
```

## Testing

### Quick RAG system test

```bash
python quick_rag_test.py
```

### Full test

```bash
python test_rag_system.py
```

### Usage example

```bash
python examples/rag_example.py
```

## Advanced usage

### Direct usage of RAG tools

```python
from agent.rag import RAGRetrievalTool, RAGManagementTool

# Initialization
rag_tool = RAGRetrievalTool(store_type="chroma")
mgmt_tool = RAGManagementTool(rag_tool)

# Adding documents
result = mgmt_tool._manage_rag("add_text:Test document")

# Search
result = rag_tool._retrieve_documents("test query")
```

### Natural language with RAG

The agent can automatically use RAG when you mention searching for information:

```python
# These queries automatically use RAG
agent.run("Find information about machine learning in the documents")
agent.run("What does the knowledge base say about Python?")
agent.run("Search the documentation about vector databases")
```

## Document metadata

Each document contains metadata:

```python
{
    'source': '/path/to/file.txt',
    'filename': 'file.txt', 
    'file_extension': '.txt',
    'doc_id': 'unique_hash',
    'file_size': 1024,
    'created_at': timestamp,
    'modified_at': timestamp,
    'chunk_id': 'doc_hash_0',
    'chunk_index': 0,
    'total_chunks': 3,
    'parent_doc_id': 'parent_hash'
}
```

## Troubleshooting

### Dependency issues

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Verify sentence-transformers
python -c "from sentence_transformers import SentenceTransformer; print('OK')"
```

### ChromaDB issues

```bash
# Clear persistent store
rm -rf ./vector_store/chroma
```

### Embedding model issues

```python
# Force model download
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
```

## Best practices

1. **Chunk size:** Optimal size is 500-1500 characters
2. **Overlap:** Use 100-200 characters overlap
3. **Number of results:** 3-10 documents for context
4. **Filtering:** Use metadata to filter by type/category
5. **Cleanup:** Regularly remove outdated documents
6. **Monitoring:** Check relevance of search results

## Future improvements

- [ ] Support for re-ranking algorithms
- [ ] Hybrid search (dense + sparse)  
- [ ] Automatic retrieval quality evaluation
- [ ] Support for multimodal embeddings
- [ ] Integration with graph databases
- [ ] Caching of frequent queries