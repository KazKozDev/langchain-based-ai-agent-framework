# ⚡ Quick Start - LangChain Agent

## 🚀 Fastest Start

```bash
# 1. Make sure Ollama is running and the model is pulled
ollama serve
ollama pull gpt-oss:20b

# 2. Start the simple interactive shell
python simple_interactive.py
```

## 💬 Usage Example

```
🚀 Starting agent in interactive mode...
✅ Agent loaded successfully. Type 'exit' or 'quit' to end the session.
--------------------------------------------------
👤 You: Hi! Calculate 15 * 25 + 100
🤖 Agent: Hi! Let's calculate: 15 * 25 + 100 = 375 + 100 = 475

👤 You: What time is it in Moscow now?
🤖 Agent: Current time in Moscow: [current time with timezone]

👤 You: Find information about Python on the internet
🤖 Agent: [web search result about Python]

👤 You: Create a file hello.txt with content "Hello World"
🤖 Agent: File hello.txt created successfully with content "Hello World"

👤 You: exit
👋 Goodbye!
```

## 🛠️ Available Capabilities

### Built-in tools:
- 🧮 **Calculator** - mathematical computations
- 🔍 **Web Search** - internet search
- 📁 **File Manager** - file operations
- 🕐 **DateTime** - date and time operations
- 🌐 **Web Scraper** - web page data extraction
- 🧠 **RAG Retrieval** - knowledge base search
- 📚 **RAG Management** - document management

### Example queries:
```
👤 You: Compute the integral of x^2 from 0 to 5
👤 You: Find the latest news about AI
👤 You: What is the weather tomorrow in Paris?
👤 You: Create a JSON file with user data
👤 You: Add the README.md file to the knowledge base
👤 You: Find information about vector databases in the documents
```

## 🔄 Other Ways to Run

### Advanced interactive shell
```bash
python interactive.py
# Commands: /help, /tools, /rag add file.pdf, /rag search query
```

### CLI commands
```bash
python -m agent.cli query "Calculate 2+2"
python -m agent.cli rag add document.pdf
```

### Python API
```python
from agent import OllamaAgent
agent = OllamaAgent()
response = agent.run("Hello!")
```

## 🆘 If something doesn't work

1. **Check Ollama:**
   ```bash
   python interactive.py --test-connection
   ```

2. **Check the model:**
   ```bash
   ollama list
   ```

3. **Quick RAG test:**
   ```bash
   python tests/quick_rag_test.py
   ```

---

🎯 **Just run `python simple_interactive.py` and start chatting with the agent!**