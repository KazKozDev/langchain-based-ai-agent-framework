# LangChain Agent with Ollama

Modular LangChain-based AI agent with Ollama integration for tool calling functionality.

## 🚀 Features

- **Ollama integration**: Support for local LLM models
- **Modern architecture**: Uses `create_tool_calling_agent` and `AgentExecutor`
- **Modular system**: Easy addition and removal of tools
- **Built-in tools**: Calculator, search, file manager, date/time, web scraping
- **Robust error handling**: `handle_parsing_errors=True` and full logging
- **YAML configuration**: Flexible agent setup
- **Conversation memory**: Keeps context across queries

## 📋 Requirements

- Python 3.8+
- Ollama server with the `gpt-oss:20b` model installed
- Libraries from `requirements.txt`

## 🔧 Installation

1. **Clone the project**:
   ```bash
   git clone <repository-url>
   cd langchain_agent
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and configure Ollama**:
   ```bash
   # Install Ollama (macOS)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama
   ollama serve
   
   # Pull model (in a new terminal)
   ollama pull gpt-oss:20b
   ```

4. **Configure** (optional):
   - Edit `config/agent_config.yaml` to configure the agent
   - Edit `config/ollama_config.yaml` to configure Ollama

## 🎯 Quick Start

### Basic usage

```python
from agent import OllamaAgent

# Create the agent
agent = OllamaAgent(
    model_name="gpt-oss:20b",
    temperature=0.1,
    verbose=True
)

# Use the agent
response = agent.run("Calculate 15 * 25 + 100")
print(response)
```

### Running examples

```bash
# Quick test (simple queries without search)
python quick_test.py

# Test all tools with detailed observation
python test_all_tools.py

# Test webscraper
python test_webscraper.py

# Full test
python test_agent.py

# Basic examples
python examples/basic_usage.py

# Custom tools
python examples/custom_tools.py
```

## 🛠️ Available Tools

### 1. Calculator (`calculator`)
Performs mathematical computations:
- Basic operations: `+`, `-`, `*`, `/`, `**`, `%`
- Parentheses for grouping
- Functions: `sqrt()`, `abs()`

**Example**: `"Calculate (5*3)**2 + sqrt(16)"`

### 2. Web Search (`web_search`)  
Search for up-to-date information on the internet:
- Uses DuckDuckGo
- Fetch definitions and short answers
- Related topics and sources

**Example**: `"Find information about Python"`

### 3. File Manager (`file_manager`)
Text file operations:
- `read:/path/to/file` - read file
- `write:/path/to/file:content` - write to file
- `append:/path/to/file:content` - append to file  
- `list:/path/to/directory` - list directory contents

**Example**: `"Create file test.txt with text 'Hello!'"`

### 4. Date and Time (`datetime`)
Date and time operations:
- `now` - current time
- `date` - current date
- `timezone:Europe/Moscow` - time in a timezone
- `add_days:5` - date after N days

**Example**: `"What time is it in Moscow?"`

## 🔨 Creating Custom Tools

```python
from langchain.tools import Tool

# Create tool function
def my_tool_function(input_data: str) -> str:
    return f"Result: {input_data}"

# Create tool
my_tool = Tool(
    name="my_tool",
    description="Description of what the tool does",
    func=my_tool_function
)

# Add to agent
agent.add_tool(my_tool)
```

## 📊 API Reference

### OllamaAgent

#### Initialization
```python
agent = OllamaAgent(
    model_name="gpt-oss:20b",      # Ollama model
    base_url="http://localhost:11434",  # Ollama server URL
    temperature=0.1,               # LLM temperature
    config_path=None,               # Path to config
    verbose=True                    # Verbose output
)
```

#### Primary methods
- `run(query: str) -> str` - Execute a query
- `add_tool(tool: Tool)` - Add a tool
- `remove_tool(tool_name: str)` - Remove a tool
- `list_tools() -> List[str]` - List tools
- `reset_memory()` - Clear memory
- `get_memory() -> List[Dict]` - Get history

### ToolManager

#### Primary methods
- `add_tool(tool: Tool)` - Add a tool
- `remove_tool(tool_name: str)` - Remove a tool
- `get_tools() -> List[Tool]` - Get all tools
- `create_custom_tool(name, description, function)` - Create a tool

## ⚙️ Configuration

### agent_config.yaml
```yaml
model_name: "gpt-oss:20b"
temperature: 0.1
verbose: true
system_message: |
  You are a helpful AI assistant...

tools:
  enabled:
    - calculator
    - web_search
    - file_manager
    - datetime
```

### ollama_config.yaml  
```yaml
server:
  host: "localhost"
  port: 11434
  
models:
  primary: "gpt-oss:20b"
  fallback: "llama2:7b"
```

## 🔍 Troubleshooting

### Ollama connection issues

1. **Check if Ollama is running**:
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. **Check model availability**:
   ```bash
   ollama list
   ```

3. **Pull model if missing**:
   ```bash
   ollama pull gpt-oss:20b
   ```

### Dependency issues

```bash
# Update pip
pip install --upgrade pip

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Tool issues

- **"Too many arguments to single-input tool"**: Fixed in the latest version
- **Search not working**:
  - Check internet connection
  - Ensure `selenium` and `webdriver-manager` are installed
  - First run may take time to download ChromeDriver
- **Files not readable**: Check file permissions
- **Calculation errors**: Verify expression syntax
- **"OUTPUT_PARSING_FAILURE"**: Resolved by switching to modern LangChain agents

## 📝 Logging

Logs are printed to the console at INFO level by default. To change the level:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Add tests
5. Open a Pull Request

## 📄 License

MIT License. See LICENSE for details.

## 🆘 Support

If you have questions or issues:

1. Check the Troubleshooting section
2. Explore examples in the `examples/` folder
3. Open an issue in the repository

## 🎉 Usage Examples

### Creating a report
```python
query = """
Create a report:
1. Compute 365 * 24 (hours in a year)
2. Get the current date
3. Find information about time zones
4. Save everything to report.txt
"""
response = agent.run(query)
```

### Data analysis
```python
response = agent.run(
    "Read the file data.csv and compute the average of numbers in the first column"
)
```

### Task planning
```python
response = agent.run(
    "What will the date be in 30 days? Create a reminder in reminder.txt"
)
```