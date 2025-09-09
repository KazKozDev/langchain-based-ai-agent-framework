"""
Core agent implementation with Ollama LLM integration.
"""

import logging
import yaml
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_ollama.chat_models import ChatOllama as Ollama
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

from .tool_manager import ToolManager
from .callbacks import DetailedAgentCallbackHandler, SimpleObservationHandler


class OllamaAgent:
    """
    Main agent class with Ollama LLM integration and tool calling capabilities.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-oss:20b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        config_path: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the Ollama agent.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Ollama server URL
            temperature: LLM temperature setting
            config_path: Path to configuration file
            verbose: Enable verbose logging
        """
        self.logger = self._setup_logging()
        self.verbose = verbose
        
        # Load configuration if provided
        self.config = self._load_config(config_path) if config_path else {}
        
        # Override with provided parameters
        self.model_name = model_name or self.config.get('model_name', 'gpt-oss:20b')
        self.base_url = base_url or self.config.get('base_url', 'http://localhost:11434')
        self.temperature = temperature or self.config.get('temperature', 0.1)
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize tool manager with RAG support
        self.tool_manager = ToolManager(enable_rag=True)
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output",
        )
        
        # System message
        self.system_message = self._get_system_message()
        
        # Initialize callbacks for detailed observation
        self.callback_handler = DetailedAgentCallbackHandler(self.logger, self.verbose)
        
        # Initialize agent
        self.agent = None
        self._initialize_agent()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Remove existing handlers to prevent duplicate logs
        for h in list(logger.handlers):
            logger.removeHandler(h)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                self.logger.info(f"Configuration loaded from {config_path}")
                return config
            else:
                self.logger.warning(f"Configuration file not found: {config_path}")
                return {}
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return {}
    
    def _initialize_llm(self) -> Ollama:
        """Initialize Ollama LLM."""
        try:
            llm = Ollama(
                model=self.model_name,
                temperature=self.temperature,
                base_url=self.base_url,
                verbose=self.verbose
            )
            self.logger.info(f"Ollama LLM initialized with model: {self.model_name}")
            self.logger.debug(f"LLM object type: {type(llm)}")
            self.logger.debug(f"LLM object dir: {dir(llm)}")
            return llm
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama LLM: {e}")
            raise
    
    def _get_system_message(self) -> str:
        """Get system message for the agent."""
        default_message = """
You are a helpful AI assistant with access to various tools.
Use the available tools to accomplish the user's tasks.
Always explain which tool you are using and why.
If the required information is not available, say so honestly.
Respond in the user's language; if the user writes in Russian, respond in Russian.

 TOOLING RULES:
 - For downloading binary files (e.g., PDFs), use the `http_download` tool. Do NOT use the web scraper or web_search for direct PDF URLs.
 - For saving local files, prefer the structured `file_write` tool (args: path, content). Do NOT embed large JSON inside string commands.
 - For RAG ingestion, use `rag_management` (actions: add_text, add_file, add_directory). Avoid inventing abstracts; fetch abstracts from authoritative sources (e.g., arXiv abs pages) or cite the source.
 - Avoid running `web_search` on a direct URL. If a direct URL is given, act directly (download or open) instead of searching it.
        """.strip()
        
        return self.config.get('system_message', default_message)
    
    def _initialize_agent(self):
        """Initialize the LangChain agent."""
        try:
            tools = self.tool_manager.get_tools()
            
            # Create prompt template for tool calling
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_message),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad")
            ])
            
            # Bind tools to the LLM
            llm_with_tools = self.llm.bind_tools(tools)

            # Create tool calling agent
            agent = create_tool_calling_agent(
                llm=llm_with_tools,
                tools=tools,
                prompt=prompt
            )
            
            # Create agent executor with proper error handling
            self.agent = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=self.verbose,
                memory=self.memory,
                max_iterations=30,
                handle_parsing_errors=True,
                return_intermediate_steps=False,
                callbacks=[self.callback_handler] if self.verbose else []
            )
            
            self.logger.info(f"Agent initialized with {len(tools)} tools")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agent: {e}")
            raise
    
    def add_tool(self, tool: Tool):
        """Add a tool to the agent."""
        try:
            self.tool_manager.add_tool(tool)
            self._initialize_agent()  # Reinitialize agent with new tools
            self.logger.info(f"Tool added: {tool.name}")
        except Exception as e:
            self.logger.error(f"Failed to add tool {tool.name}: {e}")
            raise
    
    def remove_tool(self, tool_name: str):
        """Remove a tool from the agent."""
        try:
            self.tool_manager.remove_tool(tool_name)
            self._initialize_agent()  # Reinitialize agent without removed tool
            self.logger.info(f"Tool removed: {tool_name}")
        except Exception as e:
            self.logger.error(f"Failed to remove tool {tool_name}: {e}")
            raise
    
    def list_tools(self) -> List[str]:
        """Get list of available tool names."""
        return self.tool_manager.list_tools()
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all available tools."""
        return self.tool_manager.get_tool_descriptions()
    
    def process_query(self, query: str) -> str:
        """
        Run the agent with a query.
        
        Args:
            query: User query/prompt
            
        Returns:
            Agent response
        """
        try:
            self.logger.info(f"Processing query: {query[:50]}...")
            
            # Use invoke method with proper input format
            result = self.agent.invoke({
                "input": query,
                "chat_history": self.memory.chat_memory.messages if self.memory else []
            })
            
            # Log intermediate steps for observation visibility
            if self.verbose and "intermediate_steps" in result:
                for i, (action, observation) in enumerate(result["intermediate_steps"], 1):
                    self.logger.info(f"Step {i} - Action: {action.tool} with input: {action.tool_input}")
                    self.logger.info(f"Step {i} - Observation: {str(observation)[:200]}...")
            
            response = result.get("output", "Error: failed to get a response")
            self.logger.info("Query processed successfully")
            
            return response
        except Exception as e:
            error_msg = f"Error processing query: {e}"
            self.logger.error(error_msg)
            return f"Sorry, an error occurred while processing the request: {e}"
    
    def run(self, query: str) -> str:
        """Backward-compatible alias for process_query()."""
        return self.process_query(query)
    
    def reset_memory(self):
        """Reset conversation memory."""
        self.memory.clear()
        self.logger.info("Memory reset")
    
    def get_memory(self) -> List[Dict]:
        """Get conversation history."""
        return self.memory.chat_memory.messages