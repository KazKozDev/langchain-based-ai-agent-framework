"""
Custom callbacks for enhanced observation and logging in LangChain agents.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage


class DetailedAgentCallbackHandler(BaseCallbackHandler):
    """Enhanced callback handler for detailed agent observation logging."""
    
    def __init__(self, logger: Optional[logging.Logger] = None, verbose: bool = True):
        """Initialize the callback handler."""
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.verbose = verbose
        self.step_count = 0
        self.current_tool = None
        self.current_input = None
    
    def on_agent_action(
        self,
        action: Any,
        color: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when agent performs an action."""
        self.step_count += 1
        self.current_tool = action.tool
        self.current_input = action.tool_input
        
        if self.verbose:
            self.logger.info(f"🎯 Step {self.step_count} - ACTION")
            self.logger.info(f"   Tool: {action.tool}")
            self.logger.info(f"   Input: {action.tool_input}")
            if hasattr(action, 'log') and action.log:
                self.logger.info(f"   Reasoning: {action.log}")
    
    def on_agent_finish(
        self,
        finish: Any,
        color: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when agent finishes."""
        if self.verbose:
            self.logger.info(f"✅ AGENT FINISHED")
            self.logger.info(f"   Final answer: {finish.return_values.get('output', 'No output')}")
    
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when tool starts."""
        if self.verbose:
            tool_name = serialized.get("name", "Unknown Tool")
            self.logger.info(f"🔧 TOOL START: {tool_name}")
            self.logger.info(f"   Input: {input_str}")
    
    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when tool ends - this is the OBSERVATION stage."""
        if self.verbose:
            self.logger.info(f"👁️ Step {self.step_count} - OBSERVATION")
            # Limit output length for readability
            display_output = output[:300] + "..." if len(output) > 300 else output
            self.logger.info(f"   Result: {display_output}")
            self.logger.info(f"   Tool: {self.current_tool} completed")
    
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when tool encounters an error."""
        if self.verbose:
            self.logger.error(f"❌ TOOL ERROR in {self.current_tool}: {error}")
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when LLM starts."""
        if self.verbose:
            self.logger.debug(f"🧠 LLM START")
            # Don't log full prompts as they can be very long
            self.logger.debug(f"   Prompt count: {len(prompts)}")
    
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when LLM ends."""
        if self.verbose:
            self.logger.debug(f"🧠 LLM END")
            if response.generations:
                # Show first generation only
                first_gen = response.generations[0][0].text if response.generations[0] else "No output"
                display_text = first_gen[:200] + "..." if len(first_gen) > 200 else first_gen
                self.logger.debug(f"   Response: {display_text}")
    
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when chain starts."""
        if self.verbose:
            chain_name = serialized.get("name", "Unknown Chain")
            self.logger.debug(f"🔗 CHAIN START: {chain_name}")
    
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called when chain ends."""
        if self.verbose:
            self.logger.debug(f"🔗 CHAIN END")
    
    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Any,
    ) -> Any:
        """Called on arbitrary text."""
        # This captures "Thought:" prefix text
        if self.verbose and text.strip() and "Thought:" in text:
            self.logger.info(f"💭 THOUGHT: {text.strip()}")


class SimpleObservationHandler(BaseCallbackHandler):
    """Simplified handler that focuses only on Action/Observation cycle."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.step = 0
    
    def on_agent_action(self, action: Any, **kwargs) -> Any:
        """Log agent action."""
        self.step += 1
        self.logger.info(f"🔍 Action {self.step}: {action.tool}({action.tool_input})")
    
    def on_tool_end(self, output: str, **kwargs) -> Any:
        """Log tool observation."""
        short_output = output[:150] + "..." if len(output) > 150 else output
        self.logger.info(f"👀 Observation {self.step}: {short_output}")