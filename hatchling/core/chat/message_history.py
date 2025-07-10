"""Message history management module for chat interfaces.

Provides functionality for tracking, storing, and managing chat message history
including user messages, assistant responses, and tool interactions.
"""

from typing import List, Dict, Any, Optional
from hatchling.core.logging.session_debug_log import SessionDebugLog

class MessageHistory:
    """Simple manager for chat message history without any complex optimizations."""
    
    def __init__(self, debug_log: Optional[SessionDebugLog] = None):
        """Initialize an empty message history.
        
        Args:
            debug_log (Optional[SessionDebugLog]): Optional logger for debug information.
        """
        self.messages: List[Dict[str, Any]] = []
        self.logger = debug_log
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to the history.
        
        Args:
            content (str): The message content.
        """
        self.messages.append({"role": "user", "content": content})
        if self.logger:
            self.logger.debug(f"MessageHistory - Added user message: {content}")
    
    def add_assistant_message(self, content: str, tool_calls: List[Dict[str, Any]] = None, provider: str = None) -> None:
        """Add an assistant message to the history.
        
        Args:
            content (str): The message content.
            tool_calls (List[Dict[str, Any]], optional): Optional list of tool calls.
            provider (str, optional): LLM provider ("openai" or "ollama").
        """
        if provider == "openai" or not tool_calls:
            self.messages.append({"role": "assistant", "content": content})
        else:
            self.messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls
            })
        
        if self.logger:
            self.logger.debug(f"MessageHistory - Added assistant message: {content}")
    
    def add_tool_result(self, tool_call_id: str, function_name: str, content: str, provider: str = None) -> None:
        """Add a tool result to the history.
        
        Args:
            tool_call_id (str): ID of the tool call this result is for.
            function_name (str): Name of the function that was called.
            content (str): The tool result content.
            provider (str, optional): LLM provider ("openai" or "ollama").
        """
        role = "function" if provider == "openai" else "tool"
        msg = {
            "role": role,
            "name": function_name,
            "content": content,
            "tool_call_id": tool_call_id,
            "type": "function"
        }
        self.messages.append(msg)
        if self.logger:
            self.logger.debug(f"MessageHistory - Added tool result for {function_name}: {content}")
    
    def update_message_history(self, full_response: str, message_tool_calls: List[Dict[str, Any]], 
                               tool_results: List[Dict[str, Any]], provider: str = None) -> None:
        """Update the message history with the response and tool results.
        
        Args:
            full_response (str): The complete response from the LLM.
            message_tool_calls (List[Dict[str, Any]]): List of tool calls from the response.
            tool_results (List[Dict[str, Any]]): List of tool execution results.
            provider (str, optional): LLM provider ("openai" or "ollama").
        """
        # Only update if we got a response or tool results
        if not full_response.strip() and not tool_results:
            return
            
        # First add the assistant message
        if message_tool_calls and provider != "openai":
            self.add_assistant_message(full_response, message_tool_calls, provider=provider)
        else:
            self.add_assistant_message(full_response, provider=provider)
        
        # Add all tool results to the message history
        for tool_result in tool_results:
            self.add_tool_result(
                tool_result["tool_call_id"],
                tool_result["name"],
                tool_result["content"],
                provider=provider
            )
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages.
        
        Returns:
            List[Dict[str, Any]]: List of message dictionaries.
        """
        return self.messages
    
    def get_last_user_message(self) -> Optional[str]:
        """Get the content of the last user message.
        
        Returns:
            Optional[str]: Content of the last user message, or None if not found.
        """
        for message in reversed(self.messages):
            if message["role"] == "user":
                return message["content"]
        return None
    
    def replace_last_assistant_message(self, content: str, tool_calls: List[Dict[str, Any]] = None) -> None:
        """Replace the last assistant message.
        
        Args:
            content (str): The new content.
            tool_calls (List[Dict[str, Any]], optional): Optional tool calls to include.
        """
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i]["role"] == "assistant":
                if tool_calls:
                    self.messages[i] = {
                        "role": "assistant", 
                        "content": content,
                        "tool_calls": tool_calls
                    }
                else:
                    self.messages[i] = {"role": "assistant", "content": content}
                return
    
    def copy(self) -> 'MessageHistory':
        """Create a copy of this message history.
        
        Returns:
            MessageHistory: A new MessageHistory with the same messages.
        """
        new_history = MessageHistory(self.logger)
        new_history.messages = self.messages.copy()
        return new_history
    
    def clear(self) -> None:
        """Clear all messages."""
        self.messages = []
        if self.logger:
            self.logger.info("MessageHistory - Cleared!")
    
    def __len__(self) -> int:
        """Get the number of messages.
        
        Returns:
            int: The number of messages in the history.
        """
        return len(self.messages)