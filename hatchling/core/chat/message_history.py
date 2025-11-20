"""Message history management module for chat interfaces.

Provides functionality for tracking, storing, and managing chat message history
including user messages, assistant responses, and tool interactions.
"""

from typing import List, Dict, Any, Optional
import json
from hatchling.core.logging.logging_manager import logging_manager
from hatchling.core.llm.providers.registry import ProviderRegistry
from hatchling.core.llm.event_system import EventSubscriber, Event, EventType
from hatchling.config.llm_settings import ELLMProvider

from hatchling.core.llm.data_structures import ToolCallParsedResult, ToolCallExecutionResult
from hatchling.config.settings import AppSettings

class MessageHistory(EventSubscriber):
    """Event-driven message history manager with canonical and provider-specific histories.
    
    Maintains a canonical (provider-agnostic) history and dynamically generates
    provider-specific histories based on the current LLM provider.
    """
    
    def __init__(self, settings: AppSettings = None):
        """Initialize an empty message history with dual-history support."""
        self.settings = settings or AppSettings.get_instance()
        # Canonical history storing all events in normalized format
        self.canonical_history: List[Dict[str, Any]] = []
        
        # Provider-specific history generated on demand
        self.provider_history: List[Dict[str, Any]] = []
        
        # Current provider tracking for regeneration
        self._current_provider: Optional[ELLMProvider] = None
        
        # Content buffer for assistant message assembly
        self._content_buffer: str = ""
        
        self.logger = logging_manager.get_session("MessageHistory")
    
    def get_subscribed_events(self) -> List[EventType]:
        """Return list of event types this subscriber handles.
        
        Returns:
            List[EventType]: Event types for message history management.
        """
        return [
            # LLM Response Events
            EventType.CONTENT,
            EventType.FINISH,
            # Tool Execution Events
            EventType.MCP_TOOL_CALL_DISPATCHED,
            EventType.MCP_TOOL_CALL_RESULT,
            EventType.MCP_TOOL_CALL_ERROR,
        ]
    
    def on_event(self, event: Event) -> None:
        """Handle stream events and update canonical history.
        
        Args:
            event (Event): The event to handle.
        """
        try:
            # Check for provider change and regenerate provider history if needed
            if event.provider != self._current_provider:
                self._current_provider = event.provider
                self._regenerate_provider_history()
                self.logger.debug(f"Provider changed to {event.provider}, regenerated provider history")
            
            if event.type == EventType.CONTENT:
                self._handle_content_event(event)
            elif event.type == EventType.FINISH:
                self._handle_finish_event(event)
            elif event.type == EventType.MCP_TOOL_CALL_DISPATCHED:
                self._handle_tool_call_dispatched_event(event)
            elif event.type == EventType.MCP_TOOL_CALL_RESULT:
                self._handle_tool_call_result_event(event)
            elif event.type == EventType.MCP_TOOL_CALL_ERROR:
                self._handle_tool_call_error_event(event)
                
        except Exception as e:
            self.logger.error(f"Error handling event {event.type}: {e}")
    
    def _handle_content_event(self, event: Event) -> None:
        """Handle CONTENT events by buffering content for assistant message assembly.
        
        Args:
            event (Event): The CONTENT event.
        """
        content = event.data.get("content", "")
        self._content_buffer += content
        #self.logger.debug(f"Buffered content: {len(content)} chars (total buffer: {len(self._content_buffer)})")
    
    def _handle_finish_event(self, event: Event) -> None:
        """Handle FINISH events by finalizing assistant message from buffer.
        
        Args:
            event (Event): The FINISH event.
        """
        if self._content_buffer:
            # Add complete assistant message to canonical history
            canonical_entry = {
                "type": "assistant",
                "data": {
                    "role": "assistant",
                    "content": self._content_buffer
                }
            }
            self.canonical_history.append(canonical_entry)
            
            # Add to provider-specific history
            provider_entry = {"role": "assistant", "content": self._content_buffer}
            self.provider_history.append(provider_entry)
            
            self.logger.debug(f"Added assistant message: {len(self._content_buffer)} chars")
            self._content_buffer = ""  # Reset buffer

    def _handle_tool_call_dispatched_event(self, event: Event) -> None:
        """Handle MCP_TOOL_CALL_DISPATCHED events by adding tool calls to history.
        
        Args:
            event (Event): The MCP_TOOL_CALL_DISPATCHED event.
        """
        # Create ToolCallParsedResult from event data
        tool_call = ToolCallParsedResult(
            tool_call_id=event.data.get("tool_call_id", ""),
            function_name=event.data.get("function_name", ""),
            arguments=event.data.get("arguments", {})
        )
        
        # Add to canonical history
        canonical_entry = {
            "type": "tool_call",
            "data": tool_call
        }
        self.canonical_history.append(canonical_entry)

        provider_entry = {
            "role": "assistant",
            "tool_calls": [ProviderRegistry.get_provider(self._current_provider).hatchling_to_llm_tool_call(tool_call)]
        }
        
        self.provider_history.append(provider_entry)
        
        self.logger.debug(f"Added tool call: {tool_call.function_name}")
    
    def _handle_tool_call_result_event(self, event: Event) -> None:
        """Handle MCP_TOOL_CALL_RESULT events by adding tool results to history.
        
        Args:
            event (Event): The MCP_TOOL_CALL_RESULT event.
        """
        # Create ToolCallExecutionResult from event data  
        tool_result = ToolCallExecutionResult(**event.data)
        
        # Add to canonical history
        canonical_entry = {
            "type": "tool_result",
            "data": tool_result
        }
        self.canonical_history.append(canonical_entry)

        provider_entry = {
            "role": "tool",
            **ProviderRegistry.get_provider(self._current_provider).hatchling_to_provider_tool_result(tool_result)
        }
        
        self.provider_history.append(provider_entry)
        
        self.logger.debug(f"Added tool result for: {tool_result.function_name}")
    
    def _handle_tool_call_error_event(self, event: Event) -> None:
        """Handle MCP_TOOL_CALL_ERROR events by adding error results to history.
        
        Args:
            event (Event): The MCP_TOOL_CALL_ERROR event.
        """
        # Create ToolCallExecutionResult with error from event data
        tool_result = ToolCallExecutionResult(**event.data)
        
        # Add to canonical history
        canonical_entry = {
            "type": "tool_result",
            "data": tool_result
        }
        self.canonical_history.append(canonical_entry)
        
        # Add to provider-specific history based on current provider
        provider_entry = {
            "role": "tool",
            **ProviderRegistry.get_provider(self._current_provider).hatchling_to_provider_tool_result(tool_result)
        }
        
        self.provider_history.append(provider_entry)
        
        self.logger.debug(f"Added tool error for: {tool_result.function_name}")
    
    def _regenerate_provider_history(self) -> None:
        """Regenerate provider-specific history from canonical history."""
        self.provider_history = []
        
        # Determine the provider to use for formatting
        provider_to_use = self._current_provider
        if provider_to_use is None:
            provider_to_use = self.settings.llm.provider_enum
            self.logger.debug(f"_current_provider is None, using default provider from settings: {provider_to_use}")

        for entry in self.canonical_history:
            entry_type = entry["type"]
            
            if entry_type == "user":
                provider_entry = entry["data"]
            elif entry_type == "assistant":
                provider_entry = entry["data"]
            elif entry_type == "tool_call":
                tool_call = entry["data"]
                provider_entry = {
                    "role": "assistant",
                    "tool_calls": [ProviderRegistry.get_provider(provider_to_use).hatchling_to_llm_tool_call(tool_call)]
                }
            elif entry_type == "tool_result":
                tool_result = entry["data"]
                provider_entry = {
                    "role": "tool",
                    **ProviderRegistry.get_provider(provider_to_use).hatchling_to_provider_tool_result(tool_result)
                }
            else:
                continue  # Skip unknown entry types
            
            self.provider_history.append(provider_entry)
        
        self.logger.debug(f"Regenerated provider history: {len(self.provider_history)} entries")
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to the history.
        
        Args:
            content (str): The message content.
        """
        # Add to canonical history
        canonical_entry = {
            "type": "user",
            "data": {
                "role": "user",
                "content": content
            }
        }
        self.canonical_history.append(canonical_entry)
        
        # Add to provider-specific history
        provider_entry = {"role": "user", "content": content}
        self.provider_history.append(provider_entry)
        
        self.logger.debug(f"MessageHistory - Added user message: {content}")

    def get_canonical_history(self) -> List[Dict[str, Any]]:
        """Get the canonical (provider-agnostic) history.
        
        Returns:
            List[Dict[str, Any]]: List of canonical history entries.
        """
        return self.canonical_history
    
    def get_provider_history(self, provider: Optional[ELLMProvider] = None) -> List[Dict[str, Any]]:
        """Get provider-specific history, optionally for a different provider.
        
        Args:
            provider (Optional[ELLMProvider]): Provider to format for. If None, uses current provider.
            
        Returns:
            List[Dict[str, Any]]: List of messages formatted for the specified provider.
        """
        if provider is None or provider == self._current_provider:
            self.logger.debug(f"Returning current provider ({self._current_provider.value}) history")
            return self.provider_history
        
        # Generate history for different provider without changing current state
        self.logger.debug(f"Generating history for provider: {provider.value}")
        temp_history = []
        
        for entry in self.canonical_history:
            entry_type = entry["type"]
            
            if entry_type == "user":
                temp_history.append(entry["data"])
            elif entry_type == "assistant":
                temp_history.append(entry["data"])
            elif entry_type == "tool_call":
                tool_call = entry["data"]
                provider_entry = {
                    "role": "assistant",
                    "tool_calls": [ProviderRegistry.get_provider(self._current_provider).hatchling_to_llm_tool_call(tool_call)]
                }
                temp_history.append(provider_entry)
            elif entry_type == "tool_result":
                tool_result = entry["data"]
                provider_entry = {
                    "role": "tool",
                    **ProviderRegistry.get_provider(self._current_provider).hatchling_to_provider_tool_result(tool_result)
                }
                temp_history.append(provider_entry)

        return temp_history
    
    def copy(self) -> 'MessageHistory':
        """Create a copy of this message history.
        
        Returns:
            MessageHistory: A new MessageHistory with the same canonical and provider histories.
        """
        new_history = MessageHistory()
        new_history.canonical_history = self.canonical_history.copy()
        new_history.provider_history = self.provider_history.copy()
        new_history._current_provider = self._current_provider
        new_history._content_buffer = self._content_buffer
        return new_history
    
    def clear(self) -> None:
        """Clear all histories."""
        self.canonical_history = []
        self.provider_history = []
        self._content_buffer = ""
        self._current_provider = None
        
        self.logger.info("MessageHistory - Cleared!")
    
    def delete_last_n_messages(self, n: int) -> None:
        """Delete the last 'n' messages from the history.
        
        Args:
            n (int): The number of messages to delete from the end of the history.
        """
        if n <= 0:
            self.logger.warning(f"Attempted to delete {n} messages. 'n' must be a positive integer.")
            return

        if len(self.canonical_history) < n:
            self.logger.warning(f"Attempted to delete {n} messages, but only {len(self.canonical_history)} exist. Clearing history.")
            self.canonical_history = []
        else:
            self.canonical_history = self.canonical_history[:-n]
        
        self._regenerate_provider_history()
        self.logger.info(f"Deleted last {n} messages. Current history length: {len(self.canonical_history)}")

    def delete_last_message(self) -> None:
        """Delete the last message from the history."""
        self.delete_last_n_messages(1)
        self.logger.info("Deleted last message.")

    def keep_last_n_messages(self, n: int) -> None:
        """Keep only the last 'n' messages in the history, deleting older ones.
        
        Args:
            n (int): The number of most recent messages to keep.
        """
        if n <= 0:
            self.logger.warning(f"Attempted to keep {n} messages. 'n' must be a positive integer. Clearing history.")
            self.canonical_history = []
        elif len(self.canonical_history) > n:
            self.canonical_history = self.canonical_history[-n:]
        
        self._regenerate_provider_history()
        self.logger.info(f"Kept last {n} messages. Current history length: {len(self.canonical_history)}")

    def __len__(self) -> int:
        """Get the number of entries in canonical history.
        
        Returns:
            int: The number of entries in the canonical history.
        """
        return len(self.canonical_history)

    def get_formatted_history(self, n: Optional[int] = None) -> str:
        """Get a formatted string representation of the canonical history.
        
        Args:
            n (Optional[int]): If provided, return only the last 'n' messages.
            
        Returns:
            str: A multi-line string with formatted history entries.
        """
        history_to_format = self.canonical_history
        if n is not None and n > 0:
            history_to_format = self.canonical_history[-n:]

        formatted_output = []
        for i, entry in enumerate(history_to_format):
            entry_type = entry["type"]
            data = entry["data"]

            # Use the enumerate index for display
            display_index = i + 1

            if entry_type == "user":
                formatted_output.append(f"[{display_index}] User: {data.get("content", "")}")
            elif entry_type == "assistant":
                formatted_output.append(f"[{display_index}] Assistant: {data.get("content", "")}")
            elif entry_type == "tool_call":
                tool_call = data
                formatted_output.append(f"[{display_index}] Tool Call: {tool_call.function_name}({tool_call.arguments})")
            elif entry_type == "tool_result":
                tool_result = data
                formatted_output.append(f"[{display_index}] Tool Result ({tool_result.function_name}): {tool_result.result or tool_result.error}")
            else:
                formatted_output.append(f"[{display_index}] Unknown Entry Type: {entry_type} - {data}")
        
        if not formatted_output:
            return "(History is empty)"
        
        return "\n".join(formatted_output)

    def save_history_to_file(self, file_path: str) -> None:
        """Save the canonical history to a specified file in JSON format.
        
        Args:
            file_path (str): The absolute path to the file where the history will be saved.
        """
        try:
            serializable_history = []
            for entry in self.canonical_history:
                serializable_entry = entry.copy()
                if "data" in serializable_entry and hasattr(serializable_entry["data"], "to_dict"):
                    serializable_entry["data"] = serializable_entry["data"].to_dict()
                serializable_history.append(serializable_entry)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_history, f, ensure_ascii=False, indent=4)
            self.logger.info(f"History saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save history to {file_path}: {e}")

    def load_history_from_file(self, file_path: str) -> None:
        """Load canonical history from a specified JSON file.
        
        Args:
            file_path (str): The absolute path to the file from which the history will be loaded.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_history = json.load(f)
            
            deserialized_history = []
            for entry in loaded_history:
                if entry["type"] == "tool_call":
                    entry["data"] = ToolCallParsedResult(**entry["data"])
                elif entry["type"] == "tool_result":
                    entry["data"] = ToolCallExecutionResult(**entry["data"])
                deserialized_history.append(entry)

            self.canonical_history = deserialized_history
            # After loading, ensure the current provider is set for history regeneration
            # This prevents issues where _current_provider might be None after loading
            # and _regenerate_provider_history tries to use it.
            self._current_provider = self.settings.llm.provider_enum
            self._regenerate_provider_history()
            self.logger.info(f"History loaded from {file_path}")
        except FileNotFoundError:
            self.logger.error(f"History file not found: {file_path}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON from {file_path}: {e}")
        except Exception as e:
            self.logger.error(f"Failed to load history from {file_path}: {e}")