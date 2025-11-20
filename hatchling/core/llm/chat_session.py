import logging
import uuid
from typing import List, Dict, Any

from hatchling.core.logging.logging_manager import logging_manager
from hatchling.core.chat.message_history_registry import MessageHistoryRegistry
from hatchling.core.llm.tool_management.tool_chaining_subscriber import ToolChainingSubscriber
from hatchling.core.llm.providers import ProviderRegistry

from hatchling.config.settings import AppSettings
from hatchling.mcp_utils.mcp_tool_execution import MCPToolExecution
from hatchling.mcp_utils.mcp_tool_call_subscriber import MCPToolCallSubscriber

class ChatSession:
    def __init__(self, settings: AppSettings = None):
        """Initialize a chat session with the specified settings.
        
        Args:
            settings (AppSettings, optional): Configuration settings for the chat session. 
                                            If None, uses the singleton instance.
        """
        self.settings = settings or AppSettings.get_instance()
        # Unified logger naming: ChatSession-provider-model
        self.logger = logging_manager.get_session(
            f"ChatSession",
            formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        # Initialize MCPToolExecution for execution of tool calls from LLM providers
        self.tool_execution = MCPToolExecution()
        # Initialize message components
        self.session_id = str(uuid.uuid4())
        self.history = MessageHistoryRegistry.get_or_create_history(self.session_id, settings=self.settings)

        # Create tool chaining subscriber for automatic tool calling chains
        self._tool_chaining_subscriber = ToolChainingSubscriber(self.settings, self.tool_execution, self.session_id)
        # Create and subscribe the MCP tool call subscriber for LLM tool calls
        self._tool_call_subscriber = MCPToolCallSubscriber(self.tool_execution)
        
        
        # Initializing all connections of subscribers to instances of LLM providers
        for _provider_enum in ProviderRegistry.list_providers():
            self.logger.debug(f"Initializing streaming subscriptions for : {_provider_enum.value}")
            _provider = ProviderRegistry.get_provider(_provider_enum, self.settings)

            # Subscribe core subscribers to provider's publisher
            self.logger.debug("Subscribed core subscribers to LLM provider's stream publisher")
            
            # Subscribe tool handling subscribers
            self.logger.debug("Subscribed tool handling subscribers")
            _provider.publisher.subscribe(self._tool_call_subscriber)
            _provider.publisher.subscribe(self._tool_chaining_subscriber)
            
            # Subscribe message history to provider's publisher for event-driven updates
            _provider.publisher.subscribe(self.history)

        # Subscribe the tool chaining to the tool executer    
        self.tool_execution.event_publisher.subscribe(self._tool_chaining_subscriber)
        
        # Subscribe message history to tool execution events
        self.tool_execution.event_publisher.subscribe(self.history)
    
    def register_subscriber(self, subscriber) -> None:
        """Register a subscriber to all relevant publishers.
        
        This method provides decoupled registration for UI and other subscribers
        without tight coupling to backend logic.
        
        Args:
            subscriber: The subscriber to register (must implement EventSubscriber interface).
        """
        # Subscribe to all LLM provider publishers
        for _provider_enum in ProviderRegistry.list_providers():
            _provider = ProviderRegistry.get_provider(_provider_enum, self.settings)
            _provider.publisher.subscribe(subscriber)
        
        # Subscribe to tool execution events
        self.tool_execution.event_publisher.subscribe(subscriber)
        self._tool_chaining_subscriber.publisher.subscribe(subscriber)
        
        self.logger.debug(f"Registered subscriber {type(subscriber).__name__} to all publishers")
    
    async def send_message(self, user_message: str) -> None:
        """Send the current message history to the LLM provider and stream the response.
        
        Args:
            user_message (str): The user's message to process.
        """
        # Add user message to history
        self.history.add_user_message(user_message)

        # Get current provider based on settings
        provider = ProviderRegistry.get_current_provider(self.settings)
        
        # Reset tool calling counters and collectors for a new user message
        self.tool_execution.reset_for_new_query(user_message)

        # Prepare payload using provider abstraction
        payload = provider.prepare_chat_payload(
            self.history.get_provider_history(provider.provider_enum), 
            self.settings.llm.model
        )
        
        # Add tools to payload
        # TODO: Currently, we are adding everything to the payload as we are not specifying
        # tools in add_tools_to_payload.
        # In the future, we must allow users to specify tools directly in the query.
        payload = provider.add_tools_to_payload(payload)
        
        self.logger.debug(f"Sending payload to LLM: {payload}")
        # Stream the response using provider abstraction
        await provider.stream_chat_response(payload)