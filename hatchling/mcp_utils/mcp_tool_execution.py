"""MCP tool execution management with event publishing.

This module provides enhanced functionality for handling tool execution requests from LLMs,
managing tool calling chains, and processing tool results with event-driven architecture.
"""

import logging
import time
import asyncio
import json
from mcp.types import CallToolResult

from hatchling.mcp_utils.manager import mcp_manager
from hatchling.core.logging.logging_manager import logging_manager
from hatchling.config.settings import AppSettings
from hatchling.core.llm.event_system import EventPublisher, EventType
from hatchling.core.llm.data_structures import ToolCallParsedResult, ToolCallExecutionResult

class MCPToolExecution:
    """Manages tool execution and tool calling chains with event publishing."""
    
    def __init__(self, settings: AppSettings = None):
        """Initialize the MCP tool execution manager.
        
        Args:
            settings (AppSettings, optional): The application settings.
                                            If None, uses the singleton instance.
        """
        self.settings = settings or AppSettings.get_instance()
        self.logger = logging_manager.get_session("MCPToolExecution")
        logging_manager.set_log_level(logging.INFO)
        
        # Initialize event publisher
        self._event_publisher = EventPublisher()
        
        # Tool calling control properties
        self.current_tool_call_iteration = 0
        self.tool_call_start_time = None
        self.root_tool_query = None  # Track the original user query that started the tool sequence
    
    @property
    def event_publisher(self) -> EventPublisher:
        """Get the stream publisher for this tool execution manager.
        
        Returns:
            EventPublisher: The stream publisher instance.
        """
        return self._event_publisher

    def reset_for_new_query(self, query: str) -> None:
        """Reset tool execution state for a new user query.
        
        Args:
            query (str): The user's query that's starting a new conversation.
        """
        self.current_tool_call_iteration = 0
        self.tool_call_start_time = time.time()
        self.root_tool_query = query
    
    async def execute_tool(self, parsed_tool_call: ToolCallParsedResult) -> None:
        """Execute a tool and return its result.

        Sends the tool call to the MCPManager for execution and publishes events
        for tool call dispatched, progress, result, and error handling.
        You can subscribe to `event_publisher` of this class to receive
        MCP_TOOL_CALL_DISPATCHED, MCP_TOOL_CALL_PROGRESS, MCP_TOOL_CALL_RESULT, and MCP_TOOL_CALL_ERROR events.
        That will allow you to react to tool calls in real-time and handle them accordingly.

        Args:
            parsed_tool_call (ToolCallParsedResult): The parsed tool call containing
                tool_id, function_name, and arguments.
        """
        self.logger.debug(
            f"Redirecting to tool execution for (tool_call_id: {parsed_tool_call.tool_call_id}; "
            f"function: {parsed_tool_call.function_name}; arguments: {parsed_tool_call.arguments})"
        )

        self.current_tool_call_iteration += 1

        # Publish tool call dispatched event
        self._event_publisher.publish(EventType.MCP_TOOL_CALL_DISPATCHED, parsed_tool_call.to_dict())

        try:
            # Process the tool call using MCPManager
            tool_response = await mcp_manager.execute_tool(
                tool_name=parsed_tool_call.function_name,
                arguments=parsed_tool_call.arguments
            )
            self.logger.debug(f"Tool {parsed_tool_call.function_name} executed with responses: {tool_response}")

            if tool_response and not tool_response.isError:
                # Convert CallToolResult to a serializable dictionary
                serializable_tool_response = tool_response.__dict__.copy()
                if "content" in serializable_tool_response and isinstance(serializable_tool_response["content"], list):
                    serializable_tool_response["content"] = [
                        item.text if hasattr(item, "text") else str(item)
                        for item in serializable_tool_response["content"]
                    ]

                result_obj = ToolCallExecutionResult(
                    **parsed_tool_call.to_dict(),
                    result=serializable_tool_response,
                    error=None
                )
                self._event_publisher.publish(EventType.MCP_TOOL_CALL_RESULT, result_obj.to_dict())
            else:
                # Convert CallToolResult to a serializable dictionary for error case as well
                serializable_tool_response = tool_response.__dict__.copy()
                if "content" in serializable_tool_response and isinstance(serializable_tool_response["content"], list):
                    serializable_tool_response["content"] = [
                        item.text if hasattr(item, "text") else str(item)
                        for item in serializable_tool_response["content"]
                    ]

                result_obj = ToolCallExecutionResult(
                    **parsed_tool_call.to_dict(),
                    result=serializable_tool_response,
                    error="Tool execution failed or returned no valid response"
                )
                self._event_publisher.publish(EventType.MCP_TOOL_CALL_ERROR, result_obj.to_dict())

        except Exception as e:
            self.logger.error(f"Error executing tool: {e}")
            # For error case, create a serializable representation of the error result
            error_content = [{"type": "text", "text": f"{e}"}]
            serializable_error_response = {
                "meta": None,
                "content": [item["text"] if isinstance(item, dict) and "text" in item else str(item) for item in error_content],
                "structuredContent": None,
                "isError": True,
            }
            result_obj = ToolCallExecutionResult(
                **parsed_tool_call.to_dict(),
                result=serializable_error_response,
                error=str(e)
            )
            self._event_publisher.publish(EventType.MCP_TOOL_CALL_ERROR, result_obj.to_dict())

    def execute_tool_sync(self, parsed_tool_call: ToolCallParsedResult) -> None:
        """Synchronous wrapper for execute_tool that handles async execution internally.
        
        This method creates a task to execute the tool asynchronously without blocking
        the caller. It's designed for use in synchronous contexts where you want to
        dispatch tool execution but don't need to wait for the result.
        
        Args:
            parsed_tool_call (ToolCallParsedResult): The parsed tool call containing
                tool_id, function_name, and arguments.
        """
        try:
            # Try to create a task in the current event loop
            asyncio.create_task(self.execute_tool(parsed_tool_call))
        except RuntimeError:
            # No event loop running, create one for this execution
            try:
                asyncio.run(self.execute_tool(parsed_tool_call))
            except Exception as e:
                self.logger.warning(f"Failed to execute tool synchronously: {e}")

