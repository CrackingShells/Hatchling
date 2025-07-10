"""Tool execution management for LLM interactions.

This module provides functionality for handling tool execution requests from LLMs,
managing tool calling chains, and processing tool results.
"""

import json
import logging
import time
from typing import List, Dict, Tuple, Any, Optional

from hatchling.mcp_utils.manager import mcp_manager
from hatchling.core.logging.logging_manager import logging_manager
from hatchling.config.settings import ChatSettings

class ToolExecutionManager:
    """Manages tool execution and tool calling chains."""
    
    def __init__(self, settings: ChatSettings):
        """Initialize the tool execution manager.
        
        Args:
            settings (ChatSettings): The application settings.
        """
        self.settings = settings
        provider = settings.get_active_provider()
        model = settings.get_active_model()
        self.logger = logging_manager.get_session(
            f"ToolExecutionManager-{provider}-{model}",
            formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.tools_enabled = False
        
        # Tool calling control properties
        self.current_tool_call_iteration = 0
        self.tool_call_start_time = None
        self.root_tool_query = None  # Track the original user query that started the tool sequence
    
    async def initialize_mcp(self, server_paths: List[str]) -> bool:
        """Initialize connection to MCP servers.
        
        Args:
            server_paths (List[str]): List of paths to MCP server scripts.
            
        Returns:
            bool: True if connection was successful, False otherwise.
        """
        # Use MCPManager to initialize everything
        self.tools_enabled = await mcp_manager.initialize(server_paths)
        
        if self.tools_enabled:
            self.logger.info(f"Connected to MCP servers")
        else:
            self.logger.warning("Failed to connect to any MCP server")
            
        return self.tools_enabled
    
    def get_tools_for_payload(self) -> List[Dict[str, Any]]:
        """Get the list of tools/functions for the LLM API payload, in the correct format for the provider."""
        if self.settings.llm_provider == "openai":
            # Return OpenAI function format (list of function schemas)
            tools = mcp_manager.get_ollama_tools()
            openai_functions = []
            for tool in tools:
                if tool.get("type") == "function" and "function" in tool:
                    openai_functions.append(tool["function"])
                else:
                    openai_functions.append(tool)
            return openai_functions
        else:
            return mcp_manager.get_ollama_tools()

    def reset_for_new_query(self, query: str) -> None:
        """Reset tool execution state for a new user query.
        
        Args:
            query (str): The user's query that's starting a new conversation.
        """
        self.current_tool_call_iteration = 0
        self.tool_call_start_time = time.time()
        self.root_tool_query = query
    
    async def execute_tool(self, tool_id: str, function_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a tool and return its result.
        
        Args:
            tool_id (str): The ID of the tool.
            function_name (str): The name of the function to execute.
            arguments (Dict[str, Any]): The arguments to pass to the function.
            
        Returns:
            Optional[Dict[str, Any]]: The tool result dictionary, or None if execution failed.
        """
        try:
            # Increment the tool call iteration counter each time a tool is executed
            self.current_tool_call_iteration += 1
            
            # Format the tool call for the MCPManager
            formatted_tool_call = {
                "id": tool_id,
                "function": {
                    "name": function_name,
                    "arguments": json.dumps(arguments) if isinstance(arguments, dict) else arguments
                }
            }
            
            # Process the tool call using MCPManager
            tool_responses = await mcp_manager.process_tool_calls([formatted_tool_call])
            
            if tool_responses and len(tool_responses) > 0:
                tool_response = tool_responses[0]
                result_content = tool_response.get("content", "No result")
                
                # Try to parse the content to a more user-friendly format
                try:
                    if isinstance(result_content, str):
                        parsed_result = json.loads(result_content)
                        result_content = parsed_result
                except json.JSONDecodeError:
                    # Keep as string if it's not valid JSON
                    pass
                
                # Show the tool result to the user
                self.logger.info(f"[Tool result: {result_content}]")
                
                # Return the tool result
                return {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": function_name,
                    "content": str(result_content)
                }
        except Exception as e:
            self.logger.error(f"Error executing tool: {e}")
        
        return None
    
    async def process_tool_call(self, tool_call: Dict[str, Any], tool_id: str) -> Optional[Dict[str, Any]]:
        """Process a single tool call and return the result.
        
        Args:
            tool_call (Dict[str, Any]): The tool call to process.
            tool_id (str): The ID of the tool call.
            
        Returns:
            Optional[Dict[str, Any]]: The tool result dictionary, or None if processing failed.
        """
        function_name = tool_call["function"]["name"]
        arguments = tool_call["function"]["arguments"]
        
        # Parse arguments if needed
        try:
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
        
        # Show tool usage to the user
        self.logger.info(f"[Using tool: {function_name} with arguments: {json.dumps(arguments)}]")
        
        # Execute tool and get result
        if self.tools_enabled:
            return await self.execute_tool(tool_id, function_name, arguments)
        
        return None

    async def handle_streaming_tool_calls(self, data: Dict[str, Any], message_tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process tool calls from streaming response data.
        
        Args:
            data (Dict[str, Any]): The response data from the LLM.
            message_tool_calls (List[Dict[str, Any]]): List of processed tool calls.
            
        Returns:
            List[Dict[str, Any]]: List of tool results.
        """
        tool_results = []
        current_tool_calls = data["message"]["tool_calls"]
        self.logger.info(f"Found tool calls: {json.dumps(current_tool_calls)}")
        
        # Process each tool call
        for tool_call in current_tool_calls:
            tool_id = tool_call.get("id", "unknown")
            # Skip if we've already processed this tool call
            if any(tc.get("id") == tool_id for tc in message_tool_calls):
                continue
                
            # Add to our tracking list
            message_tool_calls.append(tool_call)
            
            # Process the tool call
            tool_result = await self.process_tool_call(tool_call, tool_id)
            if tool_result:
                tool_results.append(tool_result)
        
        return tool_results

    async def handle_tool_calling_chain(self,
                    session,
                    api_manager,
                    history,
                    full_response: str,
                    message_tool_calls: List[Dict[str, Any]],
                    tool_results: List[Dict[str, Any]]
                    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Handle the response from the LLM API and manage tool calling chains.
        
        Args:
            session: The http session to use for API requests.
            api_manager: The API manager for making API calls.
            history: The message history.
            full_response (str): The latest response from the LLM.
            message_tool_calls (List[Dict[str, Any]]): List of tool calls from the response.
            tool_results (List[Dict[str, Any]]): List of tool execution results.
                        
        Returns:
            Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]: A tuple containing:
                - str: The full response text
                - List[Dict[str, Any]]: The message tool calls
                - List[Dict[str, Any]]: The tool execution results
        """
        _full_response, _message_tool_calls, _tool_results = full_response, message_tool_calls, tool_results
        
        elapsed_time = time.time() - self.tool_call_start_time
        
        # Check if we've hit limits
        reached_max_iterations = self.current_tool_call_iteration >= self.settings.max_tool_call_iteration
        reached_time_limit = elapsed_time >= self.settings.max_working_time
        
        if reached_max_iterations or reached_time_limit:
            # We've reached a limit, return what we have
            limit_reason = "maximum iterations" if reached_max_iterations else "time limit"
            self.logger.warning(f"Reached {limit_reason} for tool calling ({self.current_tool_call_iteration} iterations, {elapsed_time:.1f}s)")
            return _full_response, _message_tool_calls, _tool_results
        
        # Continue with sequential tool calling - prepare new payload with updated messages
        self.logger.debug("Preparing next payload for sequential tool calling")
        history.add_user_message(f"Given the tool results: {tool_results}, do you have enough information to answer the original query: `{self.root_tool_query}`? If not, please ask for more information or continue using tools.")
        
        # Prepare the next payload
        payload = api_manager.prepare_request_payload(history.get_messages())
        
        payload = api_manager.add_tools_to_payload(payload, self.get_tools_for_payload())
    
        __full_response, __message_tool_calls, __tool_results = await api_manager.stream_response(
                session, payload, history, self, print_output=False, update_history=True
            )

        if __tool_results:
            # Process the next step (recursive call)
            self.logger.info(f"Continuing with tool calling iteration {self.current_tool_call_iteration}/{self.settings.max_tool_call_iteration} ({elapsed_time:.1f}s elapsed)")
            _full_response, _message_tool_calls, _tool_results = await self.handle_tool_calling_chain(
                                                            session,
                                                            api_manager, 
                                                            history,
                                                            _full_response+"\n\n"+__full_response,
                                                            _message_tool_calls+__message_tool_calls,
                                                            _tool_results+__tool_results)

        # Return the original response if it was a direct LLM response with no tool usage
        return _full_response, _message_tool_calls, _tool_results