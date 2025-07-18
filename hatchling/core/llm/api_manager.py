import json
from typing import List, Dict, Tuple, Any
import logging
import aiohttp
from hatchling.core.logging.logging_manager import logging_manager
from hatchling.config.settings import AppSettings
from hatchling.core.chat.message_history import MessageHistory

class APIManager:
    """Manages API communication with the LLM."""
    
    def __init__(self, settings: AppSettings):
        """Initialize the API manager.
        
        Args:
            settings: The application settings
        """
        self.settings = settings
        provider = settings.llm.get_active_provider()
        model = settings.llm.get_active_model()
        self.logger = logging_manager.get_session(
            f"APIManager-{provider}-{model}",
            formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.model_name = model
    
    def prepare_request_payload(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare the request payload for the LLM API.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            The prepared payload.
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True  # Always stream
        }
        self.logger.debug(f"Prepared payload: {json.dumps(payload)}")
        return payload
    
    def add_tools_to_payload(self, payload: Dict[str, Any], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add tools/functions to the payload depending on provider."""
        if not tools:
            return payload
        if self.settings.llm.get_active_provider() == "openai":
            # Use the new OpenAI tools format (not the deprecated functions format)
            openai_tools = []
            for tool in tools:
                if tool.get("type") == "function":
                    # Tool is already in the correct format
                    openai_tools.append(tool)
                else:
                    # Convert to new tools format
                    openai_tools.append({
                        "type": "function",
                        "function": tool
                    })
            payload["tools"] = openai_tools
            payload["tool_choice"] = "auto"
            self.logger.debug(f"Added {len(openai_tools)} tools to OpenAI payload: {openai_tools}")
        else:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
            self.logger.debug(f"Added {len(tools)} tools to payload: {tools}")
        return payload
    
    def has_tool_calls(self, data: Dict[str, Any]) -> bool:
        """Check if the data contains tool calls.
        
        Args:
            data: The data to check from the LLM.
            
        Returns:
            True if the data contains tool calls, False otherwise.
        """
        return ("message" in data and 
                "tool_calls" in data["message"] and 
                data["message"]["tool_calls"])
    
    def has_message_content(self, data: Dict[str, Any]) -> bool:
        """Check if the data contains message content.
        
        Args:
            data: The data to check.
            
        Returns:
            True if the data contains message content, False otherwise.
        """
        return "message" in data and "content" in data["message"]
    
    async def process_response_data(self,
                                    data: Dict[str, Any],
                                    message_tool_calls: List,
                                    tool_executor
                                    ) -> Tuple[str, List]:
        """Process response data and extract content and tool calls.
        
        Args:
            data (Dict[str, Any]): The response data to process from the LLM.
            message_tool_calls (List): List of processed tool calls.
            tool_executor: The tool execution manager to handle tool calls.
            
        Returns:
            Tuple[str, List]: A tuple containing:
                - str: The extracted content from the response
                - List: The collected tool results
        """
        content = ""
        tool_results = []
        
        # Process tool calls if present
        if self.has_tool_calls(data):
            tool_calls_result = await tool_executor.handle_streaming_tool_calls(data, message_tool_calls)
            if tool_calls_result:
                tool_results.extend(tool_calls_result)
        
        # Extract message content
        if self.has_message_content(data):
            content = data["message"]["content"]
        
        return content, tool_results
    
    async def stream_response(self, 
                              session: aiohttp.ClientSession, 
                              payload: Dict[str, Any], 
                              history: MessageHistory,
                              tool_executor,
                              print_output: bool = True, 
                              prefix: str = None, 
                              update_history: bool = True) -> Tuple[str, List, List]:
        """Stream a response from the API and handle common processing.
        
        Args:
            session: The aiohttp client session to use.
            payload: The request payload to send to the API.
            history: The message history to update
            tool_executor (ToolExecutionManager): Tool execution manager
            print_output: Whether to print the output to the console
            prefix: Optional prefix to print before the response.
            update_history: Whether to update message history with the response.
            
        Returns:
            Tuple containing (full_response, message_tool_calls, tool_results).
        """
        full_response = ""
        message_tool_calls = []
        tool_results = []
        
        if prefix and print_output:
            print(prefix)
            
        async with session.post(f"{self.settings.llm.api_url}/chat", json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                self.logger.error(f"Error: {response.status}, {error_text}")
                raise Exception(f"Error: {response.status}, {error_text}")
            
            async for line in response.content.iter_any():
                if not line:
                    continue
                
                try:
                    line_text = line.decode('utf-8').strip()
                    if not line_text:
                        continue
                    
                    # Debug log the raw response
                    self.logger.debug(f"Raw response: {line_text}")
                    
                    # Parse the JSON response
                    data = json.loads(line_text)
                    
                    # Process the response data
                    content, current_tool_results = await self.process_response_data(data, message_tool_calls, tool_executor)

                    if current_tool_results:
                        tool_results.extend(current_tool_results)
                    elif content:
                        if print_output:
                            print(content, end="", flush=True)
                        full_response += content
                    
                    # Check if this is the last message
                    if data.get("done", False):
                        if print_output:
                            print()  # Add a newline after completion
                        break
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing response: {e}")
        
        # Update message history if requested
        if update_history and history:
            history.update_message_history(full_response, message_tool_calls, tool_results)
        
        return full_response, message_tool_calls, tool_results

    async def _stream_openai_response(self,
                                      session: aiohttp.ClientSession,
                                      payload: Dict[str, Any],
                                      history: MessageHistory,
                                      tool_executor,
                                      print_output: bool = True,
                                      prefix: str = None,
                                      update_history: bool = True) -> Tuple[str, List, List]:
        """Stream a response from the OpenAI API, supporting function calling."""

        full_response = ""
        message_tool_calls = []
        tool_results = []
        function_call_accumulator = None
        function_call_name = None
        function_call_args = ""
        function_call_id = None

        headers = {"Authorization": f"Bearer {self.settings.llm.openai_api_key}"}

        if prefix and print_output:
            print(prefix)

        async with session.post(f"{self.settings.llm.openai_api_url}/chat/completions",
                                json=payload,
                                headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                self.logger.error(f"Error: {response.status}, {error_text}")
                raise Exception(f"Error: {response.status}, {error_text}")

            async for line in response.content:
                if not line:
                    continue

                line_text = line.decode("utf-8").strip()
                if not line_text:
                    continue

                for chunk in line_text.split("\n\n"):
                    if not chunk:
                        continue
                    if chunk.startswith("data:"):
                        chunk = chunk[len("data:"):].strip()
                    if chunk == "[DONE]":
                        if print_output:
                            print()
                        break

                    try:
                        data = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue

                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})

                    # Handle tool calls (new format)
                    if "tool_calls" in delta:
                        tool_calls = delta["tool_calls"]
                        if tool_calls and len(tool_calls) > 0:
                            tool_call = tool_calls[0]  # Take the first tool call
                            if "function" in tool_call:
                                fc = tool_call["function"]
                                if function_call_accumulator is None:
                                    function_call_accumulator = ""
                                    function_call_name = fc.get("name")
                                    function_call_id = tool_call.get("id") or "tool_call"
                                if "arguments" in fc:
                                    function_call_accumulator += fc["arguments"]
                                continue

                    # Handle normal content
                    content_piece = delta.get("content")
                    if content_piece:
                        if print_output:
                            print(content_piece, end="", flush=True)
                        full_response += content_piece

            # If a function call was accumulated, execute it
            if function_call_accumulator and function_call_name:
                try:
                    args = json.loads(function_call_accumulator)
                except Exception:
                    args = {}
                # Execute the tool
                tool_result = await tool_executor.execute_tool(function_call_id, function_call_name, args)
                if tool_result:
                    # Add tool result in the format expected by update_message_history
                    tool_results.append({
                        "tool_call_id": function_call_id,
                        "name": function_call_name,
                        "content": tool_result["content"]
                    })
                    message_tool_calls.append({
                        "id": function_call_id,
                        "type": "function",  # Required by OpenAI
                        "function": {"name": function_call_name, "arguments": function_call_accumulator}
                    })
                    # Note: Tool result will be added to history by update_message_history

        if update_history and history:
            history.update_message_history(full_response, message_tool_calls, tool_results)

        return full_response, message_tool_calls, tool_results

    async def stream_response(self,
                              session: aiohttp.ClientSession,
                              payload: Dict[str, Any],
                              history: MessageHistory,
                              tool_executor,
                              print_output: bool = True,
                              prefix: str = None,
                              update_history: bool = True) -> Tuple[str, List, List]:
        """Stream a response using the configured provider."""

        if self.settings.llm.get_active_provider() == "openai":
            return await self._stream_openai_response(
                session, payload, history, tool_executor,
                print_output=print_output,
                prefix=prefix,
                update_history=update_history,
            )

        return await self._stream_ollama_response(
            session, payload, history, tool_executor,
            print_output=print_output,
            prefix=prefix,
            update_history=update_history,
        )
