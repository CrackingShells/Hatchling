"""OpenAI provider implementation for LLM abstraction layer.

This module provides the OpenAIProvider class that implements the LLMProvider interface
for OpenAI's API using the official openai Python client.
"""

import logging
import uuid
import json
from typing import Dict, Any, List, Optional, AsyncIterator, Union
from httpx import AsyncClient

from openai import AsyncOpenAI

from hatchling.core.llm.providers.base import LLMProvider
from hatchling.core.llm.providers.registry import ProviderRegistry
from hatchling.config.settings import AppSettings
from hatchling.config.llm_settings import ELLMProvider
from hatchling.mcp_utils import mcp_manager
from hatchling.mcp_utils.mcp_tool_data import MCPToolInfo
from hatchling.core.llm.event_system import EventPublisher, EventType
from hatchling.mcp_utils.mcp_tool_lifecycle_subscriber import ToolLifecycleSubscriber
from hatchling.core.llm.event_system.event_subscribers_examples import Event
from hatchling.core.llm.data_structures import ToolCallParsedResult, ToolCallExecutionResult

logger = logging.getLogger(__name__)


@ProviderRegistry.register(ELLMProvider.OPENAI)
class OpenAIProvider(LLMProvider):
    """OpenAI provider for ChatGPT and GPT models.
    
    This provider uses the OpenAI AsyncClient to communicate with OpenAI's API.
    It supports streaming responses, tool calling, and all OpenAI chat models.
    """
    
    def __init__(self, settings: AppSettings = None):
        """Initialize the OpenAI provider.
        
        Args:
            settings (AppSettings, optional): Application settings containing OpenAI configuration.
                                            If None, uses the singleton instance.
        
        Raises:
            ValueError: If OpenAI API key is not provided in settings.
        """
        super().__init__(settings)
        self._http_client: Optional[AsyncClient] = None  # for AsyncOpenAI compatibility
        self._client: Optional[AsyncOpenAI] = None

        # Tool call streaming state
        self._tool_call_accumulator = {}
        self._tool_call_streaming = False

        self.initialize()  # Initialize the client immediately
        
        if not self._settings.openai.api_key:
            raise ValueError("OpenAI API key is required")
    
    @property
    def provider_name(self) -> str:
        """Return the provider name.
        
        Returns:
            str: The provider name "openai".
        """
        return ELLMProvider.OPENAI.value
    
    @property
    def provider_enum(self) -> ELLMProvider:
        """Return the provider enum.
        
        Returns:
            ELLMProvider: The enum value for OpenAI provider.
        """
        return ELLMProvider.OPENAI

    def initialize(self) -> None:
        """Initialize the OpenAI async client and verify connection.
        
        Raises:
            ConnectionError: If unable to connect to OpenAI API.
            ValueError: If API key is invalid.
        """
        try:
            # Initialize OpenAI async client using settings
            client_kwargs = {
                "api_key": self._settings.openai.api_key,
                "timeout": self._settings.openai.timeout,
            }

            # Add optional parameters if provided
            if self._settings.openai.api_base and self._settings.openai.api_base != "https://api.openai.com/v1":
                client_kwargs["base_url"] = self._settings.openai.api_base
            
            self._http_client = AsyncClient(timeout=self._settings.openai.timeout)
            self._client = AsyncOpenAI(**client_kwargs, http_client=self._http_client)

            self._event_publisher = EventPublisher()
            self._toolLifecycle_subscriber = ToolLifecycleSubscriber(self._settings.llm.provider_name, self.mcp_to_provider_tool)
            mcp_manager.publisher.subscribe(self._toolLifecycle_subscriber)

            logger.info("Successfully connected to OpenAI API")
            
        except Exception as e:
            error_msg = f"Failed to initialize OpenAI client: {str(e)}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
    
    async def close(self) -> None:
        """Close the OpenAI client connection.
        
        This method should be called to clean up resources when done.
        """
        mcp_manager.publisher.unsubscribe(self._toolLifecycle_subscriber)
        self._event_publisher.clear_subscribers()
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def prepare_chat_payload(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare chat payload for OpenAI API format.
        
        Args:
            messages (List[Dict[str, Any]]): List of message dictionaries.
            model (str): Model name to use.
            **kwargs: Additional chat parameters.
        
        Returns:
            Dict[str, Any]: Formatted payload for OpenAI chat API.
        """
        # Base payload structure
        payload = {
            "model": model,
            "messages": messages,
            "stream": kwargs.get("stream", True)
        }
        
        # Add settings-based parameters with kwargs override
        payload["temperature"] = kwargs.get("temperature", self._settings.openai.temperature)
        payload["top_p"] = kwargs.get("top_p", self._settings.openai.top_p)
        payload["max_completion_tokens"] = kwargs.get("max_completion_tokens", self._settings.openai.max_completion_tokens)
    
        
        # Add other OpenAI-specific optional parameters
        openai_params = [
            "n", "stop", "presence_penalty",
            "frequency_penalty", "logit_bias", "user", "response_format", "seed",
            "logprobs", "top_logprobs", "parallel_tool_calls",
            "stream_options", "service_tier"
        ]
        
        for param in openai_params:
            if param in kwargs:
                payload[param] = kwargs[param]
        
        # Handle streaming options
        if payload["stream"] and "stream_options" not in payload:
            # Enable usage tracking in streaming mode
            payload["stream_options"] = {"include_usage": True}
        
        logger.debug(f"Prepared OpenAI chat payload for model: {model}")
        return payload
    
    def add_tools_to_payload(
        self,
        payload: Dict[str, Any],
        tools: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Add tool definitions to the OpenAI chat payload using the tool lifecycle subscriber.
        
        Args:
            payload (Dict[str, Any]): Base chat payload.
            tools (Optional[List[str]]): List of tool names to add to the payload. If None, all available and enabled tools are added.
                                         If provided, only the specified tools that are enabled will be added to the payload.

        Returns:
            Dict[str, Any]: Updated payload with tools.
        """

        openai_tools = []
        all_tools = self._toolLifecycle_subscriber.get_all_tools()
        enabled_tools = self._toolLifecycle_subscriber.get_enabled_tools()

        if not tools:
            openai_tools = [
                tool.provider_format for tool in enabled_tools.values()
            ]

        else:
            for tool_name in tools:
                if tool_name not in all_tools:
                    error_msg = f"Function definition for {tool_name} not found in tool cache"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                if tool_name not in enabled_tools:
                    warning_msg = f"Function {tool_name} is disabled with reason: {self._toolLifecycle_subscriber.prettied_reason(all_tools[tool_name].reason)}"
                    logger.warning(warning_msg)
                    continue  # skip disabled tools
                # OpenAI expects the provider_format to be compatible
                openai_tools.append(enabled_tools[tool_name].provider_format)

        if openai_tools:
            payload["tools"] = openai_tools
            if "tool_choice" not in payload:
                payload["tool_choice"] = "auto"
            logger.debug(f"Added {len(openai_tools)} tools to OpenAI payload")
        return payload
    
    async def stream_chat_response(
        self, 
        payload: Dict[str, Any]
    ):
        """Stream chat response from OpenAI.
        
        Args:
            payload (Dict[str, Any]): Chat request payload.
        
        Raises:
            RuntimeError: If client is not initialized.
            Exception: If streaming fails.
        """
        if not self._client:
            raise RuntimeError("OpenAI client not initialized. Call initialize() first.")
        
        try:
            # Given that OpenAI's API key can be set in the settings at any time by the user,
            # we always re-assign before making a request
            # TODO: Although less severe, this is similar to Ollama's case where we
            # constantly have to re-assign data. An necessary optimization will be
            # made once the command pattern to set the settings will allow callbacks
            # using the publish-subscribe pattern.
            self._client.api_key = self._settings.openai.api_key

            # Ensure streaming is enabled for this request
            payload["stream"] = True

            # Generate a unique request ID for this streaming session
            request_id = str(uuid.uuid4())
            self._event_publisher.set_request_id(request_id)
            
            # Stream the response
            response_stream = await self._client.chat.completions.create(**payload)
            async for chunk in response_stream:
                # Parse and publish events
                self._parse_and_publish_chunk(chunk)
                    
        except Exception as e:
            error_msg = f"Error streaming from OpenAI: {str(e)}"
            logger.error(error_msg)
            
            # Publish error event
            self._event_publisher.publish(EventType.ERROR, {
                "error": {
                    "message": error_msg,
                    "type": "openai_streaming_error"
                }
            })
            raise

    def _parse_and_publish_chunk(self, chunk: Any) -> None:
        """Parse a ChatCompletionChunk and publish appropriate events.

        This method accumulates tool call fragments and only emits the LLM_TOOL_CALL_REQUEST event
        when the tool call is fully streamed (i.e., when delta.tool_calls is None).

        Args:
            chunk (Any): Raw chunk from OpenAI API. The type is typically ChatCompletionChunk.
        """
        try:
            # Handle final usage chunk (has no choices but includes usage data)
            if not chunk.choices and chunk.usage:
                self._event_publisher.publish(EventType.USAGE, {
                    "usage": {
                        "completion_tokens": chunk.usage.completion_tokens,
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }
                })
                return

            # Handle chunks with no choices (skip empty chunks)
            if not chunk.choices:
                return

            # Process the first choice (typically index 0)
            choice = chunk.choices[0]
            delta = choice.delta

            # Publish metadata if available
            metadata = {}
            if chunk.system_fingerprint:
                metadata["system_fingerprint"] = chunk.system_fingerprint
            if chunk.service_tier:
                metadata["service_tier"] = chunk.service_tier
            if chunk.id:
                metadata["id"] = chunk.id
            if chunk.created:
                metadata["created"] = chunk.created
            if chunk.model:
                metadata["model"] = chunk.model

            # if metadata:
            #     self._event_publisher.publish(EventType.METADATA, metadata)

            # Handle different types of delta content
            if delta.role:
                self._event_publisher.publish(EventType.ROLE, {
                    "role": delta.role
                })

            if delta.content:
                self._event_publisher.publish(EventType.CONTENT, {
                    "content": delta.content
                })

            # if delta.refusal:
            #     self._event_publisher.publish(EventType.REFUSAL, {
            #         "refusal": delta.refusal
            #     })

            # Handle tool calls
            # Use instance variables to accumulate tool call fragments by index
            # If we receive tool_calls, accumulate them
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    idx = tool_call.index
                    if idx not in self._tool_call_accumulator:
                        self._tool_call_accumulator[idx] = {
                            "index": idx,
                            "type": tool_call.type,
                            "id": getattr(tool_call, "id", None),
                            "function": {
                                "name": "",
                                "arguments": ""
                            }
                        }
                    # Update id if present
                    if tool_call.id:
                        self._tool_call_accumulator[idx]["id"] = tool_call.id
                    # Update function name/arguments if present
                    if tool_call.function:
                        if tool_call.function.name:
                            self._tool_call_accumulator[idx]["function"]["name"] = tool_call.function.name
                        if tool_call.function.arguments:
                            # Append arguments (they come in fragments)
                            self._tool_call_accumulator[idx]["function"]["arguments"] += tool_call.function.arguments

                # Set a flag to indicate we are in a tool call stream
                self._tool_call_streaming = True

            # If tool_calls is None and we were streaming, emit the event and reset
            else:
                if self._tool_call_streaming:
                    for tool_call_data in self._tool_call_accumulator.values():
                        self._event_publisher.publish(EventType.LLM_TOOL_CALL_REQUEST, {
                            "tool_call": tool_call_data
                        })
                    # Reset accumulator and flag
                    self._tool_call_accumulator = {}
                    self._tool_call_streaming = False

            # Handle deprecated function_call
            if delta.function_call:
                function_call_data = {}
                if delta.function_call.name:
                    function_call_data["name"] = delta.function_call.name
                if delta.function_call.arguments:
                    function_call_data["arguments"] = delta.function_call.arguments

                self._event_publisher.publish(EventType.LLM_TOOL_CALL_REQUEST, {
                    "function_call": function_call_data,
                    "deprecated": True
                })
            
            # Handle finish reason
            if choice.finish_reason:
                self._event_publisher.publish(EventType.FINISH, {
                    "finish_reason": choice.finish_reason
                })
                
        except Exception as e:
            logger.error(f"Error parsing chunk: {str(e)}")
            self._event_publisher.publish(EventType.ERROR, {
                "error": {
                    "message": f"Failed to parse chunk: {str(e)}",
                    "type": "chunk_parsing_error"
                }
            })

    async def check_health(self) -> Dict[str, Union[bool, str]]:
        """Check if OpenAI API is healthy and accessible.
        
        Returns:
            Dict[str, Union[bool, str]]: Health status information containing:
                - available (bool): Whether the service is available
                - message (str): Descriptive status message
                - models (List[str], optional): Available models if accessible
        """
        if not self._client:
            return {
                "available": False,
                "message": "OpenAI client not initialized"
            }
        
        try:
            # Test connection by listing available models
            models_response = await self._client.models.list()
            models = [model.id for model in models_response.data]
            
            return {
                "available": True,
                "message": f"OpenAI API healthy with {len(models)} models available",
                "models": models
            }
            
        except Exception as e:
            return {
                "available": False,
                "message": f"OpenAI API unavailable: {str(e)}"
            }

    def llm_to_hatchling_tool_call(self, event: Event) -> Optional[ToolCallParsedResult]:
        """Parse an OpenAI tool call event.

        Args:
            event (Event): The OpenAI tool call event.

        Returns:
            Optional[ToolCallParsedResult]: Normalized tool call result, or None if parsing fails.

        Raises:
            ValueError: If the event cannot be parsed as a valid OpenAI tool call.
        """
        if event.provider != self.provider_enum:
            raise ValueError(f"Event provider {event.provider} does not match OpenAI provider {self.provider_enum}")

        try:
            data = event.data
            
            # Handle OpenAI's deprecated function_call format
            if "function_call" in data and data.get("deprecated", False):
                function_call = data["function_call"]
                return ToolCallParsedResult(
                    tool_call_id="function_call",
                    function_name=function_call.get("name", ""),
                    arguments=self._llm_to_hatchling_tool_call_arguments(function_call.get("arguments", "{}"))
                )
            
            # Handle modern tool_call format (single complete tool call)
            if "tool_call" in data:
                tool_call = data["tool_call"]
                return ToolCallParsedResult(
                    tool_call_id=tool_call.get("id", "unknown"),
                    function_name=tool_call.get("function", {}).get("name", ""),
                    arguments=self._llm_to_hatchling_tool_call_arguments(tool_call.get("function", {}).get("arguments", "{}"))
                )
            
            raise ValueError("No valid tool call data found in OpenAI event")
            
        except Exception as e:
            logger.error(f"Error parsing OpenAI tool call: {e}")
            raise ValueError(f"Failed to parse OpenAI tool call: {e}")

    def _llm_to_hatchling_tool_call_arguments(self, args_str: str) -> Dict[str, Any]:
        """Parse tool call arguments from JSON string to dictionary.
        
        Args:
            args_str (str): JSON string of arguments.
            
        Returns:
            Dict[str, Any]: Parsed arguments as a dictionary.
        """
        if not args_str:
            return {}
            
        try:
            return json.loads(args_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse OpenAI tool call arguments: {args_str}")
            return {"_raw": args_str}
        
    def hatchling_to_llm_tool_call(self, tool_call: ToolCallParsedResult) -> Dict[str, Any]:
        """Convert a Hatchling tool call parsing result back to the OpenAI format.

        Args:
            tool_call (ToolCallParsedResult): The parsed tool call result.

        Returns:
            Dict[str, Any]: The tool call in OpenAI format.
        """
        return {
            "type": "function",
            "id": tool_call.tool_call_id,
            "function": {
                "name": tool_call.function_name,
                "arguments": json.dumps(tool_call.arguments)
            }
        }

    def mcp_to_provider_tool(self, tool_info: MCPToolInfo) -> Dict[str, Any]:
        """Convert an MCP tool to OpenAI function format.
        
        Args:
            tool_info (MCPToolInfo): MCP tool information to convert. This is an in/out
                                   parameter whose provider_format field will be set 
                                   to the converted tool format.
            
        Returns:
            Dict[str, Any]: Tool in OpenAI function format.
        """
        try:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_info.name,
                    "description": tool_info.description,
                    "parameters": tool_info.schema
                }
            }
            
            # Cache the converted format in the tool info
            tool_info.provider_format = openai_tool
            
            logger.debug(f"Converted tool {tool_info.name} to OpenAI format")
            return openai_tool
            
        except Exception as e:
            logger.error(f"Failed to convert tool {tool_info.name} to OpenAI format: {e}")
            return {}

    def hatchling_to_provider_tool_result(self, tool_result: ToolCallExecutionResult) -> Dict[str, Any]:
        """Convert a Hatchling tool call execution result to the OpenAI format.

        Args:
            tool_result (ToolCallExecutionResult): The tool call execution result.

        Returns:
            Dict[str, Any]: The result in OpenAI format.
        """
        return {
            "tool_call_id": tool_result.tool_call_id,
            "content": json.dumps({
                "tool_name": tool_result.function_name,
                "content": str(tool_result.result["content"][0]["text"]) if tool_result.result and "content" in tool_result.result and tool_result.result["content"] and tool_result.result["content"][0] and "text" in tool_result.result["content"][0] else "No result",
                "structuredContent": tool_result.result.get("structuredContent"),
                "isError": tool_result.result.get("isError", False)
            })
        }