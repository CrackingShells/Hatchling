"""Ollama provider implementation for LLM abstraction layer.

This module provides the OllamaProvider class that implements the LLMProvider interface
for the Ollama local inference server using the official ollama Python client.
"""

import json
import logging
import uuid
from typing import Dict, Any, List, Optional, Union
from ollama import AsyncClient

from hatchling.core.llm.providers.base import LLMProvider
from hatchling.core.llm.providers.registry import ProviderRegistry
from hatchling.config.settings import AppSettings
from hatchling.config.llm_settings import ELLMProvider
from hatchling.mcp_utils.mcp_tool_data import MCPToolInfo
from hatchling.core.llm.event_system.event_publisher import EventPublisher
from hatchling.core.llm.event_system.event_data import EventType
from hatchling.core.llm.event_system.event_subscribers_examples import Event
from hatchling.mcp_utils.mcp_tool_lifecycle_subscriber import ToolLifecycleSubscriber
from hatchling.core.llm.data_structures import ToolCallParsedResult, ToolCallExecutionResult
from hatchling.mcp_utils.manager import mcp_manager

logger = logging.getLogger(__name__)

@ProviderRegistry.register(ELLMProvider.OLLAMA)
class OllamaProvider(LLMProvider):
    """Ollama provider for local LLM inference.
    
    This provider uses the Ollama AsyncClient to communicate with a local or remote
    Ollama inference server. It supports streaming responses, tool calling, and
    multiple model architectures available through Ollama.
    """
    
    def __init__(self, settings: AppSettings = None):
        """Initialize the Ollama provider.
        
        Args:
            settings (AppSettings, optional): Application settings containing Ollama configuration.
                                            If None, uses the singleton instance.

        Raises:
            ValueError: If Ollama settings are invalid or missing required fields.
        """
        super().__init__(settings)
        self._client: Optional[AsyncClient] = None
        self.initialize()
        
        logger.debug(f"Initialized OllamaProvider with host: {self._settings.ollama.api_base}")
    
    @property
    def provider_name(self) -> str:
        """Return the provider name.
        
        Returns:
            str: The provider name "ollama" (from ELLMProvider enum).
        """
        return ELLMProvider.OLLAMA.value

    @property
    def provider_enum(self) -> ELLMProvider:
        """Return the provider enum.
        
        Returns:
            ELLMProvider: The enum value for Ollama provider.
        """
        return ELLMProvider.OLLAMA
    
    def initialize(self) -> None:
        """Initialize the Ollama async client and verify connection.
        
        Raises:
            ConnectionError: If unable to connect to Ollama server.
            ValueError: If configuration is invalid.
        """
        try:
            self._client = AsyncClient(host=self._settings.ollama.api_base)
            self._toolLifecycle_subscriber = ToolLifecycleSubscriber(ELLMProvider.OLLAMA.value, self.mcp_to_provider_tool)
            self._event_publisher = EventPublisher()
            mcp_manager.publisher.subscribe(self._toolLifecycle_subscriber)
            
            logger.info(f"Successfully connected to Ollama server at {self._settings.ollama.api_base}")
            
        except Exception as e:
            error_msg = f"Failed to initialize Ollama client: {str(e)}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
        
    def close(self) -> None:
        """Close the Ollama client connection and clean up resources.
        
        This method should be called when the provider is no longer needed.
        It will close the AsyncClient connection gracefully.
        """
        mcp_manager.publisher.unsubscribe(self._toolLifecycle_subscriber)
        self._event_publisher.clear_subscribers()
    
    def prepare_chat_payload(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare chat payload for Ollama API format.
        
        Args:
            messages (List[Dict[str, Any]]): List of message dictionaries.
            model (str): Model name to use.
            **kwargs: Additional chat parameters.
        
        Returns:
            Dict[str, Any]: Formatted payload for Ollama chat API.
        """
        # Base payload structure
        payload = {
            "model": model,
            "messages": messages,
            "stream": kwargs.get("stream", True)
        }
        
        # Add optional parameters if provided
        optional_params = [
            "format", "template", "system", "context", 
            "raw", "keep_alive"
        ]
        
        for param in optional_params:
            if param in kwargs:
                payload[param] = kwargs[param]
        
        # Build options from settings with kwargs override
        options = {}
        
        # Map settings to Ollama options
        settings_mapping = {
            "num_ctx": self._settings.ollama.num_ctx,
            "repeat_last_n": self._settings.ollama.repeat_last_n,
            "repeat_penalty": self._settings.ollama.repeat_penalty,
            "temperature": self._settings.ollama.temperature,
            "seed": self._settings.ollama.seed,
            "num_predict": self._settings.ollama.num_predict,
            "top_k": self._settings.ollama.top_k,
            "top_p": self._settings.ollama.top_p,
            "min_p": self._settings.ollama.min_p,
        }
        
        if self._settings.ollama.stop:
            options["stop"] = self._settings.ollama.stop
        
        # Apply settings defaults
        for key, value in settings_mapping.items():
            options[key] = kwargs.get(key, value)
        
        # Map common parameters to Ollama options with kwargs override
        param_mapping = {
            "max_tokens": "num_predict",
        }
        
        for param, ollama_param in param_mapping.items():
            if param in kwargs:
                options[ollama_param] = kwargs[param]
        
        if options:
            payload["options"] = options
        
        logger.debug(f"Prepared Ollama chat payload for model: {model}")
        return payload
    
    def add_tools_to_payload(
        self, 
        payload: Dict[str, Any], 
        tools: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Add tool definitions to the Ollama chat payload.
        
        Args:
            payload (Dict[str, Any]): Base chat payload.
            tools (Optional[List[str]]): Specific list of tool names to add to the payload. If None, all available and enabled tools are added.
                                         If provided, only the specified tools that are enabled will be added to the payload.
        Returns:
            Dict[str, Any]: Updated payload with tools.
        """
        
        # Convert tools to Ollama format
        ollama_tools = []
        all_tools = self._toolLifecycle_subscriber.get_all_tools()
        enabled_tools = self._toolLifecycle_subscriber.get_enabled_tools()

        # If no specific tools provided, use all enabled tools
        if tools is None:
            ollama_tools = [
                tool.provider_format for tool in enabled_tools.values()
            ]

        else:
            for tool_name in tools:
                # Ensure the function definition exists in the tool cache
                if not tool_name in all_tools:
                    error_msg = f"Function definition for {tool_name} not found in tool cache"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                if not tool_name in enabled_tools:
                    warning_msg = f"Function {tool_name} is disabled with reason: {self._toolLifecycle_subscriber.prettied_reason(all_tools[tool_name].reason)}" 
                    logger.warning(warning_msg)
                    continue #skipping disabled tools
                
                ollama_tools.append(enabled_tools[tool_name].provider_format)
        
        if ollama_tools:
            # Add tools to payload
            payload["tools"] = ollama_tools
            logger.debug(f"Added {len(ollama_tools)} tools to Ollama payload")

        return payload
    
    async def stream_chat_response(
        self, 
        payload: Dict[str, Any]
    ):
        """Stream chat response from Ollama with publish-subscribe events.
        
        Args:
            payload (Dict[str, Any]): Chat request payload.
        
        Raises:
            RuntimeError: If client is not initialized.
            Exception: If streaming fails.
        """
        if not self._client:
            raise RuntimeError("Ollama client not initialized. Call initialize() first.")
        
        try:
            # Given Ollama's IP can be configured in settings, we ensure the client is set up correctly
            # TODO: This shouldn't be here because it means we are re-initializing the client every time
            #       Probably should be changed upon setting change.
            self._client = AsyncClient(host=self._settings.ollama.api_base)

            # Ensure streaming is enabled for this request
            payload["stream"] = True

            # Generate a unique request ID for this streaming session
            request_id = str(uuid.uuid4())
            self._event_publisher.set_request_id(request_id)
            
            # Stream the response
            response_stream = await self._client.chat(**payload)
            async for chunk in response_stream:
                # Parse chunk and publish events to subscribers
                self._parse_and_publish_chunk(chunk)
                    
        except Exception as e:
            error_msg = f"Error streaming from Ollama: {str(e)}"
            logger.error(error_msg)
            
            # Publish error event
            self._event_publisher.publish(
                EventType.ERROR,
                {
                    "error": {"message": error_msg, "type": "ollama_streaming_error"},
                    "request_id": request_id
                }
            )
            raise
    
    
    def _parse_and_publish_chunk(self, chunk: Any) -> None:
        """Parse Ollama chunk and publish events to subscribers.
        
        Args:
            chunk (Any): Raw chunk from Ollama API. Typically a Dict[str, Any] with keys like "message", "done", etc.
            
        Returns:
            Any: Standardized chunk format.
        """
        try:
            
            # Handle content (message delta)
            if "message" in chunk:
                message = chunk["message"]
                if "role" in message:
                    # Publish role event
                    self._event_publisher.publish(
                        EventType.ROLE,
                        {"role": message["role"]}
                    )
                
                if "content" in message and message["content"]:
                    content = message["content"]
                    # Publish content event
                    self._event_publisher.publish(
                        EventType.CONTENT,
                        {"content": content}
                    )
            
            # Handle completion state
            if chunk.get("done", False):
                # Publish finish event
                finish_reason = chunk.get("done_reason", "stop")
                self._event_publisher.publish(
                    EventType.FINISH,
                    {"finish_reason": finish_reason}
                )
                
                # Handle usage stats if available in final chunk
                if "eval_count" in chunk or "prompt_eval_count" in chunk:
                    usage = {
                        "prompt_tokens": chunk.get("prompt_eval_count", 0),
                        "completion_tokens": chunk.get("eval_count", 0),
                        "total_tokens": chunk.get("prompt_eval_count", 0) + chunk.get("eval_count", 0)
                    }
                    # Publish usage event
                    self._event_publisher.publish(
                        EventType.USAGE,
                        {"usage": usage}
                    )
            
            # Handle tool calls (if supported in future Ollama versions)
            if "message" in chunk and "tool_calls" in chunk["message"]:
                tool_calls = chunk["message"]["tool_calls"]
                # Publish tool calls
                self._event_publisher.publish(
                    EventType.LLM_TOOL_CALL_REQUEST,
                    {"tool_calls": tool_calls}
                )
            
        except Exception as e:
            error_msg = f"Error parsing Ollama chunk: {str(e)}"
            logger.error(error_msg)
            
            # Publish error event
            self._event_publisher.publish(
                EventType.ERROR,{"error": {"message": error_msg, "type": "chunk_parsing_error"}}
            )
    

    async def check_health(self) -> Dict[str, Union[bool, str]]:
        """Check if Ollama server is healthy and accessible.
        
        Returns:
            Dict[str, Union[bool, str]]: Health status information containing:
                - available (bool): Whether the service is available
                - message (str): Descriptive status message
                - models (List[str], optional): Available models if accessible
        """
        if not self._client:
            return {
                "available": False,
                "message": "Ollama client not initialized"
            }
        
        try:
            # Test connection by listing available models
            models_response = await self._client.list()
            models = [model.get("name", "unknown") for model in models_response.get("models", [])]
            
            return {
                "available": True,
                "message": f"Ollama server healthy with {len(models)} models available",
                "models": models
            }
            
        except Exception as e:
            return {
                "available": False,
                "message": f"Ollama server unavailable: {str(e)}"
            }

    def llm_to_hatchling_tool_call(self, event: Event) -> Optional[ToolCallParsedResult]:
        """Parse an Ollama tool call event.

        Args:
            event (Event): The Ollama tool call event.

        Returns:
            Optional[ToolCallParsedResult]: Normalized tool call result, or None if parsing fails.

        Raises:
            ValueError: If the event cannot be parsed as a valid Ollama tool call.
        """
        if event.provider != self.provider_enum:
            raise ValueError(f"Event provider {event.provider} does not match Ollama provider {self.provider_enum}")

        try:
            tool_calls = event.data.get("tool_calls", [])
            if not tool_calls:
                raise ValueError("No tool calls found in Ollama event")
                
            # Process the first tool call (if multiple, we'd need to call llm_to_hatchling_tool_call for each)
            tool_call = tool_calls[0]
            
            # Extract standard fields
            tool_id = tool_call.get("id", str(uuid.uuid4()))  # Use a UUID if no ID is provided
            function_name = tool_call.get("function", {}).get("name", "")
            arguments = tool_call.get("function", {}).get("arguments", {})
            
            # Ensure arguments is a dictionary
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse Ollama tool call arguments: {arguments}")
                    arguments = {"_raw": arguments}
            
            return ToolCallParsedResult(
                tool_call_id=tool_id,
                function_name=function_name,
                arguments=arguments
            )
            
        except Exception as e:
            logger.error(f"Error parsing Ollama tool call: {e}")
            raise ValueError(f"Failed to parse Ollama tool call: {e}")
        
    def hatchling_to_llm_tool_call(self, tool_call: ToolCallParsedResult) -> Dict[str, Any]:
        """Convert an LLM tool call parsing result back to the Ollama format.

        Args:
            tool_call (ToolCallParsedResult): The parsed tool call result.

        Returns:
            Dict[str, Any]: The tool call in Ollama format.
        """
        return {
            "type": "function",
            "id": tool_call.tool_call_id,
            "function": {
                "name": tool_call.function_name,
                "arguments": tool_call.arguments
            }
        }

    def mcp_to_provider_tool(self, tool_info: MCPToolInfo) -> Dict[str, Any]:
        """Convert an MCP tool to Ollama function format.

        Args:
            tool_info (MCPToolInfo): MCP tool information to convert. This is an in/out
                                   parameter whose provider_format field will be set 
                                   to the converted tool format.
            
        Returns:
            Dict[str, Any]: Tool in Ollama function format.
        """
        try:
            # Ollama uses a similar format to OpenAI but with some differences
            ollama_tool = {
                "type": "function",
                "function": {
                    "name": tool_info.name,
                    "description": tool_info.description,
                    "parameters": tool_info.schema
                }
            }
            
            # Cache the converted format in the tool info
            tool_info.provider_format = ollama_tool
            
            logger.debug(f"Converted tool {tool_info.name} to Ollama format")
            return ollama_tool
            
        except Exception as e:
            logger.error(f"Failed to convert tool {tool_info.name} to Ollama format: {e}")
            return {}
    
    def hatchling_to_provider_tool_result(self, tool_result: ToolCallExecutionResult) -> Dict[str, Any]:
        """Convert the result to a dictionary suitable for Ollama API.

        Args:
            tool_result (ToolCallExecutionResult): The tool call execution result.

        Returns:
            Dict[str, Any]: The result in Ollama format.
        """

        return {
            "content": json.dumps({
                "tool_name": tool_result.function_name,
                "content": str(tool_result.result["content"][0]["text"]) if tool_result.result and "content" in tool_result.result and tool_result.result["content"] and tool_result.result["content"][0] and "text" in tool_result.result["content"][0] else "No result",
                "structuredContent": tool_result.result.get("structuredContent"),
                "isError": tool_result.result.get("isError", False)
            })
        }
